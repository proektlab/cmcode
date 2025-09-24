"""Utilities for saving/loading data such as dealing with changes to conventions, etc."""
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from cmcode import caiman_analysis as cma
from cmcode.util.image import BorderSpec
from cmcode.util.paths import normalize_path

from caiman.source_extraction.cnmf.params import CNMFParams


def reconstruct_sessdata_obj(sessdata: 'cma.SessionAnalysis', loaded_info: dict[str, Any]):
    """to be called from the SessionAnalysis constructor with loaded_fields passed in"""
    if 'cnmf_params' not in loaded_info:
        raise ValueError('Cannot load SessionAnalysis without saved params')
    _set_fields(sessdata, loaded_info)
    _populate_missing_fields(sessdata)
    _fix_field_types(sessdata)
    _fix_tif_field_on_load(sessdata)
    _fix_mc_field_on_load(sessdata)
    _fix_cnmf_fields_on_load(sessdata)


def _set_fields(sessdata: 'cma.SessionAnalysis', loaded_info: dict[str, Any]):
    caiman_logger = logging.getLogger('caiman')

    # obsolete fields that we don't care about anymore
    fields_to_discard = ['structural_sbx_files', 'structural_offset', 'structural_tif_file',
                         'cnmf_fit1', 'cnmf_fit1_filename', 'cnmf_fit2', 'image_dir']

    for key, val in loaded_info.items():
        if key in fields_to_discard:
            continue
        if key in cma.SessionAnalysis.PATH_FIELDS:
            val = normalize_path(val)

        if key == 'cnmf_params' and isinstance(val, CNMFParams):
            # create a new params object and set each sub-dict for backward compatibility
            sessdata.cnmf_params = CNMFParams()
            params_dict = val.to_dict()
            
            # don't log each loaded parameter
            old_level = caiman_logger.level
            caiman_logger.setLevel(logging.WARNING)

            for subdict_key, subdict in params_dict.items():
                sessdata.cnmf_params.set(subdict_key, subdict)
            
            caiman_logger.setLevel(old_level)
        else:
            setattr(sessdata, key, val)


def _populate_missing_fields(sessdata: 'cma.SessionAnalysis'):
    """Set missing fields to what they would have been before these fields were added"""
    if not hasattr(sessdata, 'snr_type') and sessdata.cnmf_fit is not None:
        # set snr type based on whether gamma SNR values are populated
        sessdata.snr_type = 'normal' if sessdata.cnmf_fit.estimates.snr_gamma_vals is None else 'gamma'
    
    if not hasattr(sessdata, 'tag_base'):
        sessdata.tag_base = sessdata.tag

    if not hasattr(sessdata, 'crossplane_merge_thr'):
        sessdata.crossplane_merge_thr = None
    
    if not hasattr(sessdata, 'downsample_factor'):
        sessdata.downsample_factor = None

    if not hasattr(sessdata, 'crop'):
        sessdata.crop = BorderSpec()


def _fix_field_types(sessdata: 'cma.SessionAnalysis'):
    if isinstance(sessdata.frames_per_trial, list):
        # better to avoid lists for exporting
        sessdata.frames_per_trial = np.array(sessdata.frames_per_trial)


def _fix_tif_field_on_load(sessdata: 'cma.SessionAnalysis'):
    plane_tifs = sessdata.plane_tifs
    if plane_tifs is None:
        return

    new_plane_tifs = []
    for plane_tif in plane_tifs:
        # add subdir of "conversion" to file if it's not present
        file_dir, filename = os.path.split(plane_tif)
        _, last_subdir = os.path.split(file_dir)
        if last_subdir == 'conversion':
            new_plane_tifs.append(plane_tif)
            continue
        
        new_file_dir = os.path.join(file_dir, 'conversion')
        os.makedirs(new_file_dir, exist_ok=True)

        new_path = os.path.join(new_file_dir, filename)
        if os.path.exists(plane_tif):
            if os.path.exists(new_path):
                # ambiguous - just use file at new path but issue warning
                logging.warning(f'Converted file {filename} exists in both old ({file_dir}) and new ({new_file_dir}) locations. '
                                'Using file at new location; consider deleting one of them to avoid confusion.')
            else:
                # move existing file to conversion dir
                shutil.move(plane_tif, new_path)
                logging.info(f'Moved {plane_tif} into conversion subdirectory')
        new_plane_tifs.append(new_path)
    sessdata.plane_tifs = new_plane_tifs


def _fix_mc_field_on_load(sessdata: 'cma.SessionAnalysis'):
    """fix motion correction results if necessary (so els holoview can be retrieved)"""
    if sessdata.mc_result is not None:
        if sessdata.mc_result.dims is None:
            sessdata.mc_result.dims = sessdata.plane_size
        if sessdata.mc_result.motion_params is None:
            sessdata.mc_result.motion_params = sessdata.cnmf_params.motion


def _fix_cnmf_fields_on_load(sessdata: 'cma.SessionAnalysis'):
    """
    Does a couple of things:
    - moves deprecated 'cnmf_fit2_filename' to 'cnmf_fit_filename'
    - change root data dir to cnmf subdir in cnmf_fit_filename
    - if the actual CNMF fit file is in the root data dir, moves it to the 'cnmf' subdir
        (only relevant for non-mesmerize runs, which are deprecated)
    """
    if hasattr(sessdata, 'cnmf_fit2_filename'):
        # only use if we don't have a cnmf_fit_filename to use
        if not hasattr(sessdata, 'cnmf_fit_filename') or sessdata.cnmf_fit_filename is None:
            sessdata.cnmf_fit_filename = getattr(sessdata, 'cnmf_fit2_filename')
        delattr(sessdata, 'cnmf_fit2_filename')            

    for cnmf_path_field in ['cnmf_fit_filename', 'gridsearch_batch_path']:
        if (old_fn := getattr(sessdata, cnmf_path_field)) is not None:
            cnmf_filepath = Path(old_fn)
            if not cnmf_filepath.is_relative_to(sessdata.data_dir):
                return

            # change path to add cnmf subdir
            rel_path = cnmf_filepath.relative_to(sessdata.data_dir)
            if rel_path.parts[0] != 'cnmf':
                new_fn = Path(sessdata.data_dir) / 'cnmf' / rel_path
                setattr(sessdata, cnmf_path_field, str(new_fn))

                # move to subdir only if it was in the root data dir
                if cnmf_filepath.parent.resolve() == Path(sessdata.data_dir).resolve():
                    os.makedirs(os.path.join(sessdata.data_dir, 'cnmf'), exist_ok=True)
                    
                    # move file if it exists
                    if cnmf_filepath.exists():
                        if new_fn.exists():
                            # ambiguous - just use file at new path but issue warning
                            logging.warning(f'File {cnmf_filepath.name} exists in both root data dir ({sessdata.data_dir}) and cnmf subdir. '
                                            'Using file at new location; consider deleting one of them to avoid confusion.')
                        else:
                            shutil.move(old_fn, str(new_fn))
                            logging.info(f'Moved {old_fn} into cnmf subdirectory')