"""Utilities for saving/loading data such as dealing with changes to conventions, etc."""
from copy import copy
from glob import glob
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Optional

import numpy as np

from cmcode import caiman_analysis as cma, caiman_params as cmp
from cmcode.util.paths import normalize_path, params_file_for_result

from caiman.source_extraction.cnmf.params import CNMFParams


def reconstruct_sessdata_obj(sessdata: 'cma.SessionAnalysis', loaded_info: dict[str, Any]):
    """to be called from the SessionAnalysis constructor with loaded_fields passed in"""
    _populate_missing_fields(loaded_info)
    _set_params(sessdata, loaded_info)
    _set_fields(sessdata, loaded_info)
    _fix_field_types(sessdata)
    _fix_tif_field_on_load(sessdata)
    _fix_mc_field_on_load(sessdata)
    _fix_cnmf_fields_on_load(sessdata)
    _validate_or_write_missing_params_files(sessdata)


def _populate_missing_fields(loaded_info: dict[str, Any]):
    """Set missing fields to what they would have been before these fields were added"""
    if 'mmap_file_transposed' not in loaded_info:
        if (mc_result := loaded_info.get('mc_result')) is not None:
            # bypass __getattribute__ which will throw an AttributeError
            loaded_info['mmap_file_transposed'] = object.__getattribute__(mc_result, 'mmap_file_transposed')
        else:
            loaded_info['mmap_file_transposed'] = None
    
    if 'rec_type' not in loaded_info:
        if 'data_dir' in loaded_info and loaded_info['data_dir']:
            loaded_info['rec_type'] = os.path.split(os.path.split(loaded_info['data_dir'])[0])[1]
        else:
            loaded_info['rec_type'] = None


def _set_params(sessdata: 'cma.SessionAnalysis', loaded_info: dict[str, Any]):
    # find or create SessionAnalysisParams object, then populate with fields
    if 'params' in loaded_info:
        params: cmp.SessionAnalysisParams = loaded_info.pop('params')
        sessdata.params = params

    elif 'cnmf_params' in loaded_info:
        # re-create params object from dict for backward compatibility
        cnmf_params: CNMFParams = loaded_info.pop('cnmf_params')
        
        # update other parameters from loaded fields as needed
        conv_changes = {}
        for conv_field in ['odd_row_offset', 'downsample_factor', 'crop']:
            if conv_field in loaded_info:
                conv_changes[conv_field] = loaded_info.pop(conv_field)

        if 'odd_row_ndeads' in loaded_info:
            conv_changes['odd_row_ndead'] = loaded_info.pop('odd_row_ndeads')
        
        if (rec_type := loaded_info.get('rec_type')) and rec_type in ('learning_ppc_dlx', 'dlx_calibration'):
            # set channel for structural rec types to 1
            conv_changes['channel'] = 1
        
        conversion = cmp.ConversionParams(**conv_changes)

        # old method was to auto adjust indices only if rigid shifts were used
        mcorr_extra_changes = {}
        if loaded_info.get('mc_result') and not cnmf_params.motion['pw_rigid']:
            mcorr_extra_changes['_indices_are_adjusted'] = True
        
        mcorr_extra = cmp.McorrParamsExtra(**mcorr_extra_changes)

        trans_changes = {}
        if 'highpass_cutoff' in loaded_info:
            trans_changes['highpass_cutoff'] = loaded_info.pop('highpass_cutoff')
        
        transposition = cmp.TranspositionParams(**trans_changes)

        cnmf_extra_changes = {}
        if 'crossplane_merge_thr' in loaded_info:
            cnmf_extra_changes['crossplane_merge_thr'] = loaded_info.pop('crossplane_merge_thr')

        # try to infer seed params from the existing seed file, if any
        if 'cnmf_fit_filename' in loaded_info:
            sessdata.cnmf_fit_filename = normalize_path(loaded_info.pop('cnmf_fit_filename'))
        if sessdata.cnmf_fit_filename is not None:
            # look for Ain... file in same directory (should be mesmerize directory)
            logging.info('Trying to infer seed params from seed file...')
            cnmf_run_dir = os.path.split(sessdata.cnmf_fit_filename)[0]
            matching_files = glob('Ain_*.npy', root_dir=cnmf_run_dir)
            if len(matching_files) == 1:
                seed_path = os.path.join(cnmf_run_dir, matching_files[0])
                cnmf_extra_changes['seed_params'] = cmp.SeedParams.infer_from_seed_path(seed_path)
                logging.info('Successfully inferred seed params')
            elif len(matching_files) == 0:
                logging.info('No seed file found; assuming no seed was used')
            else:
                logging.warning('Failed to infer seed params: multiple seed files found')
        
        if 'seed_params' not in cnmf_extra_changes and 'seed_params' in loaded_info:
            # saved seed params take lower precedence. small chance it does not match CNMF results.
            logging.info('Loading seed params saved in SessionAnalysis object')
            cnmf_extra_changes['seed_params'] = loaded_info.pop('seed_params')
        
        cnmf_extra = cmp.CNMFParamsExtra(**cnmf_extra_changes)
                
        eval_extra_changes = {}
        if 'snr_type' in loaded_info:
            eval_extra_changes['snr_type'] = loaded_info.pop('snr_type')
        else:
            # try to infer from CNMF estimates
            if 'cnmf_fit_filename' in loaded_info:
                sessdata.cnmf_fit_filename = normalize_path(loaded_info.pop('cnmf_fit_filename'))
            if sessdata.cnmf_fit is not None:
                # set snr type based on whether gamma SNR values are populated
                snr_type = 'normal' if sessdata.cnmf_fit.estimates.snr_gamma_vals is None else 'gamma'
                eval_extra_changes['snr_type'] = snr_type

        eval_extra = cmp.EvalParamsExtra(**eval_extra_changes)

        sessdata.params = cmp.SessionAnalysisParams.from_cnmf_params(
            cnmf=cnmf_params, conversion=conversion, mcorr_extra=mcorr_extra, transposition=transposition, 
            cnmf_extra=cnmf_extra, eval_extra=eval_extra
        )
    else:
        raise ValueError('Cannot construct SessionAnalysis without either params or cnmf_params field')


def _set_fields(sessdata: 'cma.SessionAnalysis', loaded_info: dict[str, Any]):

    # obsolete fields that we don't care about anymore
    fields_to_discard = ['structural_sbx_files', 'structural_offset', 'structural_tif_file', 'tag_base',
                         'cnmf_fit1', 'cnmf_fit1_filename', 'cnmf_fit2', 'image_dir', 'gridsearch_batch_path']
    
    for key, val in loaded_info.items():
        if key in fields_to_discard:
            continue

        if key in cma.SessionAnalysis.PATH_FIELDS:
            val = normalize_path(val)

        if key == 'frames_per_trial':
            key = '_frames_per_trial'

        setattr(sessdata, key, val)


def _fix_field_types(sessdata: 'cma.SessionAnalysis'):
    if isinstance(sessdata._frames_per_trial, list):
        # better to avoid lists for exporting
        sessdata._frames_per_trial = np.array(sessdata._frames_per_trial)


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
            # need motion params to reconstruct MotionCorrect object
            sessdata.mc_result.motion_params = copy(sessdata.params._cnmf.motion)


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


def _validate_or_write_missing_params_files(sessdata: 'cma.SessionAnalysis'):
    """
    Validate existing params files and/or write missing params files for result files.
    
    For most results, the exact parameters that were used are not saved outside of params files,
    so we just have to assume when loading the first time after the SessionAnalysisParams feature
    was added that we were diligent about re-running stages after changing relevant parameters.

    For CNMF, we can check against the params in mesmerize, and if they don't match,
    write the params that were actually used, then invalidate according to what doesn't match.
    """
    def check_file(file_path: str, stage: cmp.AnalysisStage, file_desc: str) -> Optional[bool]:
        """
        Check one file, invalidating stage and returning False if params don't match
        and returning None and writing out params if no params file was found.
        """
        try:
            nonmatching_params = list(sessdata.get_nonmatching_params_for_result_file(file_path, stage))
            if len(nonmatching_params) > 0:
                logging.warning(file_desc + ' had non-matching parameters: ' + ', '.join(nonmatching_params))
                sessdata.invalidate_from_stage(stage)
                return False
        except FileNotFoundError:
            # assume current params are correct
            logging.info('Writing missing parameters for ' + file_desc)
            sessdata.write_params_for_result_file(file_path, stage)
        return True

    if sessdata.plane_tifs is not None:
        for k_plane, plane_file in enumerate(sessdata.plane_tifs):
            file_desc = f'plane {k_plane} TIF file'
            valid = check_file(plane_file, cmp.AnalysisStage.CONVERT, file_desc)
            if valid is False:
                return
            
    # to know where to invalidate if motion params saved in CNMF don't match
    # we assume the farthest back we will have to go is the conversion step
    last_validated_stage = cmp.AnalysisStage.CONVERT

    if sessdata.mc_result is not None:
        for k_plane, mcorr_file in enumerate(sessdata.mc_result.mmap_files):
            file_desc = f'plane {k_plane} motion correction'
            valid = check_file(mcorr_file, cmp.AnalysisStage.MCORR, file_desc)
            if valid is True:
                last_validated_stage = cmp.AnalysisStage.MCORR
            elif valid is False:
                return
    
    if sessdata.mmap_file_transposed is not None:
        valid = check_file(sessdata.mmap_file_transposed, cmp.AnalysisStage.TRANSPOSE,
                             'transposed/concatenated mmap file')
        if valid is True:
            last_validated_stage = cmp.AnalysisStage.TRANSPOSE
        elif valid is False:
            return
    
    if sessdata.cnmf_fit_filename is not None:
        try:
            if not sessdata.result_file_matches_params(
                sessdata.cnmf_fit_filename, cmp.AnalysisStage.CNMF, raise_on_missing_params=True):
                logging.warning('CNMF had non-matching parameters')
                sessdata.invalidate_from_stage(cmp.AnalysisStage.CNMF)
        except FileNotFoundError:
            logging.info('Params file not found for CNMF')
            # find params from the CNMF object itself and write missing params file
            # assume here that any params not saved with CNMF match what is in the object.
            cnmf_params = cmp.load_params_from_cnmf_h5(sessdata.cnmf_fit_filename)
            cnmf_params_dict = cnmf_params.to_dict()
            # make each subparam really a dict (not arbitrary mapping) to get around prohibition
            # in change_params function (meant to prevent accidentally changing more params than intended)
            dict_of_dicts = {group: {**subparams} for group, subparams in cnmf_params_dict.items()}

            # write out params for this CNMF run and get stage of mismatch
            cnmf_run_params, invalid_stage = sessdata.params.change_params_and_get_stage_to_invalidate(
                dict_of_dicts, metadata=sessdata.metadata
            )
            params_file = params_file_for_result(sessdata.cnmf_fit_filename)
            logging.info('Writing missing parameters for CNMF')
            cnmf_run_params.write_params(params_file, stage=cmp.AnalysisStage.EVAL)

            if invalid_stage is None:
                # It matches, we can update our params (only things that don't matter for comparison)
                logging.info('CNMF batch item was saved with compatible parameters - updating params')
                sessdata.params = cnmf_run_params
            else:
                # don't just invalidate this stage, because we're doing something different here:
                # we know the given stage doesn't match the CNMF results, not our current params.
                # If we have params files for previous stages, we know those results do match our params,
                # so we can keep them. But otherwise, it's ambiguous what happened, so
                # invalidate from where there is a mismatch to be safe.
                logging.warning('CNMF batch item was saved with incompatible parameters - invalidating')
                sessdata.invalidate_from_stage(cmp.AnalysisStage(max(invalid_stage, last_validated_stage)))


