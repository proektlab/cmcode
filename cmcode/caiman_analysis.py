import asyncio
from collections import Counter
from collections.abc import Container, Iterable, Sequence, Mapping
from copy import copy
from dataclasses import asdict
from datetime import date, datetime
from functools import lru_cache, partial
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
from subprocess import CalledProcessError
from typing import Optional, Union, Any, cast, Literal, overload, TYPE_CHECKING

import cv2
from hdf5storage import savemat
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import optype.numpy as onp
import pandas as pd
from pandas._libs.missing import NAType
from scipy import sparse

import caiman as cm
from caiman.base.movies import get_file_size, load_iter
from caiman.base import rois
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.utils import sbx_utils
from caiman.utils.visualization import view_quilt
import mesmerize_core as mc
from mesmerize_core.algorithms._utils import make_projection_parallel, make_correlation_parallel
from mesmerize_core.caiman_extensions.common import Waitable, DummyProcess

from cmcode import in_jupyter, alignment, cmcustom, gridsearch_analysis, mcorr, cnmf_ext, caiman_params as cmp
from cmcode.caiman_params import SessionAnalysisParams, AnalysisStage
from cmcode.remote import remoteops
from cmcode.gridsearch_analysis import ParamGrid
# from cmcode.mcorr import MCResult, PiecewiseMCInfo  # to allow unpickling
from cmcode.util import footprints, paths
from cmcode.util.cluster import Cluster
from cmcode.util.compat import reconstruct_sessdata_obj
from cmcode.util.image import make_merge, remap_image, BorderSpec, preprocess_proj_for_seed
from cmcode.util.sbx_data import find_sess_sbx_files, get_trial_numbers_from_files
from cmcode.util.scaled import ScaledDataFrame, make_um_df, make_pixel_df
from cmcode.util.types import NoBatchFileError, NoMatchingResultError, MescoreBatch, MescoreSeries

if TYPE_CHECKING or in_jupyter():
    # avoid importing caiman_viz if not in a notebook
    from cmcode import caiman_viz
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# exception identification
NO_FILES_MSG = 'No .sbx files found'

# this is a global variable
cluster = Cluster()

def setup_cluster(*args, **kwargs):
    """Start the global cluster (or initialize with 'single' backend to not use a cluster)"""
    logging.warning('cma.setup_cluster is deprecated, use cma.cluster.start instead')
    cluster.start(*args, **kwargs)


def get_mouse_params(mouse_id: Union[int, str]) -> dict[str, dict[str, Any]]:
    """Get mouse-specific parameter overrides"""
    root_data_dir = paths.get_root_data_dir()
    params_filename = root_data_dir / 'mouse_params' / f'params_{mouse_id}.json'
    if params_filename.exists():
        with open(params_filename, 'r') as json_fh:
            mouse_params: dict[str, dict[str, Any]] = json.load(json_fh)
    else:
        mouse_params = {}
    return mouse_params


def save_contour_plot_as_pdf(fig: Figure, filename):
    """Save contour plot figure to a high-resolution PDF"""
    fig.set_figwidth(fig.get_figwidth() * 15)
    fig.set_figheight(fig.get_figheight() * 15)
    fig.savefig(filename)


def get_projection_name(seed_params: cmp.SeedParams) -> str:
    """
    Make a string to identify the type of projection
    (like get_spatial_seed_name but ignores parameters only used for identifying neurons)
    """
    proj_name = seed_params.type
    if seed_params.blur_size > 1:
        proj_name += f'_blur_{seed_params.blur_size}'
    if seed_params.norm_medw is not None:
        proj_name += f'_medw_{seed_params.norm_medw}'
    
    return proj_name


def get_spatial_seed_name(seed_params: cmp.SeedParams) -> str:
    """Make a string to identify the type of seed (used as filename)"""
    proj_name = get_projection_name(seed_params)

    Ain_name = f'Ain_caiman_from_{proj_name}'
    if seed_params.gSig is not None:  # just leave out of name if default
        gSig = seed_params.gSig
        if isinstance(gSig, int) or not hasattr(gSig, '__len__'):
            gSig = [gSig]
        else:
            gSig = list(gSig)
        Ain_name += '_gSig_' + ','.join(str(gs) for gs in gSig)

    if seed_params.blur_gSig_multiple is not None:
        Ain_name += f'_blurmult_{seed_params.blur_gSig_multiple:.2f}'
    return Ain_name


def get_batch_for_session(mouse_id: Union[int, str] = 0, sess_id=0, rec_type='learning_ppc',
                          tag: Optional[str] = None, data_dir: Optional[str] = None, create=True) -> MescoreBatch:
    """
    Locate directory to save dataframe and make the dataframe with name corresponding to
    this mouse/session/[tag], or load it if it already exists.
    If data_dir is passed directly, rec_type is ignored.
    If create is False, errors instead of creating a new batch file.
    """
    if data_dir is None:
        data_dir = paths.get_processed_dir(mouse_id, rec_type=rec_type, create_if_not_found=create)

    tagstr = '_' + tag if tag is not None else ''
    batch_name = f'{mouse_id}_{sess_id:03d}{tagstr}_batch.pickle'
    batch_path = Path(data_dir) / 'cnmf' / batch_name
    if batch_path.exists():
        return cast(MescoreBatch, mc.load_batch(batch_path))
    elif (old_batch_path := Path(data_dir) / batch_name).exists():
        # move from old location
        os.makedirs(batch_path.parent, exist_ok=True)
        shutil.move(old_batch_path, batch_path)
        logging.info(f'Moved {str(old_batch_path)} into cnmf subdirectory')
        return cast(MescoreBatch, mc.load_batch(batch_path))
    elif create:
        return cast(MescoreBatch, mc.create_batch(batch_path))
    else:
        raise NoBatchFileError()


class SessionAnalysis:
    PATH_FIELDS = ['image_dir', 'data_dir', 'sbx_files', 'plane_tifs',
                   'mc_result', 'mmap_file_transposed', 'cnmf_fit2_filename', 'cnmf_fit_filename']

    def __init__(self, mouse_id: Union[int, str] = 0, sess_id=0, trials_to_include: Optional[Sequence[int]] = None, *,
                 rec_type='learning_ppc', channel=0, tag: Optional[str] = None, loaded_info: Optional[dict[str, Any]] = None,
                 sess_filename: Optional[str] = None, param_overrides: Optional[dict[str, dict[str, Any]]] = None,
                 downsample_factor: Optional[int] = None, trials_to_exclude: Optional[Sequence[int]] = None,
                 highpass_cutoff=0.):
        """
        Make an object to manage analyzing one session (or part of a session, if trials_to_include is given)
        If loaded_info is not none, ignores all other arguments and constructs from the passed dict.
        downsample_factor can be given to downsample data and then interpolate at the end. A string f'_ds{downsample_factor}'
        is appended to the tag. If the tag excluding this string matches an existing run, the corrected/converted data from that run
        may be used as a basis for saving the downsampled tif.
        """
        # initialize fields that don't depend on input parameters
        self.cluster_args: Optional[dict] = None
        self.sess_filename: str = '' if sess_filename is None else sess_filename  # .pkl file to save to (gets set on first save)
        self._cnmf_fit: Optional['cnmf_ext.CNMFExt'] = None  # CNMF results
        self._frames_per_trial: Optional[np.ndarray] = None
        self._cnmf_changed_flag = False  # flag that CNMF results should be changed next time

        self.data_dir: str = ''
        self.sbx_files: list[str] = []
        self.metadata: dict[str, Any] = {}

        # initialize fields for later results to None
        self.plane_tifs: Optional[list[str]] = None      # flag for CONVERT stage; paths to plane TIF files
        self.mc_result: Optional[mcorr.MCResult] = None  # flag for MCORR stage; results of motion correction
        self.mmap_file_transposed: Optional[str] = None  # flag for TRANSPOSE stage; transposed, concatenated video
        self.cnmf_fit_filename: Optional[str] = None     # flag for CNMF stage; path to save location for CNMF results (HDF5 file)

        self.params_to_search: Optional[Union[ParamGrid, Sequence[ParamGrid]]] = None   # used if we are doing a CNMF gridsearch

        if loaded_info is None:
            # use passed-in parameters
            if tag == '':
                logging.warning('Empty string tag is interpreted as no tag (just use None)')
                tag = None

            self.tag: Optional[str] = tag
            self.mouse_id: Union[int, str] = mouse_id
            self.sess_id: int = sess_id
            self.rec_type: Optional[str] = rec_type

            # get mouse-specific overrides
            mouse_params = get_mouse_params(self.mouse_id)

            # get specifically-passed overrides
            if param_overrides is None:
                param_overrides = {}

            # annoyingly we have to apply any snr_type override here as a special case so we have it for make_cnmf_params
            snr_type = None
            for params_dict in [param_overrides, mouse_params]:
                if 'eval_extra' in params_dict and 'snr_type' in params_dict['eval_extra']:
                    snr_type = params_dict['eval_extra']['snr_type']

            self.data_dir = paths.get_processed_dir(self.mouse_id, rec_type=rec_type, create_if_not_found=True)
            self.sbx_files = find_sess_sbx_files(mouse_id, sess_id, trials_to_include=trials_to_include, trials_to_exclude=trials_to_exclude,
                                                 rec_type=rec_type, remove_ext=True)
            if len(self.sbx_files) > 0:
                logging.info('Files found:\n' +
                              str(self.sbx_files) if len(self.sbx_files) < 5 else 
                              f'[{self.sbx_files[0]}, {self.sbx_files[1]}, ..., {self.sbx_files[-2]}, {self.sbx_files[-1]}]')
            else:
                raise RuntimeError(NO_FILES_MSG)

            self.read_metadata()

            self.params = SessionAnalysisParams.from_metadata(
                metadata=self.metadata, dims=2, downsample_factor=downsample_factor, channel=channel,
                snr_type=snr_type, highpass_cutoff=highpass_cutoff
            )

            # now apply overrides on top of defaults (can ignore invalidation info since we're at the START stage already)
            if mouse_params:
                self.params = self.params.change_params_and_get_stage_to_invalidate(mouse_params, self.metadata)[0]
            if param_overrides:
                self.params = self.params.change_params_and_get_stage_to_invalidate(param_overrides, self.metadata)[0]
        else:
            # use dict to reconstruct object
            reconstruct_sessdata_obj(self, loaded_info)

            # blank out machine-specific params; will set if necessary before running any analysis
            self.update_params({
                'data': {'fnames': None},
                'spatial': {'n_pixels_per_process': None}
            })

            # apply any specific overrides
            if param_overrides is not None:
                self.update_params(param_overrides)
            
            if 'ended_time' not in self.metadata:
                # re-read metadata
                self.read_metadata(verbose=False)

    @property
    def trial_numbers(self) -> np.ndarray:
        nums, b_valid = get_trial_numbers_from_files(self.sbx_files)
        if not all(b_valid):
            raise RuntimeError('Not all trials have trial numbers')
        return nums
    

    @property
    def frames_per_trial(self) -> np.ndarray:
        """frames concatenated from each sbx file (i.e., each trial) into tif file"""
        if self._frames_per_trial is None:
            # do not care about downsampling for frames per trial
            self._frames_per_trial = np.array([sbx_utils.sbx_shape(fn)[-1] for fn in self.sbx_files])
        return self._frames_per_trial


    @property
    def cnmf_fit(self) -> Optional['cnmf_ext.CNMFExt']:
        """Lazy loader for CNMF results"""
        if self.cnmf_fit_filename is None:
            return None  # either CNMF has not been run or it's invalidated
        
        if self._cnmf_fit is None:
            loaded_data = load_cnmf(self.cnmf_fit_filename, quiet=True)
            if loaded_data is None:
                self.cnmf_fit_filename = None
            else:
                self._cnmf_fit = loaded_data
        return self._cnmf_fit

    @cnmf_fit.setter
    def cnmf_fit(self, value: 'cnmf_ext.CNMFExt'):
        self._cnmf_fit = value

    
    @property
    def scan_day(self) -> Optional[date]:
        """Infer what day the scan took place from modified time"""
        if isinstance(dt := self.metadata['ended_time'], datetime):
            return dt.date()

        if len(self.sbx_files) == 0:
            return None
        
        datecounts: Counter[date] = Counter()
        for file in self.sbx_files:
            try:
                thisdate = date.fromtimestamp(os.stat(file + '.sbx').st_mtime)
            except FileNotFoundError:
                # try using mat file instead, if raw data is unavailable
                thisdate = date.fromtimestamp(os.stat(file + '.mat').st_mtime)
            datecounts[thisdate] += 1
        
        counts_sorted = datecounts.most_common()
        if len(counts_sorted) > 1:
            counts_str = ','.join([f'{d} ({n})' for d, n in counts_sorted])
            logging.warning('Multiple recording dates found: ' + counts_str)
        return counts_sorted[0][0]
    

    @property
    def sample_rate(self) -> float:
        """Sample rate of movie taking any downsampling into account"""
        sample_rate = self.metadata['frame_rate']
        if (ds := self.downsample_factor) is not None:
            sample_rate /= ds
        return sample_rate


    @property
    def gridsearch_batch_path(self) -> str:
        df = self.get_gridsearch_results(allow_create=True)
        return str(df.paths.get_batch_path())


    # add read-only properties just for non-mutable params, so there's no confusion about whether they can be changed
    @property
    def downsample_factor(self) -> Optional[int]:
        return self.params.conversion.downsample_factor

    @property
    def crop(self) -> BorderSpec:
        return self.params.conversion.crop
    
    @property
    def plane_size(self) -> tuple[int, int]:
        return self.crop.center_shape(self.metadata['frame_size'])
    
    @property
    def crop_slices(self) -> tuple[slice, slice]:
        return self.crop.slices(self.metadata['frame_size'])
    
    @property
    def snr_type(self) -> Literal['normal', 'gamma']:
        return self.params.eval_extra.snr_type
    
    @property
    def crossplane_merge_thr(self) -> Optional[float]:
        return self.params.cnmf_extra.crossplane_merge_thr
    
    @property
    def highpass_cutoff(self) -> float:
        return self.params.transposition.highpass_cutoff
    

    #--------------------------- PARAMS/STAGE/DATA FILE MANAGEMENT-----------------------------#

    @property
    def last_valid_stage(self) -> AnalysisStage:
        if self.plane_tifs is None:
            return AnalysisStage.START
        if self.mc_result is None:
            return AnalysisStage.CONVERT
        if self.mmap_file_transposed is None:
            return AnalysisStage.MCORR
        if self.cnmf_fit is None:
            return AnalysisStage.TRANSPOSE
        if self.cnmf_fit.estimates.idx_components_eval is None:
            return AnalysisStage.CNMF
        return AnalysisStage.EVAL
    

    def read_metadata(self, file_ind=0, verbose=True):
        """Create or replace metadata field with SBX metadata read from the nth file"""
        meta = sbx_utils.sbx_meta_data(self.sbx_files[file_ind])
        if meta['ended_time'] is None:
            logging.warning("Ended time is missing from at least the first SBX file's metadata. Inferring from modified time instead.")

        if verbose:
            logging.info(f'Shape of SBX file #{file_ind}: ' + 
                         f'({meta["num_frames"]}, {meta["frame_size"][0]}, {meta["frame_size"][1]}, {meta["num_planes"]})')
        self.metadata = meta


    def invalidate_from_stage(self, stage: AnalysisStage):
        """
        Remove info relating to given and later analysis stages,
        so it must be reloaded or recomputed with current parameters
        """
        if stage > self.last_valid_stage:
            return

        logging.info(f'Invalidating results from {stage.name} onwards')

        if stage <= AnalysisStage.CONVERT:
            self.plane_tifs = None
        
        if stage <= AnalysisStage.MCORR:
            self.mc_result = None
        
        if stage <= AnalysisStage.TRANSPOSE:
            self.mmap_file_transposed = None
        
        if stage <= AnalysisStage.CNMF:
            self.cnmf_fit_filename = None
            self._cnmf_fit = None

        elif stage <= AnalysisStage.EVAL and self.cnmf_fit is not None:
            self.cnmf_fit.estimates.idx_components_eval = None
            self.cnmf_fit.estimates.idx_components_bad_eval = None
            self._cnmf_changed_flag = True

    
    def process_stage(self, stage: AnalysisStage, load: Optional[bool] = None):
        """Run or load a given stage of the pipeline"""
        match stage:
            case AnalysisStage.START:
                pass
            case AnalysisStage.CONVERT:
                self.convert_to_tif(load=load)
            case AnalysisStage.MCORR:
                self.do_mcorr_only(load=load)
            case AnalysisStage.TRANSPOSE:
                self.concat_and_transpose(load=load)
            case AnalysisStage.CNMF:
                self.do_cnmf(load=load)
            case AnalysisStage.EVAL:
                self.do_cnmf_evaluation(recalc=(load is False))
            case AnalysisStage.FINAL:
                pass

    def process_up_to_stage(self, stage: AnalysisStage, load: Optional[bool] = None):
        """Run or load pipeline from the last valid stage through the given stage"""
        curr_stage = self.last_valid_stage
        for stage_num in range(curr_stage + 1, stage + 1):
            next_stage = AnalysisStage(stage_num)
            self.process_stage(next_stage, load=load)

    def run_to_end(self):
        self.process_up_to_stage(AnalysisStage.FINAL)


    def get_nonmatching_params_for_result_file(
            self, file_path: Union[str, Path], stage: AnalysisStage) -> Iterable[str]:
        """Get a string for the first param that does not match the result file, if any"""
        params_path = paths.params_file_for_result(file_path)
        yield from self.params.get_differing_params_from_file(params_path, metadata=self.metadata, stage=stage)


    def result_file_matches_params(self, file_path: Union[str, Path], stage: AnalysisStage,
                                   raise_on_missing_params=False) -> bool:
        """
        Test whether a result file can be loaded for a given analysis stage.
        If the result file does not exist, the saved params do not match the current params for that stage, 
        return false. If there is no accompanying params file, catches and returns false
        unless raise_on_missing_params is true, in which case a FileNotFoundError is raised.
        """
        if not os.path.isfile(file_path):
            return False

        # check whether saved params match
        try:
            return not any(self.get_nonmatching_params_for_result_file(file_path, stage))
        except FileNotFoundError:
            if raise_on_missing_params:
                raise
            else:
                return False


    def write_params_for_result_file(self, file_path: Union[str, Path], stage: AnalysisStage):
        """Write params file for a newly-produced result file"""
        params_path = paths.params_file_for_result(file_path)
        self.params.write_params(path=params_path, stage=stage)


    def update_params(self, param_changes: Union[Mapping[str, dict[str, Any]], str, Path]):
        """
        Update params, also allowing changes to some things that are not part of CNMFParams:

        - "conversion" -> params for convert_to_tif (see caiman_params.ConversionParams)
        - "transposition" -> params for transpose_flatten_mc_mmap (see caiman_params.TranspositionParams)
        - "cnmf_extra" -> custom params for CNMF (see caiman_params.CNMFParamsExtra)
            - includes "seed_params" subkey for projection and seed used for initializing CNMF (see caiman_params.SeedParams)
        - "eval_extra" -> custom params for CNMF evaluation (see caiman_params.EvalParamsExtra)

        All other sub-fields of CNMFParams may also be present as keys.
        Results that are not compatible with the new parameters will be invalidated (set to None).
        
        Also supports changing from a path to a CNMF HDF5 file.
        """
        if isinstance(param_changes, (str, Path)):
            self.params, invalid_stage = self.params.change_from_cnmf_h5_and_get_stage_to_invalidate(param_changes, self.metadata)
        else:
            self.params, invalid_stage = self.params.change_params_and_get_stage_to_invalidate(param_changes, self.metadata)

        if invalid_stage is not None:
            self.invalidate_from_stage(invalid_stage)


    def save(self, save_cnmf: Optional[bool] = None):
        if save_cnmf is None:
            save_cnmf = self._cnmf_changed_flag

        if self.sess_filename == '':
            file_pattern = get_session_analysis_file_pattern(self.mouse_id, self.sess_id, tag=self.tag)  
            filename = paths.make_timestamped_filename(file_pattern)
            self.sess_filename = os.path.join(self.data_dir, filename)

        logging.info(f'Saving session analysis to {self.sess_filename}')
        fields_to_skip = ['sess_filename', 'cluster_args', 'cnmf_fit1', 'cnmf_fit2',
                          'cnmf_fit1_filename', 'cnmf_fit2_filename', '_cnmf_fit', 'tag_base']
        fields_to_save = {name: val for (name, val) in vars(self).items() if name not in fields_to_skip}

        # relativize paths
        for field in self.PATH_FIELDS:
            if field in fields_to_save:
                fields_to_save[field] = paths.relativize_path(fields_to_save[field])

        with open(self.sess_filename, 'wb') as info_file:
            pickle.dump(fields_to_save, info_file)
        
        # also save params file
        params_filename = os.path.splitext(self.sess_filename)[0] + '.json'
        self.params.write_params(params_filename)

        if save_cnmf and self.cnmf_fit is not None and self.cnmf_fit_filename is not None:
            logging.info(f'Saving CNMF to {self.cnmf_fit_filename}')
            self.cnmf_fit.save(self.cnmf_fit_filename)
            self.write_params_for_result_file(self.cnmf_fit_filename, AnalysisStage.EVAL)
            self._cnmf_changed_flag = False
            cnmf_ext.clear_cnmf_cache()


    #--------------------------- PREPROCESSING --------------------------------#


    def preview_raw_data(self, frames_to_average: Union[int, slice] = 50, channel=0, title: Optional[str] = None):
        """Display interface to preview N frames of raw data and adjust bidirectional offset"""
        # local import because canvas might not be available
        if not in_jupyter():
            raise RuntimeError('preview_raw_data only available in Jupyter')

        def save_callback(new_offset: int):
            self.update_params({'conversion': {'odd_row_offset': new_offset}})
            self.save(save_cnmf=False)

        curr_offset = self.params.conversion.odd_row_offset

        widget = caiman_viz.RawDataPreviewContainer(
            self.sbx_files, frames=frames_to_average, subinds_spatial=self.crop_slices,
            curr_offset=curr_offset, offset_save_callback=save_callback, channel=channel, title=title)
        return widget.show()


    def convert_to_tif(self, load: Optional[bool] = None, **convert_kwargs):
        """
        Concatenate sbx files and convert each plane (by default) or the entire 3D movie to .tif file(s).
        To make sure the frame rate remains mostly correct, if subindices are passed, for each file the
        indices along time must either be a slice with step of 1 or an array where np.diff(subinds) is majority 1
        (could still subvert by doing something stupid, so don't do that)

        load: Whether to try loading previously-computed results.
            None: load previous results if params match, otherwise compute anew
            True: load previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results if they already exist.
        """
        # first update with passed-in params
        self.update_params({'conversion': convert_kwargs})

        # estimate ndead explicitly if needed so we can use it later
        if self.params.conversion.odd_row_ndead is None and (
            self.metadata['scanning_mode'] == 'bidirectional' or self.params.conversion.force_estim_ndead_offset):
                odd_row_ndead = [sbx_utils.get_odd_row_ndead(f) for f in self.sbx_files]
                self.update_params({'conversion': {'odd_row_ndead': odd_row_ndead}})

        # deal with downsampling
        if (ds := self.params.conversion.downsample_factor) is not None:
            subindices = (slice(None, None, ds),) + self.crop_slices
        else:
            subindices = (slice(None),) + self.crop_slices

        def convert_one(filename: str, plane: Optional[int]):
            """Convert a single plane. If downsample_factor is not None, also downsample."""
            plane_prefix = f'Plane {plane}: ' if plane is not None else ''  # for logging

            if load is not False and os.path.isfile(filename):
                # check whether saved params match
                if self.result_file_matches_params(filename, AnalysisStage.CONVERT):
                    if load is None:
                        logging.info(plane_prefix + 'using existing .tif file for this mouse/session.')
                    return
                
                if load is None:
                    logging.info(plane_prefix + 'cannot find params file or params did not match.')
                else:
                    raise NoMatchingResultError(plane_prefix + 'could not find TIF files matching current params.')

            logging.info(plane_prefix + f'converting {len(self.sbx_files)} .sbx file(s) into one .tif file...')

            param_dict = asdict(self.params.conversion)
            # remove items that are not arguments to sbx_chain_to_tif
            for other_param in ['crop', 'downsample_factor', 'keep_3d']:
                del param_dict[other_param]

            sbx_utils.sbx_chain_to_tif(
                self.sbx_files, fileout=filename, subindices=subindices, plane=plane, dview=cluster.dview, **param_dict)
            self.write_params_for_result_file(filename, AnalysisStage.CONVERT)
      
        conversion_dir = os.path.join(self.data_dir, 'conversion')
        os.makedirs(conversion_dir, exist_ok=True)
        tagstr = '_' + self.tag if self.tag is not None else ''

        if self.params.conversion.keep_3d:
            tif_filename = os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}.tif')
            convert_one(tif_filename, plane=None)
            tif_filenames = [tif_filename]
        else:
            tif_filenames = [os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_plane{plane}.tif')
                             for plane in range(self.metadata['num_planes'])]
            
            if (len(tif_filenames) == 1):
                convert_one(tif_filenames[0], plane=None)
            else:
                for i, filename in enumerate(tif_filenames):
                    convert_one(filename, plane=i)
        
        self.plane_tifs = tif_filenames
        self.save(save_cnmf=False)


    #-------------------------- MOTION CORRECTION -----------------------------#


    def do_mcorr_only(self, load: Optional[bool] = None):
        """
        Do motion correction. 
        If results are already available, skip unless force is true.
        Saves the SessionAnalysis object if motion correction is not skipped.

        load: Whether to try loading previously-computed results.
            None: load previous results if params match, otherwise compute anew
            True: load previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results if they already exist.
        """
        if self.plane_tifs is None:
            raise RuntimeError('Must convert to TIF before doing motion correction')

        plane_results: list[mcorr.PlaneMcorrResult] = []
        is_piecewise = self.params.motion.pw_rigid

        for k_plane, tif_path in enumerate(self.plane_tifs):
            plane_prefix = f'Plane {k_plane}: ' if len(self.plane_tifs) > 1 else ''

            # update indices if necessary
            if self.params.mcorr_extra.indices_exclude_fringe and not self.params.mcorr_extra._indices_are_adjusted:
                new_indices = mcorr.compute_adjusted_indices(self.params)
                param_updates: dict[str, dict[str, Any]] = {'mcorr_extra': {'_indices_are_adjusted': True}}
                if new_indices is not None:
                    logging.info(f'Changing indices to {new_indices} to avoid dead pixels')
                    param_updates['motion'] = {'indices': new_indices}
                
                self.update_params(param_updates)

            if load is not False:  # try to load existing results
                candidate_files = mcorr.get_candidate_mcorr_result_files(tif_path, is_piecewise)
                found_result = None
                for path in candidate_files:
                    if self.result_file_matches_params(path, AnalysisStage.MCORR):
                        found_result = mcorr.load_mcorr_result(path)
                        if load is None:  # only log if we were unsure whether to load
                            logging.info(plane_prefix + 'using existing motion correction results from ' + path)
                        break
                
                if found_result is not None:
                    plane_results.append(found_result)
                    continue
                else:
                    no_match_msg = plane_prefix + 'cannot find matching motion correction results.'
                    if load is None:
                        logging.info(no_match_msg)
                    else:  # error to not load results if load is True
                        raise NoMatchingResultError(no_match_msg)
            
            # we are doing mcorr for this plane
            logging.info(plane_prefix + 'doing motion correction.')

            plane_result = mcorr.motion_correct_file(tif_path, motion_params=self.params.motion, dview=cluster.dview)
            plane_results.append(plane_result)
            self.write_params_for_result_file(plane_result['mmap_path'], AnalysisStage.MCORR)
        
        # build MCResult from list of PlaneMcorrResults
        shifts_els: Optional[list[np.ndarray]] = None
        if self.params.motion.pw_rigid:
            shifts_els = []
            for res in plane_results:
                assert res['shifts_els'] is not None, 'Should have piecewise shifts in results'
                shifts_els.append(res['shifts_els'])

        self.mc_result = mcorr.MCResult(
            mmap_files=[res['mmap_path'] for res in plane_results],
            border_to_0=max(res['border_to_0'] for res in plane_results),
            border_asym=[res['border_asym'] for res in plane_results],
            shifts_rig=[res['shifts_rig'] for res in plane_results],
            shifts_els=shifts_els,
            dims=self.plane_size,
            motion_params=self.params.motion
        )

        self.save(save_cnmf=False)
    

    def do_motion_correction(self, load: Optional[bool] = None):
        """"
        For convenience/compatibility, do motion correction and transpose in one function.
            
        load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
        """
        # do motion correction if necessary
        if self.mc_result is None or load is False:
            self.do_mcorr_only(load=load)
            assert self.mc_result is not None

        self.concat_and_transpose(load=load)

    
    def apply_motion_correction(self, mc_result: 'mcorr.MCResult', do_transpose=False, force=False):
        """
        TODO: update, figure out how this works with new params system (if at all)
        Apply motion correction result (typically from another channel of the same movie) to the current tiffs.
        By default concatenates the results, but does not transpose to C order, since the output
        will typically not be used for CNMF.
        """
        raise NotImplementedError('Not implemented for new params system')
        # # sanity checks
        # if self.plane_tifs is None:
        #     raise RuntimeError('Must convert to TIF first')

        # if len(mc_result.mmap_files) != len(self.plane_tifs):
        #     raise RuntimeError('Number of planes does not match given motion correction results')
        
        # if mc_result.dims is None or mc_result.motion_params is None:
        #     raise RuntimeError('Must set dims and motion_params on MCResult before proceeding')
        
        # if mc_result.is_piecewise and tuple(mc_result.dims) != tuple(self.plane_size):
        #     raise RuntimeError('Cannot apply piecewise results to movie of different size')
        
        # # check if already done
        # if not force and self.mc_result is not None and self.mc_result.has_same_shifts_as(mc_result):
        #     logging.info('Our current mcorr shifts match the passed ones - not re-applying.')
        #     return

        # # should be safe to do a shallow copy, not really any situation where any of the fields would be mutated
        # this_result = copy(mc_result)

        # # make MotionCorrect objects and apply the passed shifts
        # mcorr_objs = mc_result.recreate_mcorr_objects()
        # this_result.mmap_files = [mcorr.apply_mcorr_to_file(mcorr_obj, tif_file)
        #                           for mcorr_obj, tif_file in zip(mcorr_objs, self.plane_tifs)]
        # self.mmap_file_transposed = mcorr.transpose_flatten_mc_mmap(
        #     this_result.mmap_files, this_result.border_to_0, sample_rate=self.sample_rate,
        #     highpass_cutoff=self.highpass_cutoff, do_transpose=do_transpose, force=True
        # )
        # self.mc_result = this_result
        # self.save(save_cnmf=False)


    def get_original_and_corrected_movies(self, ds_ratio=5, xy_ds_ratio=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Make movie(s) to show original (top) vs. motion-corrected (bottom) data
        """
        subindices = [slice(0, None, ds_ratio), slice(0, None, xy_ds_ratio), slice(0, None, xy_ds_ratio)]
        # don't use spatial subindices when loading; instead, load_iter and then process each volume
        if self.plane_tifs is None or self.mc_result is None:
            raise RuntimeError('Motion correction not done')
        
        fr = self.sample_rate / ds_ratio
        t_subinds = subindices[0]
        dims, T = get_file_size(self.mc_result.mmap_files[0])
        nplanes = len(self.mc_result.mmap_files)
        assert len(dims) >= 2, 'Movie must be at least 2D'
        y_ds = dims[0] // xy_ds_ratio
        x_ds = dims[1] // xy_ds_ratio * nplanes
        T_ds = math.ceil(cast(int, T) / ds_ratio)
        orig_planes = np.empty((y_ds, x_ds, T_ds), order='F').view(cm.movie)
        corrected_planes = orig_planes.copy()

        # iterate through tuples of planes that must be concatenated 
        orig_vols: Iterable[tuple[np.ndarray, ...]] = zip(*[
            load_iter(tif_file, subindices=t_subinds)  # type: ignore
            for tif_file in self.plane_tifs
        ])
        corrected_vols: Iterable[tuple[np.ndarray, ...]] = zip(*[
            load_iter(mmap_file, subindices=t_subinds)  # type: ignore
            for mmap_file in self.mc_result.mmap_files
        ])

        for vols, dest, name in zip((orig_vols, corrected_vols), (orig_planes, corrected_planes), ('original', 'corrected')):
            for vol_planes, dest_im in tqdm(zip(vols, dest.T), desc=f'Processing {name} movie...', total=T_ds):  # dest_im is X x Y
                plane_movies = [cm.movie(plane.T[np.newaxis, :, :], fr=fr) for plane in vol_planes]  # now list of 1 x X x Y
                planes_ds = [plane.resize(fx=1/xy_ds_ratio, fy=1/xy_ds_ratio) for plane in plane_movies]  # type: ignore
                dest_im[:] = np.concatenate(planes_ds, axis=1)
        
        mov_orig = orig_planes.transpose((2, 0, 1))
        mov_corrected = corrected_planes.transpose((2, 0, 1))
        return mov_orig, mov_corrected
    

    def make_mc_comparison_movie(self, ds_ratio=5, xy_ds_ratio=1) -> cm.movie:
        mov_orig, mov_corrected = self.get_original_and_corrected_movies(
            ds_ratio=ds_ratio, xy_ds_ratio=xy_ds_ratio)
        return cm.concatenate([mov_orig, mov_corrected], axis=1)


    def make_mc_comparison_movie_nb(self, ds_ratio=1, xy_ds_ratio=4):
        """Make ImageWidget to compare original (top) vs motion-corrected (bottom) data"""
        if self.mc_result is None:
            raise RuntimeError('Motion correction not run')

        # local import b/c canvas might not be available
        from .caiman_viz import check_mcorr_nb
        mov_orig, mov_corrected = self.get_original_and_corrected_movies(
            ds_ratio=ds_ratio, xy_ds_ratio=xy_ds_ratio)

        return check_mcorr_nb(mov_orig, mov_corrected, self.mc_result)
    
    
    def save_mc_comparison_movie(self, ds_ratio=5, xy_ds_ratio=1, compress=0):
        """
        Save movie(s) to show original (top) vs. motion-corrected (bottom) data as avi.
        """
        mov = self.make_mc_comparison_movie(ds_ratio=ds_ratio, xy_ds_ratio=xy_ds_ratio)
        # get prefix of mcorr result file that includes datetime
        if self.mc_result is None:
            raise RuntimeError('Motion correction not done')
        
        mmap_fname = os.path.split(self.mc_result.mmap_files[0])[1]
        d1_ind = mmap_fname.index('__d1')
        movie_fname = mmap_fname[:d1_ind] + '_comparison.avi'
        mov.save(os.path.join(self.data_dir, 'mcorr', movie_fname), compress=compress)   


    def make_mc_comparison_summary_plots(self) -> tuple[Figure, Figure]:
        """Compare mean projection and correlation images - original (top) vs. motion-corrected (bottom)"""
        n_planes = self.metadata['num_planes']
        figs: list[Figure] = []

        for proj_type, type_label in zip(('mean', 'corr'), ('mean projection', 'correlation')):
            fig, axss = plt.subplots(2, n_planes, figsize=(3*n_planes, 6), sharex=True, sharey=True, squeeze=False)
            tagstr = '/' + self.tag if self.tag else ''
            fig.suptitle(f'Mouse {self.mouse_id}, session {self.sess_id}{tagstr}, {type_label}')

            for b_corrected, axs, corrected_label in zip((False, True), axss, ('original', 'corrected')):
                plane_projs = self.get_plane_projections(proj_type, motion_corrected=b_corrected)

                for k_plane, plane_proj in enumerate(plane_projs):
                    ax: Axes = axs[k_plane]
                    ax.imshow(plane_proj, cmap='viridis',
                              vmin=np.percentile(np.ravel(plane_proj), 50), 
                              vmax=np.percentile(np.ravel(plane_proj), 99.5))
                    ax.set_title(f'Plane {k_plane} ({corrected_label})')
            figs.append(fig)
        return figs[0], figs[1]


    def save_mc_comparison_summary_plots(self):
        """Compare max projection and correlation images - original (top) vs. motion-corrected (bottom) and save as png"""
        if self.mc_result is None:
            raise RuntimeError('Motion correction not done')

        # get filename to use for saving
        mmap_fname = os.path.split(self.mc_result.mmap_files[0])[1]
        fname_prefix = mmap_fname[:mmap_fname.index('__d1')]

        with plt.ioff():
            fig_mean, fig_corr = self.make_mc_comparison_summary_plots()
            fig_mean.savefig(os.path.join(self.data_dir, 'mcorr', fname_prefix + '_meanproj.png'), dpi=200)
            fig_corr.savefig(os.path.join(self.data_dir, 'mcorr', fname_prefix + '_corr.png'), dpi=200)


    def view_mcorr_pcs(self):
        """Make plot to visualize the movie PCs after motion correction of each plane"""
        if self.mc_result is None:
            raise RuntimeError('Motion correction not done')

        plane_pc_metrics = [self.mc_result.get_pc_metrics(plane=k) for k in range(self.metadata['num_planes'])]
        return caiman_viz.view_mcorr_pcs(plane_pc_metrics, sample_rate=self.sample_rate)


    # ------------------------------- PLANE CONCATENATION/TRANSPOSITION ------------------------#

    def concat_and_transpose(self, load: Optional[bool] = None):
        """
        Concatenate motion-corrected planes and convert to C-order for CNMF.
        Also does blurring and high-pass filtering steps if requested.

        load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
        """
        if self.mc_result is None:
            raise RuntimeError('Must do motion correction before concatenating/transposing planes')
          
        self.mmap_file_transposed = mcorr.do_or_load_transpose(
            self.mc_result, self.params, fr=self.sample_rate, metadata=self.metadata, dview=cluster.dview, load=load
        )
        self.save(save_cnmf=False)


    #------------------------------------ SUMMARY IMAGES ----------------------------------------#


    def make_projection(self, proj_type: str, blur_kernel_size: Optional[int] = None, motion_corrected=True) -> np.ndarray:
        """
        Make correlation image or {mean/std/max}-projection
        If blur_kernel_size > 1, do gaussian blur on a downsampled copy of the movie before computing projection
            (defaults to the same as what is in the transpose params if motion_corrected=True, else 1)
        Because it uses the transposed file, setting motion_corrected=True also includes the high-pass filter if any.
        """
        if motion_corrected:
            if self.mc_result is None:
                raise RuntimeError('Motion correction not run')
            
            curr_trans_params = self.params.transposition
            if blur_kernel_size is None:
                blur_kernel_size = curr_trans_params.blur_kernel_size

            blur_matches_params = blur_kernel_size == curr_trans_params.blur_kernel_size
            # if mean, it's a linear projection; we can blur after projecting if the initial projection has no blur
            post_blur = proj_type == 'mean' and curr_trans_params.blur_kernel_size == 1 and not blur_matches_params

            if blur_matches_params or post_blur:
                if self.mmap_file_transposed is None:
                    # move forward with the transpose step
                    self.concat_and_transpose()
                    assert self.mmap_file_transposed is not None
                mov_or_filename = self.mmap_file_transposed

            else:  # do a one-off transposition for this call
                trans_params = curr_trans_params.replace(blur_kernel_size=blur_kernel_size)
                params_copy = copy(self.params)
                params_copy.transposition = trans_params

                mov_or_filename = mcorr.do_or_load_transpose(
                    self.mc_result, params_copy, fr=self.sample_rate, metadata=self.metadata, dview=cluster.dview
                )                

            if proj_type == 'corr':
                proj = make_correlation_parallel(mov_or_filename, cluster.dview)
            else:
                ignore_nan = self.params.motion.border_nan is True
                proj = make_projection_parallel(mov_or_filename, proj_type, cluster.dview, ignore_nan=ignore_nan)
        else:
            if self.plane_tifs is None:
                raise RuntimeError('Conversion to TIF not run')
            
            if blur_kernel_size is None:
                blur_kernel_size = 1
            
            if blur_kernel_size > 1 and proj_type != 'mean':
                raise NotImplementedError('Blurring not supported for raw movie (except for mean projection)')
            
            post_blur = blur_kernel_size > 1
            
            # operate on each plane individually, then combine
            plane_projs = []
            for plane_tif in self.plane_tifs:
                if proj_type == 'corr':
                    plane_proj = make_correlation_parallel(plane_tif, cluster.dview)
                else:
                    plane_proj = make_projection_parallel(plane_tif, proj_type, cluster.dview)
                plane_projs.append(plane_proj)
            proj = np.concatenate(plane_projs, axis=1)
        
        if post_blur:
            # linear operation, can just blur here
            proj = cv2.GaussianBlur(
                proj, ksize=(blur_kernel_size, blur_kernel_size), sigmaX=blur_kernel_size/4,
                sigmaY=blur_kernel_size/4, borderType=cv2.BORDER_REPLICATE)

        proj[np.isnan(proj)] = 0
        return proj


    def load_projection_from_result(self, proj_type: str) -> tuple[np.ndarray, MescoreSeries]:
        """
        Attempt to load a projection (corr, mean, std, or max) from a previous mesmerize result.
        Returns the projection and the item the projection was pulled from (a Series).
        Raises a NoMatchingResultError if no matching item was found.
        """
        try:
            batch = self.get_gridsearch_results()
        except NoBatchFileError as e:
            raise NoMatchingResultError('No matching mesmerize items to pull projection from') from e

        completed_runs = batch.loc[[o is not None and o['success'] for o in batch.outputs]]
        # check whether params up to transposition match for each candidate item
        trans_params_match = np.zeros(len(completed_runs), dtype=bool)
        for i in range(len(completed_runs)):
            res_row = cast(MescoreSeries, completed_runs.iloc[i])
            res_file = res_row.cnmf.get_output_path()
            trans_params_match[i] = self.result_file_matches_params(res_file, AnalysisStage.TRANSPOSE)

        matching_runs = completed_runs.loc[trans_params_match, :]
        if len(matching_runs) == 0:
            raise NoMatchingResultError('No matching mesmerize items to pull projection from')

        completed_row = cast(MescoreSeries, matching_runs.iloc[-1])
        if proj_type == 'corr':
            return completed_row.caiman.get_corr_image(), completed_row
        else:
            uuid = str(completed_row.at['uuid'])
            batch_path = completed_row.paths.get_batch_path()
            proj_path = batch_path.parent / uuid / f'{uuid}_{proj_type}_projection.npy'
            return np.load(proj_path), completed_row


    def get_borders(self, motion_corrected=True) -> list[BorderSpec]:
        if motion_corrected:
            # load from motion correction results
            if self.mc_result is None:
                raise RuntimeError('Motion correction not run')
            return self.mc_result.border_asym
        else:
            if self.plane_tifs is None:
                raise RuntimeError('Conversion to TIF not run')
            return [BorderSpec.equal(0)] * len(self.plane_tifs)


    def get_projection(self, proj_type: str, blur_kernel_size=1, motion_corrected=True) -> np.ndarray:        
        if blur_kernel_size > 1:
            logging.info('Cannot pull from mesmerize (blur_kernel_size > 1)')
        elif not motion_corrected or proj_type not in ['corr', 'mean', 'std', 'max']:
            pass
        else:
            try:
                return self.load_projection_from_result(proj_type=proj_type)[0]
            except NoMatchingResultError:
                logging.info('No matching mesmerize items - computing projection anew')

        return self.make_projection(proj_type, blur_kernel_size=blur_kernel_size, motion_corrected=motion_corrected)

        
    def get_projection_for_seed(self, type='mean', motion_corrected=True, blur_size=1, norm_medw: Optional[int] = None,
                                borders: Optional[list[BorderSpec]] = None, **seed_params_extra) -> tuple[np.ndarray, dict]:
        """
        Make 2D projection image to use for making seed with given params.
        Returns the projection along with any unused params
        """
        proj = self.get_projection(proj_type=type, blur_kernel_size=blur_size, motion_corrected=motion_corrected)

        if borders is None:
            borders = self.get_borders(motion_corrected=motion_corrected)

        if norm_medw is not None:
            proj = preprocess_proj_for_seed(proj, med_w=norm_medw, borders=borders)

        seed_params_extra['borders'] = borders
        return proj, seed_params_extra
    

    def get_plane_projections(self, projection_params: Union[str, dict], motion_corrected=True, exclude_border=True) -> list[np.ndarray]:
        if isinstance(projection_params, str):
            projection_params = {'type': projection_params}

        proj, seed_params_extra = self.get_projection_for_seed(motion_corrected=motion_corrected, **projection_params)
        borders: list[BorderSpec] = seed_params_extra['borders']

        plane_projs = np.split(proj, len(borders), axis=1)
        if exclude_border:
            plane_projs = [plane[border.slices(plane.shape)] for plane, border in zip(plane_projs, borders)]
        return plane_projs


    def get_zmax_projection(self, projection_params: Optional[Union[str, dict]] = None,
                            exclude_border=False) -> np.ndarray:
        """
        Get max of temporal projection over planes for the given session.
        Defaults to using the projection used for the seed of the selected CNMF run if projection_params
        are not given.
        """
        if projection_params is None:
            ind = self.get_selected_index()
            if ind is None:
                raise RuntimeError('No selected CNMF run; cannot use seed projection')

            item = cast(MescoreSeries, self.get_gridsearch_results().iloc[ind])
            output_path = item.cnmf.get_output_path()
            proj_path = output_path.parent / 'projection_for_seed.npy'
            if proj_path.exists():
                full_proj = np.load(proj_path)
                plane_projs = np.split(full_proj, self.metadata['num_planes'], axis=1)
                if exclude_border:
                    borders = self.get_borders(motion_corrected=True)
                    plane_projs = [plane[border.slices(plane.shape)] for plane, border in zip(plane_projs, borders)]
            else:
                # fall back to recomputing based on seed_params
                seed_params = asdict(self.params.cnmf_extra.seed_params)
                logging.warning(f'projection_for_seed not saved - recomputing with params {seed_params}')
                plane_projs = self.get_plane_projections(seed_params, exclude_border=exclude_border)
        else:
            plane_projs = self.get_plane_projections(projection_params, exclude_border=exclude_border)

        return np.max(plane_projs, axis=0)


    #--------------------------------  CNMF  ----------------------------------------#


    def check_patch_params(self, ax=None):
        """Use view_quilt to check the current CNMF patch parameters"""
        rf = self.params.patch.rf
        if rf is None:
            raise RuntimeError('rf is None - movie will not be processed in patches.')
        
        if isinstance(rf, Sequence):
            raise NotImplementedError('check_patch_params is only supported for square patches')
        
        stride = self.params.patch.stride
        if stride is None:
            stride = int(rf * 2 * .1)
        elif isinstance(stride, Sequence):
            raise NotImplementedError('check_patch_params is only supported for square patches')

        patch_width = rf * 2 + 1
        patch_overlap = stride + 1
        patch_stride = patch_width - patch_overlap
        print(f'Patch width: {patch_width}, stride: {patch_stride}, overlap: {patch_overlap}')

        # make correlation image of selected plane
        corr_image = self.get_projection('corr')

        patch_ax = view_quilt(corr_image, patch_stride, patch_overlap,
                              vmin=float(np.percentile(np.ravel(corr_image), 50)),
                              vmax=float(np.percentile(np.ravel(corr_image), 99.5)),
                              ax=ax, figsize=(4, 4))

        patch_ax.set_title(f'width={patch_width}\noverlap={patch_overlap}')
    

    def do_cnmf(self, load: Optional[bool] = None) -> None:
        """
        Do or load CNMF.

        load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
        """
        uuid, _ = self.start_cnmf_with_mescore(load=load, backend='local', wait=True)
        self.finish_cnmf_processing(uuid)


    def start_cnmf_with_mescore(self, load: Optional[bool] = None, backend='local_async', wait=False,
                                **run_args) -> tuple[str, Waitable]:
        """
        Start a CNMF run through mesmerize-core, so results will be in the batch dataframe. Defaults to running in background.
        run_args are set to series.caiman.run in mescore.
        Returns the UUID and process, which may still be running depending on whether 'wait' is set to True in run_args.
        Once the process is done, finish_cnmf_processing should be called with the resulting UUID.

        load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
        """        
        # Get params to pass to mescore
        batch = self.get_gridsearch_results(allow_create=True)
        
        if load is not False:  # try loading existing
            completed_runs = batch.loc[[out is not None and out['success'] for out in batch.outputs], :]
            for _, row in completed_runs.iterrows():
                row = cast(MescoreSeries, row)

                # check whether CNMF params match
                # eval param differences are handled in finish_cnmf_processing.
                output_path = row.cnmf.get_output_path()
                params_path = paths.params_file_for_result(output_path)
                try:
                    saved_params = cmp.UpToEvalParamStruct.read_from_file(params_path)
                except FileNotFoundError:
                    continue
                
                if self.params.do_params_match(saved_params, metadata=self.metadata, stage=cmp.AnalysisStage.CNMF):
                    uuid = str(row.at['uuid'])
                    logging.info(f'Found matching CNMF run with uuid {uuid}')
                    return uuid, DummyProcess()
            
            # no matching run found
            if load is True:
                raise RuntimeError('No CNMF run found matching current parameters')
            else:
                logging.info('No previous matching CNMF run found - starting process')

        
        params_obj = self.params.read_cnmf_params()

        # set params that depend on previous results
        if self.mc_result is None:
            raise RuntimeError('No MC results; cannot run CNMF')

        if self.mmap_file_transposed is None:
            raise RuntimeError('No transposed data file; cannot run CNMF')

        params_obj.change_params({
            'data': {'fnames': [self.mmap_file_transposed]},
            'patch': {'border_pix': self.mc_result.border_to_0}  # only allows scalar value for border
        })

        # infer name of spatial seed, if using
        seed_params = self.params.cnmf_extra.seed_params
        if seed_params.type != 'none':
            Ain_name = get_spatial_seed_name(seed_params)
        else:
            Ain_name = None

        # create mesmerize item (do this first to get a UUID, which tells us where to save the seed)
        batch.caiman.add_item(
            algo='cnmf',
            item_name=batch.paths.get_batch_path().stem,
            input_movie_path=self.mmap_file_transposed,
            params={
                'main': params_obj.to_dict(),
                'refit': seed_params.type == 'none',
                'Ain_path': f'{Ain_name}.npy' if Ain_name is not None else None
            }
        )

        item = cast(MescoreSeries, batch.iloc[-1])
        uuid = str(item.uuid)
        output_dir = batch.paths.get_batch_path().parent.joinpath(uuid).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # preemptively save params that will be used for this run
        output_path = output_dir / f'{uuid}.hdf5'
        params_path = paths.params_file_for_result(output_path)
        mesmerize_cnmf_params = self.params.copy_with_mesmerize_run_differences()
        mesmerize_cnmf_params.write_params(params_path)

        if seed_params.type != 'none':
            # actually make and save seed
            seed_param_dict = asdict(seed_params)
            # fill in default value of gSig
            if seed_param_dict['gSig'] is None:
                seed_param_dict['gSig'] = params_obj.init.gSig[0]

            proj, seed_params_extra = self.get_projection_for_seed(**seed_param_dict)
            np.save(output_dir / 'projection_for_seed.npy', proj)
            Ain = footprints.make_spatial_seed_from_projection(proj, seed_params_extra)
            np.save(output_dir / f'{Ain_name}.npy', np.array(Ain))  # saves as object array

        if 'dview' not in run_args and backend in ['local', 'local_async']:
            run_args['dview'] = cluster.dview

        proc = item.caiman.run(backend=backend, wait=wait, **run_args)
        return uuid, proc
    

    def finish_cnmf_processing(self, finished_or_loaded_cnmf_uuid: str):
        """
        Post-processing to do after a mesmerize CNMF run has finished or been loaded,
        to select (load) the run and ensure results match cnmf_extra and eval parameters.
        """
        # Load results from mesmerize. We want the params actually used to match current self.params
        # once this method is done.
        target_params = self.params
        self.select_gridsearch_run(finished_or_loaded_cnmf_uuid, quiet=True)
        assert self.cnmf_fit is not None, 'cnmf_fit must be assigned if select_gridsearch_run succeeds'

        # fix remaining differences
        do_crossplane_merge = False
        redo_eval = False
        diff_params = self.params.get_differing_params(target_params, metadata=self.metadata)
        self.params = target_params  # reset from temporary variable

        for diff_param in diff_params:
            if diff_param == 'cnmf_extra.crossplane_merge_thr':
                do_crossplane_merge = self.metadata['num_planes'] > 1  # only matters if there are at least 2 planes
            elif diff_param.split('.')[0] in cmp.EvalParamStruct.model_fields:
                redo_eval = True
            else:
                self.invalidate_from_stage(cmp.AnalysisStage.CNMF)
                raise RuntimeError('Unexpected differences between current parameters and those used in CNMF run; invalidating')

        # do crossplane merging if necessary
        if do_crossplane_merge:
            logging.info('Doing crossplane merging')
            n_merged = self.cnmf_fit.estimates.merge_components_crossplane(
                n_planes=self.metadata['num_planes'], params=self.cnmf_fit.params, thr=self.crossplane_merge_thr)
            logging.info(f'Merged {n_merged} sets of components')
            if n_merged > 0:
                logging.info('Redoing evaluation after merging')
                redo_eval = True
        else:
            n_merged = 0
        
        # redo evaluation after merging or switching SNR type
        if redo_eval:
            if n_merged == 0:
                logging.info(f'Redoing evaluation with {self.snr_type} SNR type')
            self.do_cnmf_evaluation(recalc=True)

        self.make_df_over_f(recalc=False)

        if self.downsample_factor is not None:
            assert self.frames_per_trial is not None, 'frames per trial should be set during conversion'
            logging.info(f'Upsampling (interpolating) results by a factor of {self.downsample_factor}')
            self.cnmf_fit.estimates.interpolate_t(self.downsample_factor, self.frames_per_trial)
        
        self.save(save_cnmf=True)


    def do_cnmf_gridsearch(self, gridsearch_params: Union[ParamGrid, Sequence[ParamGrid]], wait=True,
                           backend='local_async') -> list[Waitable]:
        """
        Test CNMF with every combination of given parameters, using mesmerize-core. This function runs on the local host.
        Each key of gridsearch_params should be a tuple (group, name) specifying a specific CNMF parameter.
        Each entry is a list of parameter values to test (can be of length 1 to set value for all runs).
        Runs on a remote cluster if host is given and matches an entry in host_info.py (e.g. login.tj).
        To run locally, use 'localhost'.
        If wait is false, launches grid search process but does not wait for it to finish or validate success.
        """
        # Make/get a dataframe that will hold results
        df = self.get_gridsearch_results(allow_create=True)
        orig_nruns = len(df)
        procs = gridsearch_analysis.do_cnmf_gridsearch(self, gridsearch_params, backend=backend)
        if wait:
            while len(procs) > 0:
                try:
                    procs.pop().wait()  # will throw if there is an error
                except CalledProcessError:  # get more information by reading the batch file
                    raise gridsearch_analysis.GridsearchError(df, orig_nruns)
        return procs


    async def do_cnmf_gridsearch_remote(self, gridsearch_params: Union[ParamGrid, Sequence[ParamGrid]], host: str) -> asyncio.Task[None]:
        """
        Same as do_cnmf_gridsearch, but runs asynchronously on a remote host. Returns an asyncio Task that can then be awaited
        (among other things) in an async environment (note that process is cancelled if not awaited)
        """
        # Make/get a dataframe that will hold results
        df = self.get_gridsearch_results(allow_create=True)
        orig_nruns = len(df)

        host_spec, hostinfo = remoteops.resolve_host(host)

        # set up the remote computer to call the gridsearch script        
        self.params_to_search = gridsearch_params
        self.save(save_cnmf=False)

        args = [hostinfo.lex.quote(self.sess_filename), '--wait']  # need so that submission script waits for all to finish
        sess_dir = os.path.split(self.sess_filename)[0]
        log_file = os.path.join(sess_dir, 'gridsearch-%j.out')
        log_file_norm = paths.normalize_path(log_file, for_host=hostinfo)
        log_file_esc = hostinfo.lex.quote(log_file_norm)
        slurm_opts = '--wait --cpus-per-task=1 --output=' + log_file_esc # don't need multiple CPUs to just launch CNMF runs

        task = await remoteops.start_script_on_host('cnmf_gridsearch', args, host=host_spec,
                                                    slurm_opts=slurm_opts, check=True)
        async def finish_and_check_gridsearch() -> None:
            finished_proc = await task
            if finished_proc.returncode != 0:
                raise gridsearch_analysis.GridsearchError(df, orig_nruns)

        return asyncio.create_task(finish_and_check_gridsearch())

    do_cnmf_gridsearch_remote_sync = remoteops.make_sync(do_cnmf_gridsearch_remote)
    """Same as do_cnmf_gridsearch, but runs on a remote host (and waits for completion)"""
    

    def make_df_over_f(self, recalc: bool = False, denoised: Optional[bool] = None):
        """
        Calculates delta F over F for all ROIs using our default settings and
        saves to estimates.F_dff and estimates.F_dff_denoised. Set denoised to True or False
        to only calculate one or the other
        """
        if self.cnmf_fit is None:
            raise RuntimeError('CNMF fit not found')
        est = self.cnmf_fit.estimates   
    
        if denoised is None or not denoised:
            if recalc or est.F_dff is None:
                logging.info('Calculating df/f')
                est.F_dff = calc_df_over_f(est, use_residuals=True)
            else:
                logging.info('Found df/f - not recalculating')
        
        if denoised is None or denoised:
            if recalc or est.F_dff_denoised is None:
                logging.info('Calculating denoised df/f')
                est.F_dff_denoised = calc_df_over_f(est, use_residuals=False)
            else:
                logging.info('Found denoised df/f - not recalculating')


    #------------------------- ROI EVALUATION -----------------------------#


    def do_cnmf_evaluation(self, recalc=False):
        """
        Recalculates idx_components based on automatic criteria.
        new_quality_params: Unused, throws an error if not None, use self.update_params({'quality': <new params>}) instead
        snr_type: which type of SNR to use; if left as None, uses self.snr_type. Otherwise updates self.snr_type.
        recalc: If true, recomputes all metrics; otherwise reuses them if they already exist.
        """
        if self.mmap_file_transposed is None:
            raise RuntimeError('Cannot do CNMF evaluation without input file (motion correction result)')

        if self.cnmf_fit_filename is None:
            raise RuntimeError('CNMF fit not found')
        
        if not recalc:  # if the eval params match, no need to redo.
            params_file = paths.params_file_for_result(self.cnmf_fit_filename)
            try:
                old_eval_params = cmp.UpToEvalParamStruct.read_from_file(params_file)
            except FileNotFoundError:
                pass
            else:
                if self.params.do_params_match(old_eval_params, self.metadata, stage=cmp.AnalysisStage.EVAL):
                    logging.info('Eval params have not changed - skipping evaluation')
                    return

        new_quality_params = self.params.quality
        snr_type = self.params.eval_extra.snr_type

        logging.info('Doing CNMF component evaluation')
        assert (cnmf_fit := self.cnmf_fit) is not None
        evaluate_cnmf(cnmf_fit, self.mmap_file_transposed, snr_type=snr_type,
                      new_quality_params=new_quality_params.copy(), recalc=recalc, dview=cluster.dview)

        # re-save cnmf object with evaluation data added
        logging.info('Re-saving CNMF object with evaluation data')
        self.save(save_cnmf=True)

    
    #--------------------- ACCESSING CNMF RESULTS -----------------------------#
            
    
    def get_gridsearch_results(self, allow_create=False) -> MescoreBatch:
        df = get_batch_for_session(self.mouse_id, self.sess_id, tag=self.tag, data_dir=self.data_dir, create=allow_create)

        # fix input_movie_path for new directory structure - make relative to parent raw data path
        any_changed = False
        for _, row in df.iterrows():
            abs_path_from_data_dir = Path(self.data_dir) / str(row['input_movie_path'])
            if abs_path_from_data_dir.exists():
                new_rel_path = df.paths.split(abs_path_from_data_dir)[1]
                row['input_movie_path'] = str(PurePosixPath(new_rel_path))
                any_changed = True
        if any_changed:
            df.caiman.save_to_disk()            
        return df


    def get_gridsearch_diffs(
        self, batch: Optional[MescoreBatch] = None, params_to_exclude: Optional[Container[str]] = None, exclude_quality=True
        ) -> pd.DataFrame:
        """
        Get table of parameter differences between all the gridsearch runs that have been run so far
        as well as the current parameter values.
        If batch is given, use this batch dataframe instead of calling get_gridsearch_results()
        """
        if batch is None:
            df = self.get_gridsearch_results()
        else:
            df = batch.reset_index(drop=True)

        # subset to algorithm == CNMF
        sub_df = df[df.algo == 'cnmf']

        # get a list of params objects
        params_objs: list[Union[cmp.SessionAnalysisParams, CNMFParams]] = []
        differing_params: set[str] = set()

        # iterate through and build list of params that differ from current ones
        for ind in sub_df.index:
            params = cmp.load_params_from_batch_item(sub_df.loc[ind, :])
            for changed_param in self.params.get_differing_params(
                params, metadata=self.metadata, params_to_exclude=params_to_exclude, exclude_quality=exclude_quality):
                differing_params.add(changed_param)
            params_objs.append(params)

        # now make a dataframe of the differences
        param_names = sorted(differing_params)
        records: dict[str, Union[list, pd.Series]] = {param_name: [] for param_name in param_names}

        for params in params_objs:
            for param_name in param_names:
                group, key = param_name.split('.')
                if hasattr(params, group):
                    val = getattr(getattr(params, group), key)
                else:  # (for CNMFParams where we don't have all the groups available)
                    val = "<unknown>"
                records[param_name].append(val)

        records['uuid'] = sub_df.uuid
        return pd.DataFrame(records, dtype=object, index=sub_df.index)
            

    def select_gridsearch_run(self, uuid: Optional[str] = None, *_, index=None, quiet=False, force_reload=True,
                              allow_rerunning_prereqs=False) -> str:
        """
        Select a CNMF gridsearch run by either UUID or index in the dataframe, and make it the current run
        Sets params and prerequisite and CNMF outputs to match the selected run
        Returns the UUID of the selected run.

        allow_rerunning_prereqs: Set to true to re-compute prerequisites (motion correction, etc.) 
            rather than just loading if outputs matching the saved params are not found (or if it's indeterminate)
        """
        if index is not None and uuid is not None:
            raise ValueError('Cannot specify both uuid and index')
        
        if index is None and uuid is None:
            raise ValueError('Must specify either uuid or index')

        if not force_reload:
            # see if it is already loaded
            curr_uuid = self.get_selected_uuid()
            if index is not None:
                curr_idx = self.get_selected_index()
                matched = curr_idx is not None and curr_idx == index
            else:
                assert uuid is not None
                matched = curr_uuid is not None and curr_uuid == uuid

            if matched:
                assert curr_uuid is not None
                if not quiet:
                    logging.info('Requested CNMF run is already selected')
                return curr_uuid

        df = self.get_gridsearch_results()

        if index is not None:
            try:
                row = cast(MescoreSeries, df.iloc[index])
                uuid = str(row.uuid)
            except KeyError:
                raise ValueError(f'Index {index} not present in gridsearch dataframe')
        else:
            assert uuid is not None
            rows = df.loc[df.uuid == uuid]
            if len(rows) != 1:
                raise ValueError(f'UUID {uuid} not present in gridsearch dataframe')
            row = cast(MescoreSeries, rows.iloc[0])

        if row.outputs is None or not row.outputs['success']:
            raise RuntimeError('Run has not completed or did not succeed')
        
        if str(row.algo) != 'cnmf':
            raise RuntimeError('Selected item is not a CNMF run')
        
        # OK, we have a CNMF run and it has completed successfully. Set fields.
        # Load prerequisites first
        cnmf_path = str(row.cnmf.get_output_path())
        params_path = paths.params_file_for_result(cnmf_path)

        try:
            loaded_params = cmp.SessionAnalysisParams.read_from_file(params_path)
            stage_to_invalidate = self.params.get_first_nonmatching_stage(loaded_params, metadata=self.metadata)
            self.invalidate_from_stage(stage_to_invalidate)
            self.params = loaded_params

        except FileNotFoundError:
            # load partial params from CNMF object
            logging.info('CNMF params file not found - assuming params not specified in the CNMF object match current params.')
            self.update_params(cnmf_path)

        # try loading or running prerequisites to fill in invalidated data
        try:
            load = None if allow_rerunning_prereqs else True
            self.process_up_to_stage(AnalysisStage.TRANSPOSE, load=load)

        except NoMatchingResultError as e:
            logging.warning(
                'Cannot load all prerequisite results of the selected CNMF run. '
                'The CNMF result will still be loaded, but some functionality may not work '
                'without re-running to get the missing results (see exception info).', exc_info=e)

        self.cnmf_fit_filename = cnmf_path
        cnmf_fit = cnmf_ext.load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=quiet)
        self.cnmf_fit = cnmf_fit
        return uuid
    
    def select_successful_run(self, quiet=True) -> Optional[str]:
        """Try to select the successful gridsearch run if there is only one"""
        df = self.get_gridsearch_results()
        df_successful = df.loc[[out is not None and out['success'] for out in df.outputs], :]
        if df_successful.shape[0] == 1:
            return self.select_gridsearch_run(uuid=df_successful.iloc[0].uuid, quiet=quiet)
        else:
            return None
        
    def get_selected_index(self) -> Optional[int]:
        """Get index of selected CNMF run in gridsearch results table, if any"""
        if self.cnmf_fit_filename is None:
            return None
        
        try:
            batch = self.get_gridsearch_results()
        except NoBatchFileError:
            return None
        
        # find cnmf_fit_filename in batch
        selected_filename = os.path.split(self.cnmf_fit_filename)[1]
        for ind, row in batch.iterrows():
            outputs = cast(Optional[dict], row.at['outputs'])
            if outputs is None or not outputs['success']:
                continue

            run_filename = os.path.split(outputs['cnmf-hdf5-path'])[1]
            if run_filename == selected_filename:
                return cast(int, ind)
        return None
        
    def get_selected_uuid(self) -> Optional[str]:
        index = self.get_selected_index()
        if index is not None:
            batch = self.get_gridsearch_results()
            return str(batch.loc[index, 'uuid'])
        return None
    

    def get_relative_depths(self) -> np.ndarray:
        """Get depth of each plane, with top plane being 0, in um"""
        if self.metadata['num_planes'] == 1:
            return np.array([0.])

        elif 'etl_pos' not in self.metadata or len(self.metadata['etl_pos']) == 0:
                raise RuntimeError('3D recording but there is no ETL information! Cannot infer depths')

        depths: list[int] = self.metadata['etl_pos']
        # make top plane 0
        # um per pixel for plane should be negative, e.g. plane 1 is at -30
        return np.array([d - depths[0] for d in depths])


    def get_coms_3d(self, unit: Literal['um', 'pixels'] = 'pixels') -> ScaledDataFrame:
        """
        Calculates the center of mass of each component (distance from top left corner of top plane)
        This weights the 2D COMs according to the number of pixels
        The pixel size is taken into account, particularly spacing between z-planes which may be uneven.
        Return value is a 3D position object with vectors for x y and z positions in um.
        """
        if self.cnmf_fit is None:
            raise RuntimeError('No CNMF run selected')
        
        est = self.cnmf_fit.estimates
        if est.A is None or est.dims is None:
            raise RuntimeError('CNMF not run')

        # interpret dims as being in 3D
        pix_y, pix_x = self.plane_size
        spacing_y = self.metadata['um_per_pixel_y']
        um_vals_y = np.arange(pix_y) * spacing_y
        spacing_x = self.metadata['um_per_pixel_x']
        um_vals_x = np.arange(pix_x) * spacing_x

        spacings = {'y': spacing_y, 'x': spacing_x}
        um_vals_z = self.get_relative_depths()
        if len(um_vals_z) == 1:
            um_vals = (um_vals_y, um_vals_x)
        else:
            um_vals = (um_vals_y, um_vals_x, um_vals_z)
            # if the planes are uniform, we save the spacing; otherwise just set it to None
            spacings_z = np.diff(um_vals_z)
            if np.all(spacings_z == spacings_z[0]):
                spacings['plane'] = spacings_z[0]
            else:
                spacings['plane'] = None

        # now actually calculate the COMs
        coms_3d_um = cmcustom.my_com(est.A, *um_vals)

        # if we want the result in pixels but the z-spacing is nonuniform, we have to interpolate
        if 'plane' in spacings and spacings['plane'] is None and unit == 'pixels':
            spacings.pop('plane')
            yx_df = make_um_df(coms_3d_um[:, :-1], pixel_size=spacings)
            plane_pix = np.interp(coms_3d_um[:, -1], um_vals[-1], range(len(um_vals[-1])))
            plane_df = make_pixel_df({'plane': plane_pix})
            coms_df = cast(ScaledDataFrame, pd.concat([yx_df, plane_df], axis=1))
        else:
            coms_df = make_um_df(coms_3d_um, pixel_size=spacings)

        return coms_df.to_unit(unit)


    @overload
    def get_xy_footprints(self, binarize: Literal[True], normalize: Literal[False], **binarize_kwargs) -> sparse.csc_matrix[np.bool_]:
        ...

    @overload
    def get_xy_footprints(self, binarize: Literal[False] = ..., normalize: bool = ..., **binarize_kwargs) -> sparse.csc_matrix[np.floating]:
        ...
    
    def get_xy_footprints(self, binarize=False, normalize=True, **binarize_kwargs) -> sparse.csc_matrix:
        """
        Get footprints (A) of selected CNMF run compressed to just X/Y coordinates
        added across the z dimension and normalized.
        """
        if normalize and binarize:
            logging.warning('normalize has no effect when binarize is true')

        if self.cnmf_fit is None:
            raise RuntimeError('CNMF not run or selected')
        est = self.cnmf_fit.estimates
        if est.A is None:
            raise RuntimeError('A should be populated')

        A = sparse.csc_matrix(est.A)
        n_planes = self.metadata['num_planes']
        if binarize:
            xy_footprints = footprints.collapse_footprints_to_xy(A, n_planes, binarize=True, **binarize_kwargs)
        else:
            xy_footprints = footprints.collapse_footprints_to_xy(A, n_planes, binarize=False, **binarize_kwargs)
            if normalize:
                xy_footprints = footprints.normalize_footprints(xy_footprints)
        return xy_footprints
    

    @overload
    def get_footprints_per_plane(self, binarize: Literal[False] = ..., normalize: bool = ..., **binarize_kwargs) -> list[sparse.csc_matrix[np.floating]]:
        ...
    
    @overload
    def get_footprints_per_plane(self, binarize: Literal[True], normalize: Literal[False] = ..., **binarize_kwargs) -> list[sparse.csc_matrix[np.bool_]]:
        ...    

    def get_footprints_per_plane(self, binarize=False, normalize=False, **binarize_kwargs) -> list[sparse.csc_matrix]:
        """
        Get masks (A) of selected CNMF run for each plane separately,
        normalized so that each nonempty mask is a unit vector (if normalize is true)
        """
        if normalize and binarize:
            logging.warning('normalize has no effect when binarize is true')

        if self.cnmf_fit is None:
            raise RuntimeError('CNMF not run or selected')
        est = self.cnmf_fit.estimates
        if est.A is None:
            raise RuntimeError('A should be populated')

        A = sparse.csc_matrix(est.A)
        n_planes = self.metadata['num_planes']
        if A.shape[0] % n_planes != 0:
            raise RuntimeError('Number of planes does not go into number of pixels')
        pix_per_plane = A.shape[0] // n_planes

        xy_footprints_each: list[sparse.csc_matrix] = []
        for k_plane in range(n_planes):
            submat = A[pix_per_plane*k_plane:pix_per_plane*(k_plane+1), :]
            if binarize:
                submat = footprints.binarize_footprints(submat, **binarize_kwargs)
            elif normalize:
                submat = footprints.normalize_footprints(submat)
            xy_footprints_each.append(submat)
        return xy_footprints_each


    def save_estimates(self, save_background=False, format: Literal['.mat', '.npz'] = '.mat', 
                       separate_metadata=True, force_dff=True) -> tuple[str, str]:
        """
        Export CNMF results to .mat or .npz data file.
        If separate_metadata is true, info about the recording and each cell is reformatted into a 
        tabular format where appropriate and saved in a separate .pkl file.
        force_dff: Compute F_dff and F_dff_denoised if they are None, rather than skipping them.
        Returns (data_file_path, metadata_file_path) (these are the same if separate_metadata is false).
        """
        if self.cnmf_fit_filename is None:
            if (uuid := self.select_successful_run()) is None:
                raise RuntimeError('No CNMF results to save (or multiple gridsearch runs, must select one')
            else:
                logging.info(f'Auto-selected successful run with uuid {uuid}')
                assert self.cnmf_fit_filename is not None

        # reload from disk (e.g. to sync changes from mesmerize_viz)
        cnmf_fit = cnmf_ext.load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=True)
        self.cnmf_fit = cnmf_fit
        est = cnmf_fit.estimates
        assert est.A is not None and est.dims is not None, 'Estimates elements should not be None - has CNMF been run?'

        export_dir = os.path.join(self.data_dir, 'export')
        os.makedirs(export_dir, exist_ok=True)
        tagstr = '_' + self.tag if self.tag is not None else ''
        uuid = self.get_selected_uuid()
        assert uuid is not None, 'Should have a selected UUID if there are CNMF results'

        filename_base = os.path.join(export_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_{uuid}')
        if separate_metadata:
            filename_data = filename_base + f'_activity{format}'
        else:
            filename_data = filename_base + f'_estimates{format}'
        filename_meta = filename_base + '_metadata.pkl'
        
        # collect data that will be saved
        est_data_fields = ['C', 'F_dff', 'F_dff_denoised', 'S', 'YrA']
        if save_background:
            est_data_fields.extend(['f', 'b'])

        if force_dff and (est.F_dff is None or est.F_dff_denoised is None):
            self.make_df_over_f()
            self.save(save_cnmf=True)  # save to CNMF hdf file so this doesn't have to be repeated

        est_meta_fields= ['idx_components', 'idx_components_eval', 'accepted_list',
                          'rejected_list', 'idx_components_marked', 'SNR_comp', 'r_values', 'cnn_preds']

        save_data = {field: getattr(est, field) for field in est_data_fields if getattr(est, field) is not None}
        save_meta = {field: getattr(est, field) for field in est_meta_fields if getattr(est, field) is not None}

        sess_fields_to_save = ['frames_per_trial', 'trial_numbers', 'mouse_id', 'sess_id', 'tag', 'metadata', 'scan_day']
        save_meta.update({field: getattr(self, field) for field in sess_fields_to_save if getattr(self, field) is not None})
        if self.tag is None:
            save_meta['tag'] = ''

        # add centers of mass
        save_meta['com_3d'] = self.get_coms_3d(unit='pixels')

        # save estimates data
        if not separate_metadata:
            # combine metadata with rest of data
            save_data.update(save_meta)
            # convert COM to a matrix
            save_data['com_3d'] = np.stack([save_data['com_3d'].y, save_data['com_3d'].x, save_data['com_3d'].plane],
                                           axis=1)

        if format == '.mat':
            # truncate_existing is necessary to avoid error re-saving dict when file already exists (https://github.com/frejanordsiek/hdf5storage/issues/127)
            savemat(filename_data, save_data, truncate_existing=True)
        elif format == '.npz':
            np.savez_compressed(filename_data, **save_data)
        else:
            raise RuntimeError('Unrecognized save format')
        
        logging.info(f'Saved estimates to {filename_data}')

        # save metadata
        if separate_metadata:
            save_meta = tabularize_estimates_metadata(save_data, **save_meta)
            with open(filename_meta, 'wb') as file:
                pickle.dump(save_meta, file)
            logging.info(f'Saved metadata to {filename_meta}')
            return filename_data, filename_meta
        else:
            return filename_data, filename_data


    #------------------------- VISUALIZATIONS ----------------------------#


    def plot_cnmf_rois(self, bg_type: Union[str, dict] = 'corr', mark_accepted=True, force_mpl=False):
        """Plot ROIs on top of a projection; by default marks accepted ones in a different color"""
        if self.cnmf_fit is None:
            if (uuid := self.select_successful_run()) is None:
                raise RuntimeError('No CNMF results to plot (or multiple gridsearch runs, must select one')
            else:
                logging.info(f'Auto-selected successful run with uuid {uuid}')
                assert self.cnmf_fit is not None
        est = self.cnmf_fit.estimates

        # get background image
        if isinstance(bg_type, str):
            bg_proj = self.get_projection(bg_type)
        else:
            bg_proj, _ = self.get_projection_for_seed(**bg_type)

        # now plot contours on top of correlation image
        if in_jupyter() and not force_mpl:
            est.plot_contours_nb(img=bg_proj, idx=est.idx_components if mark_accepted else None)
            fig = None
        else:
            fig = cmcustom.my_plot_contours(est, img=bg_proj, idx=est.idx_components if mark_accepted else None)
        return fig
    

    def plot_cnmf_rois_mpl(self, bg_type: Union[str, dict] = 'corr', mark_accepted=True) -> Figure:
        """Same as plot_cnmf_rois but always produces a matplotlib figure."""
        fig = self.plot_cnmf_rois(bg_type=bg_type, mark_accepted=mark_accepted, force_mpl=True)
        assert isinstance(fig, Figure)
        return fig


    def save_cnmf_roi_plot(self, mark_accepted=True, filename: Optional[str] = None):
        """By default, saves ROI plot as a PDF with the same base name as the CNMF fit."""
        if filename is None:
            if self.cnmf_fit_filename is None:
                if (uuid := self.select_successful_run()) is None:
                    raise RuntimeError('No CNMF results to plot (or multiple gridsearch runs, must select one')
                else:
                    logging.info(f'Auto-selected successful run with uuid {uuid}')
                    assert self.cnmf_fit_filename is not None
            filename = re.sub(r'\.hdf5$', ('_labeled.pdf' if mark_accepted else '.pdf'), self.cnmf_fit_filename)
        
        with plt.ioff():
            fig = self.plot_cnmf_rois_mpl(mark_accepted=mark_accepted)
            # make bigger to shrink plot elements
            save_contour_plot_as_pdf(fig, filename)
            plt.close(fig)


    def plot_marked_rois(self, k_plane: Optional[int] = None, bg_type: Union[str, dict] = 'mean',
                         ax: Optional[Axes] = None, remove_border=True, cmap='viridis',
                         marked_color: Optional[str] = 'r', unmarked_color: Optional[str] = 'g'):
        """
        Make plot of marked and/or unmarked ROI contours on top of a background image.
        If k_plane is given, only plot ROIs from the given plane.
        If marked_color or unmarked_color is None, contours of that type will not be plotted.
        If remove_border is true, sets x and y bounds to hide border_to_0 pixels around the image.
        If k_plane is None, internal borders will still be visible with remove_border=True.
        """
        if self.cnmf_fit is None:
            raise RuntimeError('No CNMF run available (may need to select)')
        est = self.cnmf_fit.estimates
        idx_marked = est.idx_components_marked
        idx_unmarked = est.idx_components_unmarked
        if idx_marked is None or idx_unmarked is None:
            raise RuntimeError('Marked ROIs not available')
        
        # get background image
        if isinstance(bg_type, str):
            bg_proj = self.get_projection(bg_type)
        else:
            bg_proj, _ = self.get_projection_for_seed(**bg_type)

        # get ROIs
        assert est.A is not None and est.dims is not None, 'No ROIs/dims in CNMF results?'
        A = sparse.csc_matrix(est.A)
        
        # restrict to single plane if requested
        if k_plane is not None:
            plane_pixels = int(np.prod(self.plane_size))
            A = A[k_plane*plane_pixels:(k_plane+1)*plane_pixels, :]
            idx_used = np.flatnonzero(A.getnnz(axis=0) > 0)
            bg_proj = bg_proj[:, k_plane*self.plane_size[1]:(k_plane+1)*self.plane_size[1]]
        else:
            idx_used = np.arange(A.shape[1], dtype=int)

        # create axes if necessary
        if ax is None:
            _, ax = plt.subplots()
        else:
            plt.sca(ax)
        assert ax is not None
        
        # plot background and contours
        if marked_color is not None:
            idx_marked = np.intersect1d(idx_used, idx_marked)
            cmcustom.my_vis_plot_contours(A[:, idx_marked], bg_proj, display_numbers=False,
                                          cmap=cmap, colors=marked_color)
        if unmarked_color is not None:
            idx_unmarked = np.intersect1d(idx_used, idx_unmarked)
            cmcustom.my_vis_plot_contours(A[:, idx_unmarked], bg_proj, display_numbers=False,
                                          cmap=cmap, colors=unmarked_color)
        if remove_border:
            assert self.mc_result is not None, 'No mcorr results?'
            border = self.mc_result.border_to_0
            ax.set_ybound(border-0.5, est.dims[0]-0.5-border)
            xsize = est.dims[1] if k_plane is None else self.plane_size[1]
            ax.set_xbound(border-0.5, xsize-0.5-border)


    def explore_components(self, force_windowed=False, accepted_only=True, order_by='default'):
        # get correlation image
        if self.cnmf_fit is None:
            if (uuid := self.select_successful_run()) is None:
                raise RuntimeError('No CNMF results to save (or multiple gridsearch runs, must select one')
            else:
                logging.info(f'Auto-selected successful run with uuid {uuid}')
                assert self.cnmf_fit is not None
        est = self.cnmf_fit.estimates
        assert est.A is not None and est.idx_components is not None, 'Estimates elements should not be None - has CNMF been run?'

        # determine indices and title
        if accepted_only:
            title = 'Accepted ROIs'
            inds = est.idx_components
        else:
            title = 'All ROIs'
            inds = range(est.A.shape[1])  # type: ignore
        
        if order_by.lower() == 'snr':
            if est.SNR_comp is None:
                raise RuntimeError('Cannot sort by component SNR - not calculated')
            title += ', sorted by SNR'
            inds = sorted(inds, key=lambda i: est.SNR_comp[i])  # type: ignore
        elif 'corr' in order_by.lower():
            if est.r_values is None:
                raise RuntimeError('Cannot sort by spatial correlation - not calculated')
            title += ', sorted by spatial correlation'
            inds = sorted(inds, key=lambda i: est.r_values[i])  # type: ignore
        elif 'cnn' in order_by.lower():
            if est.cnn_preds is None:
                raise RuntimeError('Cannot sort by CNN score - not calculated')
            title += ', sorted by CNN confidence'
            inds = sorted(inds, key=lambda i: est.cnn_preds[i])  # type: ignore
        elif order_by.lower() != 'default':
            raise ValueError('Unrecognized "order_by" value')
                
        corr_concat = self.get_projection('corr')
        
        if in_jupyter() and not force_windowed:
            est.nb_view_components(img=corr_concat, idx=inds, cmap='gray', denoised_color=(0, 0, 1))
        else:   
            est.view_components(img=corr_concat, idx=inds, show_spatial_component=False, display_inds=list(inds))
            plt.suptitle(title)


    def plot_background_components(self):
        if self.cnmf_fit is None:
            raise RuntimeError('CNMF estimates not found')
        est = self.cnmf_fit.estimates
        assert est.f is not None and est.b is not None, 'Estimates elements should not be None - has CNMF been run?'
        
        n_comps = len(est.f)
        fig = plt.figure()
        gs = fig.add_gridspec(2 * n_comps, 1, height_ratios=[3, 1] * n_comps)
        image_dims = self.cnmf_fit.dims

        for i, (comp_s, comp_t) in enumerate(zip(est.b.T, est.f)):  # type: ignore
            im_ax = fig.add_subplot(gs[2*i, 0])
            im_ax.imshow(
                comp_s.reshape(image_dims, order='F'), cmap='gray',
                vmin=np.percentile(comp_s, 10), vmax=np.percentile(comp_s, 99.5))
            im_ax.set_title(f'Component {i+1}')

            t_ax = fig.add_subplot(gs[2*i+1, 0])
            t_ax.plot(comp_t, color='k')
            t_ax.set_xticks([])    
            t_ax.set_xlabel('Time')
            t_ax.set_frame_on(False)

        fig.suptitle('Background components')
        fig.tight_layout()
        return fig


    def show_cnmf_results(self, image_data_options: Optional[list[str]] = None, denoised_temporal=True):
        """
        Make CNMF visualization of completed runs with controls to adjust automatic evaluation & manual accepted/rejected cells,
        linked to this object so that saving updates & saves the parameters if appropriate.
        """
        batch = self.get_gridsearch_results()
        is_completed = [out is not None and out['success'] for out in batch.outputs]
        if not any(is_completed):
            raise RuntimeError('No completed CNMF results to show')

        completed_runs = batch.loc[is_completed, :].reset_index(drop=True)

        if image_data_options is None:
            image_data_options = ['corr', 'mean_equalized']

        start_uuid = self.get_selected_uuid()
        if start_uuid is None or start_uuid not in completed_runs.uuid:
            start_i = -1
        else:
            start_i = list(completed_runs.uuid).index(start_uuid)

        temporal_kwargs = {}
        if not denoised_temporal:
            temporal_kwargs['add_residuals'] = True

        viz: 'caiman_viz.CNMFVizWideContainer' = completed_runs.cnmf.viz_wide(
            image_data_options=image_data_options, image_widget_kwargs={'cmap': 'gray'}, n_planes=self.metadata['num_planes'],
            start_index=completed_runs.iloc[start_i].name, temporal_kwargs=temporal_kwargs)

        viz.on_save(partial(self._update_params_from_viz, viz))
        return viz.show()


    def _update_params_from_viz(self, viz: 'caiman_viz.CNMFVizWideContainer', _save_button):
        """Callback to update eval params from visualization"""
        viz_ind = viz._get_selected_row()
        if viz_ind is None:
            return

        viz_row = cast(MescoreSeries, viz._dataframe.iloc[viz_ind])
        viz_uuid = viz_row.uuid
        cnmf_path = viz_row.cnmf.get_output_path()

        # get updates to save
        eval_updates = {'quality': viz._eval_controller.get_data()}

        curr_uuid = self.get_selected_uuid()
        if viz_uuid == curr_uuid:
            # update params and save - here we don't need to invalidate, since we're just updating it to reflect
            # the processing that has already been done
            self.params, _ = self.params.change_params_and_get_stage_to_invalidate(eval_updates, metadata=self.metadata)
            self.write_params_for_result_file(cnmf_path, cmp.AnalysisStage.EVAL)

            # also save SessionAnalysis & params file, but don't re-save CNMF
            self.save(save_cnmf=False)
        else:
            # only write out new params if a params file already exists; otherwise writing to CNMF file is sufficient
            params_path = paths.params_file_for_result(cnmf_path)
            if os.path.exists(params_path):
                curr_params = cmp.SessionAnalysisParams.read_from_file(params_path)
                new_params, _ = curr_params.change_params_and_get_stage_to_invalidate(eval_updates, metadata=self.metadata)
                new_params.write_params(params_path)


    #-------------------------- STRUCTURAL DATA -----------------------#


    def save_marked_rois(self, structural_sessinfo: 'SessionAnalysis', force=False, force_reload_cnmf=True, plot_results=True,
                         template_params: Optional[dict] = None, structural_template_params: Optional[dict] = None,
                         structural_gSig: Optional[Union[int, Sequence[int]]] = None, subset='accepted', cmap='viridis',
                         align_options: Optional[dict] = None, **register_rois_kwargs):
        """
        Find which ROIs in the current CNMF estimates correspond to cells "marked" by the structural channel.
        Saves this array of ROIs to the estimates object.
        If force is false and the estimates object already contains 'idx_components_marked', skip.
        Then saves the estimates to disk.
        structrual_sessinfo is passed, another SessionAnalysis instance with structural data to use (should be motion corrected)
        template_params: arguments to get_projection_for_seed to use for functional and structural templates;
            by default (with None) uses {'type': 'mean', 'norm_medw': 25}
        structural_gSig: set to an odd integer to use a gSig for extract_binary_masks_from_structural_channel
            that is different from the one used for CNMF (i.e. params.init['gSig'][0])
        align_options: override default tile_and_correct arguments (see default overrides below)
            e.g. for rigid alignment, set max_deviation_rigid to 0.
        Set align_flag=False (one of the register_rois_kwargs) to skip alignment.
        subset: 'accepted' to only consider accepted functional cells; 'all' to use all
        """
        if self.cnmf_fit_filename is None:
            raise RuntimeError('No CNMF results to use')
        
        if force_reload_cnmf or self.cnmf_fit is None:
            # reload from disk (e.g. to sync changes from mesmerize_viz)
            self.cnmf_fit = cnmf_ext.load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=True)

        est = self.cnmf_fit.estimates
        assert isinstance(est.dims, tuple) and len(est.dims) == 2, 'should be 2D CNMF'
        if self.mc_result is None:
            raise RuntimeError('No MC results to use')

        doing_reg = force or est.structural_reg_res is None
        doing_plot = plot_results and in_jupyter()

        if not doing_reg:
            logging.info('Using saved registration')

        if doing_reg or doing_plot:
            if structural_gSig is None:
                structural_gSig = int(self.cnmf_fit.params.init['gSig'][0])
            
            if isinstance(structural_gSig, int):
                structural_gSig = [structural_gSig]

            if any(gSig % 2 == 0 for gSig in structural_gSig):
                raise RuntimeError('Cannot use even number for structural gSig')

            if template_params is None:
                template_params = {'type': 'mean', 'norm_medw': 25}
            
            if structural_template_params is None:
                structural_template_params = template_params

            functional_template = self.get_projection_for_seed(**template_params)[0]
            
            # get masks from structural images
            if structural_sessinfo.mc_result is None:
                raise ValueError('Passed SessionAnalysis does not have motion correction results')
            structural_template = structural_sessinfo.get_projection_for_seed(**structural_template_params)[0]        

            if doing_reg:
                border = max(self.mc_result.border_to_0, structural_sessinfo.mc_result.border_to_0)
                A_structural = cmcustom.my_extract_binary_masks_from_structural_channel(
                    structural_template, gSig=structural_gSig)[0]
                identify_marked_rois(self.cnmf_fit, self.cnmf_fit_filename, A_structural,
                                     structural_template, functional_template, subset=subset,
                                     align_options=align_options, n_planes=self.metadata['num_planes'],
                                     border=border, **register_rois_kwargs)

            assert est.structural_reg_res is not None, 'registration should be done'

            if doing_plot:
                srr = est.structural_reg_res
                if srr.A2_orig is None:
                    logging.warning('Recomputing structural masks - assuming same seed settings')
                    A_structural = cmcustom.my_extract_binary_masks_from_structural_channel(
                        structural_template, gSig=structural_gSig)[0]
                    srr.A2_orig = sparse.csc_matrix(A_structural)

                return caiman_viz.my_check_register_ROIs(matched1=srr.matched1, matched2=srr.matched2,
                                                         unmatched1=srr.unmatched1, unmatched2=srr.unmatched2,
                                                         performance=srr.performance, A1=srr.A1, A2=srr.A2, A2_orig=srr.A2_orig,
                                                         x_remap=srr.x_remap, y_remap=srr.y_remap,
                                                         dims=est.dims,
                                                         background1=functional_template,
                                                         background2=structural_template,
                                                         contour1_name='functional',
                                                         contour2_name='structural',
                                                         cmap=cmap)

        assert est.structural_reg_res is not None, 'registration should be done'
        perf = est.structural_reg_res.performance
        logging.info('Structural masks registered to functional')
        logging.info(f'{perf["recall"] * 100:.2f}% of functional ROIs matched')
        logging.info(f'{perf["precision"] * 100:.2f}% of structural ROIs matched')


    def plot_structural_reg_res(self, structural_sessinfo: 'SessionAnalysis',
                                template_params: Optional[dict] = None, structural_template_params: Optional[dict] = None,
                                structural_gSig: Optional[int] = None, cmap='viridis'):
        """Conveneience function for just plotting structural registration/marked ROI results"""
        if self.cnmf_fit is None or self.cnmf_fit.estimates.structural_reg_res is None:
            raise RuntimeError('No structural registration results to use')

        return self.save_marked_rois(structural_sessinfo=structural_sessinfo, force=False, force_reload_cnmf=False,
                                     plot_results=True, structural_gSig=structural_gSig, 
                                     template_params=template_params, structural_template_params=structural_template_params,
                                     cmap=cmap)


    def map_structural_image(self, im_structural: np.ndarray):
        """
        Use mapping found in save_marked_rois (with align_flag==True) to remap a structural
        image to functional movie space
        """
        if self.cnmf_fit is None:
            raise RuntimeError('No CNMF results to use')
        if self.cnmf_fit.estimates.structural_reg_res is None:
            raise RuntimeError('No structural registration results to use')
        srr = self.cnmf_fit.estimates.structural_reg_res
        if isinstance(srr.x_remap, NAType) or isinstance(srr.y_remap, NAType):
            raise RuntimeError('Structural->functional mapping not saved - have to re-run')
        
        return remap_image(im_structural, x_remap=srr.x_remap, y_remap=srr.y_remap)


    def get_mapped_structural_projection(self, sessinfo_structural: 'SessionAnalysis',
                                         proj_type: Union[str, dict]) -> np.ndarray:
        """
        Get a projection from the given structural movie, and then use mapping 
        found in save_marked_rois (with align_flag==True) to map this into the
        functional movie space.
        """
        # get the projection 
        if isinstance(proj_type, str):
            proj = sessinfo_structural.get_projection(proj_type)
        else:
            proj, _ = sessinfo_structural.get_projection_for_seed(**proj_type)
        return self.map_structural_image(proj)

    def get_merge_image(self, sessinfo_structural: 'SessionAnalysis', proj_type: Union[str, dict],
                        color_functional: Union[Sequence[float], str] = 'g',
                        color_structural: Union[Sequence[float], str] = 'r',
                        clip_percentile=0.1) -> np.ndarray:
        """Make an RGB image with the merge of functional/structural projections (with registration)"""
        im_structural = self.get_mapped_structural_projection(sessinfo_structural, proj_type=proj_type)
        if isinstance(proj_type, str):
            im_functional = self.get_projection(proj_type)
        else:
            im_functional, _ = self.get_projection_for_seed(**proj_type)
        
        return make_merge(im_functional, im_structural, color1=color_functional, color2=color_structural,
                          clip_percentile=clip_percentile)
    

    #------------------------- MISCELLANEOUS -------------------------#

    
    def get_dims_in_um(self) -> tuple[Union[slice, np.ndarray], ...]:
        """
        Gets arrays of pixel locations in each dimension, in um.
        Currently requires self.cnmf_fit to be populated, although that could change if necessary.
        """
        if self.cnmf_fit is None:
            raise RuntimeError('No CNMF results to use')
        est = self.cnmf_fit.estimates
        if not isinstance(est.dims, tuple):
            raise RuntimeError('CNMF has not been run')

        dims = [slice(0, est.dims[0] * self.metadata['um_per_pixel_y'], self.metadata['um_per_pixel_y']),
                slice(0, est.dims[1] * self.metadata['um_per_pixel_x'], self.metadata['um_per_pixel_x'])]
        if len(est.dims) == 3:
            # add plane locations according to etl table
            dims.append(self.metadata['etl_pos'])
        return tuple(dims)


    def save_roi_thumbnails(self, roi_ids: Optional[Sequence[int]] = None, proj_type: Union[str, dict] = 'mean',
                            box_size=25, vmin_pct=1., vmax_pct=99.96, cmap='gray', force=False) -> list[str]:
        """
        Save square thumbnails of projections around each requested ROI (or all by default).
        Save location is "thumbnails" subdir of the export folder. Returns a list of saved paths.
        """
        if self.cnmf_fit is None:
            raise RuntimeError('CNMF not done or not selected')
        est = self.cnmf_fit.estimates
        if est.A is None:
            raise RuntimeError('CNMF not done or not selected')
        uuid = self.get_selected_uuid()
        assert uuid is not None, 'Selected CNMF run does not have a UUID?'

        n_comps = est.A.shape[1]  # type: ignore

        if roi_ids is None:
            roi_ids = range(n_comps)

        if any([id not in range(n_comps) for id in roi_ids]):
            raise RuntimeError('Not all roi_ids are in range')
        
        # make save dir
        thumbnail_dir = os.path.join(self.data_dir, 'export', 'thumbnails')
        if not os.path.exists(thumbnail_dir):
            os.makedirs(thumbnail_dir, exist_ok=True)

        # pre-compute save paths
        tagstr = '_' + self.tag if self.tag else ''
        fns = [f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_{uuid}_roi{roi_id}_{box_size}x{box_size}.png'
               for roi_id in roi_ids]
        save_paths = [os.path.join(thumbnail_dir, fn) for fn in fns]

        # get centers of mass
        dims = cast(tuple[int, int], est.dims)
        A = sparse.csc_array(est.A)
        coms: onp.Array2D[np.floating] = rois.com(A[:, roi_ids], *dims)  # type: ignore

        # get projection
        if isinstance(proj_type, str):
            proj = self.get_projection(proj_type)
        else:
            proj, _ = self.get_projection_for_seed(**proj_type)
        
        fig, ax = plt.subplots()
        thumbnail = np.empty((box_size, box_size), dtype=proj.dtype)

        for roi_id, path, com in zip(roi_ids, save_paths, tqdm(coms, desc='Saving thumbnails...', unit='ROI')):
            if not force and os.path.exists(path):
                logging.info(f'Skipping {roi_id} since it is already saved')
                continue

            com_y = float(com[0])
            com_x = float(com[1])

            # get data for image, filling with zeros if necessary
            top = round(com_y - box_size / 2)
            bottom = top + box_size
            left = round(com_x - box_size / 2)
            right = left + box_size
            nclip_top = max(0, -top)
            nclip_bottom = max(0, bottom - dims[0])
            nclip_left = max(0, -left)
            nclip_right = max(0, right - dims[1])
            thumbnail[:] = 0
            data = proj[top+nclip_top:bottom-nclip_bottom, left+nclip_left:right-nclip_right]
            thumbnail[nclip_top:box_size-nclip_bottom, nclip_left:box_size-nclip_right] = data

            # plot, taking colormap etc. into account
            vmin = float(np.percentile(data[~np.isnan(data)], vmin_pct))
            vmax = float(np.percentile(data[~np.isnan(data)], vmax_pct))
            ax.clear()
            ax.imshow(thumbnail, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_axis_off()

            # save as PNG
            fig.savefig(path, bbox_inches='tight', pad_inches=0)

        return save_paths

def get_session_analysis_file_pattern(mouse_id: Union[str, int], sess_id: int, tag: Optional[str]) -> str:
    tagstr = '_' + tag if tag else ''
    return f'{mouse_id}_{sess_id:03d}{tagstr}_%dt.pkl'


@lru_cache  # must clear cache when new CNMF results are saved!
def load(filename: str, quiet=True, lazy=True, **field_overrides) -> SessionAnalysis:
    with open(filename, 'rb') as info_file:
        if not quiet:
            logging.info(f'Loading session analysis from {filename}')
        loaded_fields: dict[str, Any] = pickle.load(info_file)
    loaded_fields.update(field_overrides)

    sessdata = SessionAnalysis(loaded_info=loaded_fields)

    # load CNMF
    sessdata._cnmf_fit = None
    if sessdata.cnmf_fit_filename is not None and not lazy:
        sessdata._cnmf_fit = load_cnmf(sessdata.cnmf_fit_filename, quiet=quiet)
        if sessdata._cnmf_fit is None:
            sessdata.cnmf_fit_filename = None
    return sessdata


def load_cnmf(cnmf_filename: str, quiet=True) -> Optional['cnmf_ext.CNMFExt']:
    """
    Tries to load CNMF results
    """
    if not quiet:
        logging.info(f'Loading CNMF results from {cnmf_filename}')
    try:
        cnmf_obj = cnmf_ext.load_CNMFExt(cnmf_filename, dview=cluster.dview, quiet=quiet)
    except FileNotFoundError:
        logging.warning('CNMF file could not be found; not loaded')
        return None
    else:
        # set n_processes since refit reads from it and passing this to load_CNMF doesn't actually do anything
        # also blank out fnames to avoid check; will set it before actually running any analysis
        cnmf_obj.params.change_params({'data': {'fnames': None}, 'patch': {'n_processes': cluster.ncores}})
        return cnmf_obj


def load_latest(mouse_id: Union[int, str], sess_id: int, rec_type: str = 'learning_ppc',
                tag: Optional[str] = None, quiet=True, lazy=True) -> SessionAnalysis:
    """Load latest saved analysis for given mouse/session/tag"""
    data_dir = paths.get_processed_dir(mouse_id, rec_type=rec_type)
    file_pattern = get_session_analysis_file_pattern(mouse_id, sess_id, tag=tag)
    latest_file = paths.get_latest_timestamped_file(data_dir, file_pattern)
    if latest_file is None:
        raise RuntimeError('No saved analysis found')
    return load(latest_file, quiet=quiet, lazy=lazy)


def load_cell_metadata(mouse_id: Union[int, str], sess_id: int, uuid: str,
                       tag: Optional[str] = None, rec_type='learning_ppc'):
    """Load cell info from a specific CNMF run (with uuid)"""
    data_dir = paths.get_processed_dir(mouse_id, rec_type=rec_type)
    export_dir = os.path.join(data_dir, 'export')
    tagstr = '_' + tag if tag else ''
    filename = os.path.join(export_dir, f'{mouse_id}_{sess_id:03d}{tagstr}_{uuid}_metadata.pkl')
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError as e:
        raise RuntimeError(f'Could not find cell metadata file for {mouse_id}/{sess_id}{tagstr}') from e


def load_selected_metadata(mouse_id: Union[int, str], sess_id: int, tag: Optional[str] = None, rec_type='learning_ppc'):
    """Loads cell info from the selected CNMF run (usually the last one)"""
    sessinfo = load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, lazy=True)
    uuid = sessinfo.get_selected_uuid()
    if uuid is None:
        tagstr = '_' + tag if tag is not None else ''
        raise RuntimeError(f'No selected CNMF run found for {mouse_id}/{sess_id}{tagstr}')
    return load_cell_metadata(mouse_id, sess_id, uuid=uuid, tag=tag, rec_type=rec_type)


def evaluate_cnmf(cnmf_fit: 'cnmf_ext.CNMFExt', mc_res_path_or_images: Union[str, np.ndarray], snr_type: Literal['normal', 'gamma'] = 'gamma',
                  new_quality_params: Optional[dict] = None, recalc=False, n_planes=1, dview=None):
    """
    Do CNMF evaluation. If the image is 2D and n_planes > 1 (meaning planes are concatenated along X), takes the additional step of
    rejecting all ROIs with center of mass between planes (i.e., within border_pix pixels of the edge of any plane).
    Otherwise, it is easier to handle by setting the border_pix patch parameter, so this step is skipped.
    """
    if isinstance(mc_res_path_or_images, str):
        Yr, dims, num_frames = cm.load_memmap(mc_res_path_or_images)
        images = np.reshape(Yr.T, [num_frames] + list(dims), order='F')
    else:
        images = mc_res_path_or_images
    est = cnmf_fit.estimates_base

    if not isinstance(est.dims, tuple) or est.A is None:
        raise RuntimeError('CNMF not run; cannot do CNMF evaluation')

    logging.info('Doing CNMF component evaluation')
    cnmf_fit.estimates.snr_type = snr_type
    if recalc:
        if new_quality_params is not None:
            cnmf_fit.params.change_params({'quality': new_quality_params})
        cnmf_fit.estimates.evaluate_components(images, cnmf_fit.params, dview=dview)
    else:
        if new_quality_params is None:
            new_quality_params = {}
        cnmf_fit.estimates.filter_components(images, cnmf_fit.params, new_dict=new_quality_params, dview=dview)
    assert est.idx_components_bad is not None, 'idx_components_bad must be populated after running evaluate_components'

    # Border rejection
    border_pix = cnmf_fit.params.patch['border_pix']
    if n_planes > 1 and border_pix > 0 and len(est.dims) == 2:
        # Get COM for each ROI
        coms = cmcustom.my_com(est.A, *est.dims)
        y_size = est.dims[0]
        x_size = est.dims[1] // n_planes
        reject_y = np.abs(coms[:, 0] + 0.5 - y_size / 2) > (y_size / 2 - border_pix)
        reject_x = np.abs(coms[:, 1] % x_size + 0.5 - x_size / 2) > (x_size / 2 - border_pix)
        reject = np.flatnonzero(reject_y | reject_x)
        est.idx_components_bad = np.union1d(est.idx_components_bad, reject)
        est.idx_components = np.setdiff1d(range(est.A.shape[-1]), est.idx_components_bad)


def identify_marked_rois(cnmf_obj: 'cnmf_ext.CNMFExt', cnmf_filename: Optional[str], A_structural: sparse.csc_array,
                         structural_template: np.ndarray, functional_template: np.ndarray,
                         n_planes=1, border: Union[BorderSpec, int] = 0, subset='all', **register_rois_kwargs):
    """
    Find which ROIs in the current CNMF estimates correspond to cells "marked" by the structural channel.
    Saves this array of ROIs to the estimates object.
    If force is false and the estimates object already contains 'idx_components_marked', skip.
    Then saves estimates to disk.
    Note it is probably a good idea to change n_planes and border from the defaults depending on the data.
    """
    est = cnmf_obj.estimates
    if est.A is None:
        raise RuntimeError('CNMF has not been run')
    n_comps = est.A.shape[1]

    assert isinstance(est.dims, tuple) and len(est.dims) == 2, 'should be 2D CNMF'

    if subset == 'all':
        used_comps = np.arange(n_comps, dtype=int)
    elif subset == 'accepted':
        if est.idx_components is None:
            raise RuntimeError('Component evaluation not run')
        used_comps = est.idx_components
    else:
        raise RuntimeError(f'Subset {subset} not valid')
    
    if isinstance(est.A, (sparse.sparray, sparse.spmatrix)):
        A_functional = sparse.csc_matrix(est.A)[:, used_comps]  # type: ignore
    else:
        A_functional = est.A[:, used_comps]

    structural_reg_res = alignment.register_ROIs(
        A_functional, A_structural, est.dims, template1=functional_template, template2=structural_template,
        use_opt_flow=False, n_planes=n_planes, border=border, **register_rois_kwargs
    )

    structural_reg_res.components_used = used_comps
    est.structural_reg_res = structural_reg_res

    if cnmf_filename is not None:
        cnmf_obj.save(cnmf_filename)
        cnmf_ext.clear_cnmf_cache()
    else:
        logging.warning('CNMF not saved - no filename')


def calc_df_over_f(est: cnmf_ext.EstimatesExt, use_residuals=True, roi_subset: Optional[Sequence[int]] = None,
                   detrend_window=500) -> np.ndarray:
    """
    Calculates dF/F using CaImAn background components to define baseline, which is the default in
    CaImAn. TODO see whether it makes more sense to use something closer to the raw data for baseline.
    The default here is now to use residuals, i.e. not just rescale the "denoised" traces (C) from CaImAn.
    However, both could potentially be useful - including noise is more trustworthy for initial evaluation/
    data inspection, whereas the denoised dF/F could be better to feed into a downstream analysis.

    Update 8/11/25: considering changing the default number of frames the baseline is calculated over from 500 to 5000
    because I saw that what was being removed looked more like a scaled version of the original data than a trend.
    However, still looking into this.
    """
    if est.A is None or est.b is None or est.C is None or est.f is None or est.YrA is None:
        raise RuntimeError('CNMF fit is not complete; do not have data needed for dF/F')

    if roi_subset is not None:
        A = est.A[:, roi_subset]
        C = est.C[roi_subset]
        YrA = est.YrA[roi_subset]
    else:
        A = est.A
        C = est.C
        YrA = est.YrA

    if use_residuals:
        dff = detrend_df_f(A, est.b, C, est.f, YrA, flag_auto=False, frames_window=detrend_window)
    else:
        dff = detrend_df_f(A, est.b, C, est.f, None, flag_auto=False, frames_window=detrend_window)
    assert isinstance(dff, np.ndarray), 'detrend_df_f returned something unexpected'
    return dff


def tabularize_estimates_metadata(estimates_data: dict, *, com_3d: ScaledDataFrame, SNR_comp: np.ndarray, r_values: np.ndarray,
                                  cnn_preds: np.ndarray, trial_numbers: np.ndarray, frames_per_trial: np.ndarray,
                                  idx_components: np.ndarray, idx_components_eval: np.ndarray,
                                  idx_components_marked: Optional[np.ndarray] = None,
                                  accepted_list: np.ndarray, rejected_list: np.ndarray, **other_metadata) -> dict[str, Any]:
    """Reformat CNMF metadata into pandas tables to make it easier to query"""
    cells_table = pd.DataFrame({
        'cell_id': range(len(SNR_comp)),
        'com_y': com_3d.y.to_numpy(),
        'com_x': com_3d.x.to_numpy(),
        'com_z': com_3d.plane.to_numpy() if 'plane' in com_3d else 0,
        'snr': SNR_comp,
        'spatial_corr': r_values,
        'cnn_score': cnn_preds if len(cnn_preds) > 0 else np.nan,
        'auto_accepted': False,
        'manual_accepted': False,
        'manual_rejected': False,
        'accepted': False,
        'mean_dff': np.mean(estimates_data['F_dff_denoised'], axis=1)
    })
    cells_table.loc[idx_components_eval, 'auto_accepted'] = True
    cells_table.loc[accepted_list, 'manual_accepted'] = True
    cells_table.loc[rejected_list, 'manual_rejected'] = True
    cells_table.loc[idx_components, 'accepted'] = True
    if idx_components_marked is not None:
        cells_table.loc[:, 'matched_to_structural'] = False
        cells_table.loc[idx_components_marked, 'matched_to_structural'] = True
    else:
        cells_table.loc[:, 'matched_to_structural'] = np.nan

    trials_table = pd.DataFrame({
        'imaging_trial_number': trial_numbers,
        'n_frames': frames_per_trial
    })

    return {'cells': cells_table, 'trials': trials_table, 'pixel_spacing': com_3d.um_per_pixel,
            **other_metadata}