import asyncio
from collections import Counter
from copy import copy, deepcopy
from datetime import date
from functools import lru_cache
from itertools import product, accumulate, pairwise
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
from subprocess import CalledProcessError
from typing import Optional, Union, Any, Iterable, Sequence, cast, Literal

import cv2
from hdf5storage import savemat
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType
from roipoly import MultiRoi
from scipy import sparse
from scipy.spatial import KDTree
from scipy.stats import mode
from tifffile import memmap as tiff_memmap
from trycast import isassignable

import caiman as cm
from caiman.base.movies import get_file_size, load_iter
from caiman.base import rois
from caiman.source_extraction.cnmf import cnmf
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.spatial import threshold_components
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.utils import sbx_utils
from caiman.utils.visualization import view_quilt
import mesmerize_core as mc
from mesmerize_core.algorithms._utils import make_projection_parallel, make_correlation_parallel
from mesmerize_core.caiman_extensions.cnmf import cnmf_cache
from mesmerize_core.caiman_extensions.common import Waitable, DummyProcess
from mesmerize_core.utils import get_params_diffs

from cmcode import in_jupyter, alignment, cmcustom, gridsearch_analysis, mcorr
from cmcode.caiman_params import make_cnmf_params, get_default_seed_params
from cmcode.remote import remoteops
from cmcode.gridsearch_analysis import ParamGrid
from cmcode.cnmf_ext import CNMFExt, EstimatesExt, load_CNMFExt
from cmcode.mcorr import PiecewiseMCInfo, MCResult  # to allow unpickling
from cmcode.util import footprints, paths
from cmcode.util.cluster import Cluster
from cmcode.util.compat import reconstruct_sessdata_obj
from cmcode.util.image import make_merge, remap_image, BorderSpec, preprocess_proj_for_seed
from cmcode.util.sbx_data import find_sess_sbx_files, get_trial_numbers_from_files
from cmcode.util.scaled import ScaledDataFrame, make_um_df, make_pixel_df
from cmcode.util.types import NoBatchFileError, MaybeSparse, NoMatchingResultError, MescoreBatch, MescoreSeries


if in_jupyter():
    from . import caiman_viz
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

# exception identification
NO_FILES_MSG = 'No .sbx files found'

# parameters to not show in diffs between CNMF runs
EXCLUDE_FROM_DIFFS = [
    'online.path_to_model',  # depends on caiman data dir
    'online.init_batch',
    'online.movie_name_online',
    'preprocess.n_pixels_per_process',
    'spatial.n_pixels_per_process',
    'patch.n_processes',
    'data.caiman_version',  # could be irrelevant at some point, but don't show for now
    'data.last_commit',
    'data.fnames'
]

# this is a global variable
cluster = Cluster()

def setup_cluster(*args, **kwargs):
    """Start the global cluster (or initialize with 'single' backend to not use a cluster)"""
    logging.warning('cma.setup_cluster is deprecated, use cma.cluster.start instead')
    cluster.start(*args, **kwargs)


def get_mouse_params(mouse_id: Union[int, str]) -> dict:
    """Get mouse-specific parameter overrides"""
    root_data_dir = paths.get_root_data_dir()
    params_filename = root_data_dir / 'mouse_params' / f'params_{mouse_id}.json'
    if params_filename.exists():
        # load json and select update params with "main" entry
        with open(params_filename, 'r') as json_fh:
            mouse_params = json.load(json_fh)
        
        # special-case processing for non-JSON data
        if 'crop' in mouse_params:
            mouse_params['crop'] = BorderSpec(**mouse_params['crop'])
    else:
        mouse_params = {}
    return mouse_params


def save_contour_plot_as_pdf(fig: Figure, filename):
    """Save contour plot figure to a high-resolution PDF"""
    w, h = fig.get_size_inches()
    fig.set_figwidth(w * 15)
    fig.set_figheight(h * 15)
    fig.savefig(filename)


def get_projection_name(seed_params: dict) -> str:
    """
    Make a string to identify the type of projection
    (like get_spatial_seed_name but ignores parameters only used for identifying neurons)
    """
    proj_type = seed_params.get('type', 'mean')
    norm_medw = seed_params.get('norm_medw', None)
    blur_size = seed_params.get('blur_size', 1)

    proj_name = proj_type
    if blur_size > 1:
        proj_name += f'_blur_{blur_size}'
    if norm_medw is not None:
        proj_name += f'_medw_{norm_medw}'
    
    return proj_name


def get_spatial_seed_name(seed_params: dict) -> str:
    """Make a string to identify the type of seed (used as filename)"""
    proj_name = get_projection_name(seed_params)
    seed_blur_multiple = seed_params.get('blur_gSig_multiple', 0.75)

    Ain_name = f'Ain_caiman_from_{proj_name}'
    if 'gSig' in seed_params:  # just leave out of name if not specified
        gSig = seed_params['gSig']
        if not hasattr(gSig, '__len__'):
            gSig = [gSig]
        Ain_name += '_gSig_' + ','.join(str(gs) for gs in gSig)
    if seed_blur_multiple != 0.75:
        Ain_name += f'_blurmult_{seed_blur_multiple:.2f}'
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
                   'mc_result', 'cnmf_fit2_filename', 'cnmf_fit_filename', 'gridsearch_batch_path']

    def __init__(self, mouse_id: Union[int, str] = 0, sess_id=0, trials_to_include: Optional[Sequence[int]] = None, *_,
                 rec_type='learning_ppc', tag: Optional[str] = None, loaded_info: Optional[dict[str, Any]] = None,
                 sess_filename: Optional[str] = None, param_overrides: Optional[dict] = None,
                 downsample_factor: Optional[int] = None, trials_to_exclude: Optional[Sequence[int]] = None,
                 highpass_cutoff=0.):
        """
        Make an object to manage analyzing one session (or part of a session, if trials_to_include is given)
        If loaded_info is not none, ignores all other arguments and constructs from the passed dict.
        downsample_factor can be given to downsample data and then interpolate at the end. A string f'_ds{downsample_factor}'
        is appended to the tag. If the tag excluding this string matches an existing run, the corrected/converted data from that run
        may be used as a basis for saving the downsampled tif.
        """
        # fields that aren't saved and thus must be set no matter what
        self.cluster_args: Optional[dict] = None
        self.sess_filename: str = '' if sess_filename is None else sess_filename  # .pkl file to save to (gets set on first save)
        self._cnmf_fit: Optional[CNMFExt] = None  # CNMF results

        if loaded_info is None:
            # set defaults/passed-in parameters
            if tag == '':
                logging.warning('Empty string tag is interpreted as no tag (just use None)')
                tag = None

            self.tag_base: Optional[str] = tag
            if downsample_factor is None:
                self.tag: Optional[str] = self.tag_base
            else:
                tags = [tag] if tag is not None else []
                tags.append(f'ds{downsample_factor}')
                self.tag = '_'.join(tags)
            self.mouse_id: Union[int, str] = mouse_id
            self.sess_id: int = sess_id
            self.data_dir: str = ''
            self.sbx_files: list[str] = []
            self.metadata: dict[str, Any] = {}
            self.crop = BorderSpec()  # how to crop each plane
            self.downsample_factor: Optional[int] = downsample_factor  # process only every Nth frame, then interpolate CNMF results to original size.
            self.highpass_cutoff: float = highpass_cutoff  # highpass filter cutoff to use after motion correction

            self.snr_type: Literal['normal', 'gamma'] = 'gamma'
            self.crossplane_merge_thr: Optional[float] = 0.7

            # initialize fields for later results to None
            self.odd_row_offset: Optional[int] = None            # row alignment correction to apply (if recording is bidirectional)
            self.odd_row_ndeads: Optional[list[int]] = None      # number of saturated pixels on odd rows for each sbx file
            self.frames_per_trial: Optional[np.ndarray] = None   # frames concatenated from each sbx file (i.e., each trial) into tif file
            self.plane_tifs: Optional[list[str]] = None          # paths to plane TIF files
            self.mc_result: Optional[MCResult] = None            # results of motion correction
            self.cnmf_fit_filename: Optional[str] = None         # Path to save location for CNMF results (HDF5 file)
            self.params_to_search: Optional[Union[ParamGrid, Sequence[ParamGrid]]] = None   # for gridsearch
            self.gridsearch_batch_path: Optional[str] = None     # path to dataframe for use with mesmerize-core for gridsearch

            # get mouse-specific overrides
            mouse_params = get_mouse_params(self.mouse_id)

            # get specifically-passed overrides
            if param_overrides is None:
                param_overrides = {}

            # annoyingly we have to apply any snr_type override here as a special case so we have it for make_cnmf_params
            if 'snr_type' in param_overrides:
                self.snr_type = param_overrides['snr_type']
            elif 'snr_type' in mouse_params:
                self.snr_type = mouse_params['snr_type']

            self.data_dir = paths.get_processed_dir(self.mouse_id, rec_type=rec_type, create_if_not_found=True)
            self.sbx_files = find_sess_sbx_files(mouse_id, sess_id, trials_to_include=trials_to_include, trials_to_exclude=trials_to_exclude,
                                                 rec_type=rec_type, remove_ext=True)
            if len(self.sbx_files) > 0:
                logging.info('Files found:\n' +
                              str(self.sbx_files) if len(self.sbx_files) < 5 else 
                              f'[{self.sbx_files[0]}, {self.sbx_files[1]}, ..., {self.sbx_files[-2]}, {self.sbx_files[-1]}]')
            else:
                raise RuntimeError(NO_FILES_MSG)

            meta = sbx_utils.sbx_meta_data(self.sbx_files[0])
            logging.info('Shape of first file: ' + 
                         f'({meta["num_frames"]}, {meta["frame_size"][0]}, {meta["frame_size"][1]}, {meta["num_planes"]})')
            self.metadata = meta
            self.cnmf_params = make_cnmf_params(meta, dims=2, snr_type=self.snr_type, downsample_factor=downsample_factor)
            self.seed_params = get_default_seed_params(meta)

            # now apply overrides on top of defaults
            self.update_params(mouse_params)
            self.update_params(param_overrides)
        else:
            reconstruct_sessdata_obj(self, loaded_info)

            # blank out machine-specific params; will set if necessary before running any analysis
            self.cnmf_params.change_params({
                'data': {'fnames': None},
                'preprocess': {'n_pixels_per_process': None},
                'spatial': {'n_pixels_per_process': None}
            })

            # apply any specific overrides
            if param_overrides is not None:
                self.update_params(param_overrides)

    @property
    def trial_numbers(self) -> np.ndarray:
        nums, b_valid = get_trial_numbers_from_files(self.sbx_files)
        if not all(b_valid):
            raise RuntimeError('Not all trials have trial numbers')
        return nums


    @property
    def cnmf_fit(self) -> Optional[CNMFExt]:
        """Lazy loader for CNMF results"""
        if self.cnmf_fit_filename is not None and self._cnmf_fit is None:
            loaded_data = load_cnmf(self.cnmf_fit_filename, metadata=self.metadata, quiet=True)
            if loaded_data is None:
                self.cnmf_fit_filename = None
            else:
                self._cnmf_fit = loaded_data
        return self._cnmf_fit

    @cnmf_fit.setter
    def cnmf_fit(self, value: CNMFExt):
        self._cnmf_fit = value

    
    @property
    def scan_day(self) -> Optional[date]:
        """Infer what day the scan took place from modified time"""
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
        if self.downsample_factor is not None:
            sample_rate /= self.downsample_factor
        return sample_rate
    

    @property
    def plane_size(self) -> tuple[int, int]:
        return self.crop.center_shape(self.metadata['frame_size'])


    def update_params(self, param_changes: dict):
        """
        Update params, also allowing changes to some things that are not part of CNMFParams:
        (note that for dicts, unlike for sub-dicts of CNMFParams, the whole dict will be replaced rather than updated)
        - "seed_params" -> params for constructing projection and seed for initializing CNMF (updates self.seed_params)
        - "cluster_args" -> multiprocessing options (updates self.cluster_args)
        - "snr_type" -> change SNR type ("normal" or "gamma")
        - "crossplane_merge_thr" -> change threshold for crossplane merging (or None to skip crossplane merging)
        All other sub-fields of CNMFParams may also be present as keys.
        """
        # TODO I should subclass CNMFParams and add another field for my custom params to clean things up and make them more visible
        param_changes = param_changes.copy()
        updatable_fields = ['seed_params', 'cluster_args', 'snr_type', 'crossplane_merge_thr', 'highpass_cutoff', 'crop']

        for fieldname in updatable_fields:
            if fieldname in param_changes:
                setattr(self, fieldname, param_changes.pop(fieldname))
        
        self.cnmf_params.change_params(param_changes)


    def save(self, save_cnmf: bool = True):
        if self.sess_filename == '':
            file_pattern = get_session_analysis_file_pattern(
                self.mouse_id, self.sess_id, tag=self.tag_base, downsample_factor=self.downsample_factor)  
            filename = paths.make_timestamped_filename(file_pattern)
            self.sess_filename = os.path.join(self.data_dir, filename)

        logging.info(f'Saving session analysis to {self.sess_filename}')
        fields_to_skip = ['sess_filename', 'cluster_args', 'cnmf_fit1', 'cnmf_fit2',
                          'cnmf_fit1_filename', 'cnmf_fit2_filename', '_cnmf_fit']
        fields_to_save = {name: val for (name, val) in vars(self).items() if name not in fields_to_skip}

        # relativize paths
        for field in self.PATH_FIELDS:
            if field in fields_to_save:
                fields_to_save[field] = paths.relativize_path(fields_to_save[field])

        with open(self.sess_filename, 'wb') as info_file:
            pickle.dump(fields_to_save, info_file)
            
        if save_cnmf and self.cnmf_fit is not None and self.cnmf_fit_filename is not None:
            logging.info(f'Saving CNMF to {self.cnmf_fit_filename}')
            self.cnmf_fit.save(self.cnmf_fit_filename)
        
        load.cache_clear()


    #--------------------------- PREPROCESSING --------------------------------#


    def preview_raw_data(self, frames_to_average: Union[int, slice] = 50, channel=0, title: Optional[str] = None):
        """Display interface to preview N frames of raw data and adjust bidirectional offset"""
        # local import because canvas might not be available
        if not in_jupyter():
            raise RuntimeError('preview_raw_data only available in Jupyter')

        def save_callback(new_offset: int):
            self.odd_row_offset = new_offset
            self.save(save_cnmf=False)

        widget = caiman_viz.RawDataPreviewContainer(self.sbx_files, frames=frames_to_average, curr_offset=self.odd_row_offset,
                                                    offset_save_callback=save_callback, channel=channel, title=title)
        return widget.show()


    def convert_to_tif(self, to3D=False, force=False, channel=0, **convert_kwargs):
        """
        Concatenate sbx files and convert each plane (by default) or the entire 3D movie to .tif file(s).
        To make sure the frame rate remains mostly correct, if subindices are passed, for each file the
        indices along time must either be a slice with step of 1 or an array where np.diff(subinds) is majority 1
        (could still subvert by doing something stupid, so don't do that)
        """
        # use odd_row_offset unless explicitly given
        if 'odd_row_offset' not in convert_kwargs and self.odd_row_offset is not None:
            convert_kwargs['odd_row_offset'] = self.odd_row_offset

        # get nsaturated explicitly for bidirectional recordings so we can use it later
        if self.metadata['scanning_mode'] == 'bidirectional':
            if 'odd_row_ndead' in convert_kwargs:
                ndead = convert_kwargs['odd_row_ndead']
                self.odd_row_ndeads = list(ndead) if isinstance(ndead, Sequence) else [ndead] * len(self.sbx_files)
            else:
                if self.odd_row_ndeads is None:
                    self.odd_row_ndeads = [sbx_utils.get_odd_row_ndead(f) for f in self.sbx_files]
                convert_kwargs['odd_row_ndead'] = self.odd_row_ndeads
        else:
            self.odd_row_ndeads = None

        def convert_one(filename: str, plane: Optional[int], filename_base: str):
            """
            Convert a single plane. If downsample_factor is not None, also downsample, either while converting
            or from filename_base if it exists.
            """
            if self.frames_per_trial is None:
                # do not care about downsampling for frames per trial
                logging.debug('Frames per file is missing; inferring') 
                self.frames_per_trial = np.array([sbx_utils.sbx_shape(fn)[-1] for fn in self.sbx_files])
                self.save(save_cnmf=False)
            assert self.frames_per_trial is not None

            if not force and os.path.isfile(filename):
                logging.info((f'Plane {plane}: ' if plane is not None else '') + 'using existing .tif file for this mouse/session.')
            elif not force and self.downsample_factor is not None and os.path.isfile(filename_base):
                # Re-save downsampled version
                logging.info(f'Plane {plane}: saving downsampled version of {filename_base} to {filename}')
                slices_orig = []  # slices of original data
                frames_per_file_ds = []
                for start, stop in pairwise(accumulate(self.frames_per_trial, initial=0)):
                    frames_per_file_ds.append(math.ceil((stop - start) / self.downsample_factor))
                    slices_orig.append(slice(start, stop, self.downsample_factor))

                # create file to hold downsampled data
                mmap_orig = tiff_memmap(filename_base, mode='r')
                shape_orig = mmap_orig.shape
                dtype = mmap_orig.dtype
                shape_ds = (sum(frames_per_file_ds),) + shape_orig[1:]
                _, out_memmap_args = sbx_utils._prepare_concat_output_memmap(
                    filename, shape_ds, frames_per_file_ds, dtype=dtype,
                    bigtiff=convert_kwargs.get('bigtiff', True), imagej=convert_kwargs.get('imagej', False)
                )

                # save downsampled data to file
                for in_slice, out_args in zip(tqdm(slices_orig, desc='Downsampling each trial...', unit='trial'), out_memmap_args):
                    out_memmap = np.memmap(**out_args)
                    out_memmap[:] = mmap_orig[in_slice]             
            else:
                logging.info((f'Plane {plane}: ' if plane is not None else '') +
                             f'converting {len(self.sbx_files)} .sbx file(s) into one .tif file...')
                
                # deal with downsampling
                if self.downsample_factor is not None:
                    subindices = (slice(None, None, self.downsample_factor),) + self.crop.slices(self.metadata['frame_size'])
                else:
                    subindices = (slice(None),) + self.crop.slices(self.metadata['frame_size'])

                sbx_utils.sbx_chain_to_tif(self.sbx_files, fileout=filename, channel=channel, plane=plane,
                                           dead_pix_mode=self.cnmf_params.motion['border_nan'],
                                           dview=cluster.dview, subindices=subindices, **convert_kwargs)
      
        conversion_dir = os.path.join(self.data_dir, 'conversion')
        os.makedirs(conversion_dir, exist_ok=True)
        tagstr_base = '_' + self.tag_base if self.tag_base is not None else ''
        tagstr = '_' + self.tag if self.tag is not None else ''

        if to3D:
            tif_filename_base = os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr_base}.tif')
            tif_filename = os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}.tif')
            convert_one(tif_filename, plane=None, filename_base=tif_filename_base)
            tif_filenames = [tif_filename]
        else:
            tif_filenames_base = [os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr_base}_plane{plane}.tif')
                             for plane in range(self.metadata['num_planes'])]
            tif_filenames = [os.path.join(conversion_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_plane{plane}.tif')
                                for plane in range(self.metadata['num_planes'])]
            
            if (len(tif_filenames) == 1):
                convert_one(tif_filenames[0], plane=None, filename_base=tif_filenames_base[0])
            else:
                for i, (filename, filename_base) in enumerate(zip(tif_filenames, tif_filenames_base)):
                    convert_one(filename, plane=i, filename_base=filename_base)
        
        self.plane_tifs = tif_filenames
        self.save(save_cnmf=False)
        

    def convert_functional_to_tif(self, force: bool = False, **convert_kwargs) -> None:
        logging.warning('Deprecated, just use convert_to_tif')
        self.convert_to_tif(to3D=False, force=force, channel=0, **convert_kwargs)


    #-------------------------- MOTION CORRECTION -----------------------------#


    def do_motion_correction(self, force: bool = False):
        """
        Do motion correction, also re-saving the results in C order and with planes concatenated along the X axis.
        If results are already available, skip unless force is true.
        Saves the SessionAnalysis object if motion correction is not skipped.
        """
        if self.plane_tifs is None:
            self.convert_to_tif(channel=0)
            assert self.plane_tifs is not None

        if (not force and self.mc_result is not None and
            all(os.path.exists(f) for f in self.mc_result.mmap_files)):
            logging.info('Using existing motion correction results')
            if not os.path.exists(self.mc_result.mmap_file_transposed):
                self.mc_result.mmap_file_transposed = mcorr.transpose_flatten_mc_mmap(
                    self.mc_result.mmap_files, self.mc_result.border_to_0, sample_rate=self.sample_rate,
                    highpass_cutoff=self.highpass_cutoff)
                self.save(save_cnmf=False)
        else:
            try:
                if not self.cnmf_params.motion['pw_rigid']:
                    # take left border into account (i.e. exclude it) if applicable
                    ndead_max = 0 if self.odd_row_ndeads is None else max(self.odd_row_ndeads)
                    shift_max = 0 if self.odd_row_offset is None else math.ceil(abs(self.odd_row_offset) / 2)
                    n_to_clip = ndead_max + shift_max
                else:
                    n_to_clip = 0  # can't apply piecewise shifts to different-size movie (doesn't make sense)

                inds = list(self.cnmf_params.motion['indices'])
                if n_to_clip > 0:
                    inds[1] = slice(n_to_clip, None, None)
                else:
                    inds[1] = slice(None)
                inds = tuple(inds)
                self.cnmf_params.change_params({'motion': {'indices': inds}})

                # initialize variables to update for each plane
                mmap_files: list[str] = []
                shifts: list[np.ndarray] = []
                shifts_els: Optional[list[np.ndarray]] = []
                max_b20 = 0
        
                any_new = False
                for k_plane, plane_tif in enumerate(self.plane_tifs):
                    logging.info(f'Correcting plane {k_plane}')
                    mmap_file, this_shifts, b20, this_shifts_els, new_result = mcorr.motion_correct_file(
                        plane_tif, self.cnmf_params, cluster_args=self.cluster_args, force=force)
                    mmap_files.append(mmap_file)
                    shifts.append(this_shifts)
                    max_b20 = max(max_b20, b20)
                    if this_shifts_els is not None:
                        shifts_els.append(this_shifts_els)
                    any_new = any_new or new_result

                if len(shifts_els) == 0:
                    shifts_els = None

                mmap_file_transposed = mcorr.transpose_flatten_mc_mmap(
                    mmap_files, max_b20, sample_rate=self.sample_rate, highpass_cutoff=self.highpass_cutoff, force=any_new)

                self.mc_result = MCResult(
                    mmap_files=mmap_files, mmap_file_transposed=mmap_file_transposed,
                    border_to_0=max_b20, shifts_rig=shifts, shifts_els=shifts_els,
                    dims=self.plane_size, motion_params=self.cnmf_params.motion)
            except:
                # remove final result file to ensure we don't erroneously think we're done
                if self.mc_result is not None and os.path.exists(self.mc_result.mmap_file_transposed):
                    os.remove(self.mc_result.mmap_file_transposed)
                raise
            self.save(save_cnmf=False)

    
    def apply_motion_correction(self, mc_result: MCResult, do_transpose=False, force=False):
        """
        Apply motion correction result (typically from another channel of the same movie) to the current tiffs.
        By default concatenates the results, but does not transpose to C order, since the output
        will typically not be used for CNMF.
        """
        # sanity checks
        if self.plane_tifs is None:
            raise RuntimeError('Must convert to TIF first')

        if len(mc_result.mmap_files) != len(self.plane_tifs):
            raise RuntimeError('Number of planes does not match given motion correction results')
        
        if mc_result.dims is None or mc_result.motion_params is None:
            raise RuntimeError('Must set dims and motion_params on MCResult before proceeding')
        
        if mc_result.is_piecewise and tuple(mc_result.dims) != tuple(self.plane_size):
            raise RuntimeError('Cannot apply piecewise results to movie of different size')
        
        # check if already done
        if not force and self.mc_result is not None and self.mc_result.has_same_shifts_as(mc_result):
            logging.info('Our current mcorr shifts match the passed ones - not re-applying.')
            return

        # should be safe to do a shallow copy, not really any situation where any of the fields would be mutated
        this_result = copy(mc_result)

        # make MotionCorrect objects and apply the passed shifts
        mcorr_objs = mc_result.recreate_mcorr_objects()
        this_result.mmap_files = [mcorr.apply_mcorr_to_file(mcorr_obj, tif_file)
                                  for mcorr_obj, tif_file in zip(mcorr_objs, self.plane_tifs)]
        this_result.mmap_file_transposed = mcorr.transpose_flatten_mc_mmap(
            this_result.mmap_files, this_result.border_to_0, sample_rate=self.sample_rate,
            highpass_cutoff=self.highpass_cutoff, do_transpose=do_transpose, force=True
        )
        self.mc_result = this_result
        self.save(save_cnmf=False)
                

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
        mov.save(os.path.join(self.data_dir, 'mcorr', f'{self.mouse_id}_{self.sess_id:03d}_comparison.avi'), compress=compress)   


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

                for k_plane, (ax, plane_proj) in enumerate(zip(axs, plane_projs)):
                        ax.imshow(plane_proj, cmap='viridis',
                                    vmin=np.percentile(np.ravel(plane_proj), 50), 
                                    vmax=np.percentile(np.ravel(plane_proj), 99.5))
                        ax.set_title(f'Plane {k_plane} ({corrected_label})')
            figs.append(fig)
        return figs[0], figs[1]


    def save_mc_comparison_summary_plots(self):
        """Compare max projection and correlation images - original (top) vs. motion-corrected (bottom) and save as png"""
        with plt.ioff():
            fig_mean, fig_corr = self.make_mc_comparison_summary_plots()
            fig_mean.savefig(os.path.join(self.data_dir, 'mcorr', f'{self.mouse_id}_{self.sess_id:03d}_meanproj.png'), dpi=200)
            fig_corr.savefig(os.path.join(self.data_dir, 'mcorr', f'{self.mouse_id}_{self.sess_id:03d}_corr.png'), dpi=200)


    #------------------------------------ SUMMARY IMAGES ----------------------------------------#


    def make_projection(self, proj_type: str, blur_kernel_size=1, motion_corrected=True) -> np.ndarray:
        """
        Make correlation image or {mean/std/max}-projection
        If blur_kernel_size > 1, do gaussian blur on a downsampled copy of the movie before computing projection
        blur_downsample_factor: factor to downsample full movie (in time) when doing blur to avoid OOM (default = 10)
        Because it uses the transposed file, setting motion_corrected=True also includes the high-pass filter if any.
        """
        if motion_corrected:
            if self.mc_result is None:
                raise RuntimeError('Motion correction not run')

            if blur_kernel_size > 1 and proj_type != 'mean':  # for nonlinear projections, must blur before projecting
                mov_or_filename = mcorr.transpose_flatten_mc_mmap(
                    self.mc_result.mmap_files, self.mc_result.border_to_0, sample_rate=self.sample_rate,
                    blur_kernel_size=blur_kernel_size, highpass_cutoff=self.highpass_cutoff)
            else:
                mov_or_filename = self.mc_result.mmap_file_transposed

            if proj_type == 'corr':
                proj = make_correlation_parallel(mov_or_filename, cluster.dview)
            else:
                ignore_nan = self.cnmf_params.motion['border_nan'] == True
                proj = make_projection_parallel(mov_or_filename, proj_type, cluster.dview, ignore_nan=ignore_nan)
        else:
            if self.plane_tifs is None:
                raise RuntimeError('Conversion to TIF not run')
            
            if blur_kernel_size > 1 and proj_type != 'mean':
                raise NotImplementedError('Blurring not supported for raw movie (except for mean projection)')
            
            # operate on each plane individually, then combine
            plane_projs = []
            for plane_tif in self.plane_tifs:
                if proj_type == 'corr':
                    plane_proj = make_correlation_parallel(plane_tif, cluster.dview)
                else:
                    plane_proj = make_projection_parallel(plane_tif, proj_type, cluster.dview)
                plane_projs.append(plane_proj)
            proj = np.concatenate(plane_projs, axis=1)
        
        if blur_kernel_size > 1 and proj_type == 'mean':
            # linear operation, can just blur here
            proj = cv2.GaussianBlur(
                proj, (blur_kernel_size, blur_kernel_size), blur_kernel_size//4, None, blur_kernel_size//4, cv2.BORDER_REPLICATE)

        proj[np.isnan(proj)] = 0
        return proj


    def load_projection_from_result(self, proj_type: str) -> tuple[np.ndarray, pd.Series]:
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
        # check whether motion params match for each candidate item
        motion_params_match = np.empty(len(completed_runs), dtype=bool)
        for i in range(len(completed_runs)):
            # update blank CNMFParams with these motion parameters so that default and newly added params are taken into account
            # can't update with all the params because it will error if the input file(s) aren't loadable
            item_params = CNMFParams()
            item_params.change_params({'motion': completed_runs.iloc[i].params['main']['motion']})
            motion_params_match[i] = item_params.motion == self.cnmf_params.motion

        matching_runs = completed_runs.loc[motion_params_match, :]
        if len(matching_runs) == 0:
            raise NoMatchingResultError('No matching mesmerize items to pull projection from')

        completed_row = matching_runs.iloc[-1]
        if proj_type == 'corr':
            return completed_row.caiman.get_corr_image(), completed_row
        else:
            uuid = completed_row['uuid']
            batch_path = completed_row.paths.get_batch_path()
            proj_path = batch_path.parent / uuid / f'{uuid}_{proj_type}_projection.npy'
            return np.load(proj_path), completed_row


    def get_projection_and_border(self, proj_type: str, blur_kernel_size=1, need_border=True, motion_corrected=True) -> tuple[np.ndarray, int]:
        if blur_kernel_size > 1:
            logging.info('Cannot pull from mesmerize (blur_kernel_size > 1)')
        elif not motion_corrected or proj_type not in ['corr', 'mean', 'std', 'max']:
            pass
        else:
            try:
                proj, item = self.load_projection_from_result(proj_type=proj_type)
                # get border size used to run this item
                if need_border:
                    cnmf = load_CNMFExt(item.cnmf.get_output_path(), quiet=True)
                    border = cnmf.params.patch['border_pix']
                else:
                    border = 0  # skip getting border to save time
                return proj, border
            except NoMatchingResultError:
                logging.info('No matching mesmerize items - computing projection anew')

        proj = self.make_projection(proj_type, blur_kernel_size=blur_kernel_size, motion_corrected=motion_corrected)
        if motion_corrected:
            assert self.mc_result is not None  # would throw error in make_projection
            border = self.mc_result.border_to_0
        else:
            border = 0
        return proj, border


    def get_projection(self, proj_type: str, blur_kernel_size=1, motion_corrected=True) -> np.ndarray:
        """Try to pull projection from previous mesmerize result, falling back to make_projection if not available"""
        return self.get_projection_and_border(proj_type=proj_type, blur_kernel_size=blur_kernel_size,
                                              need_border=False, motion_corrected=motion_corrected)[0]        

        
    def get_projection_for_seed(self, type='mean', motion_corrected=True, blur_size=1, norm_medw: Optional[int] = None,
                                border: Union[None, int, BorderSpec] = None, **seed_params_extra) -> tuple[np.ndarray, dict]:
        """
        Make 2D projection image to use for making seed with given params. See make_spatial_seed for parameter meanings.
        Returns the projection along with any unused params
        """
        assert self.mc_result is not None, 'Motion correction not done'

        if border is None:
            proj, border = self.get_projection_and_border(proj_type=type, blur_kernel_size=blur_size, motion_corrected=motion_corrected)
        else:
            proj = self.get_projection(proj_type=type, blur_kernel_size=blur_size, motion_corrected=motion_corrected)
            
        if not isinstance(border, BorderSpec):
            border = BorderSpec.equal(border)

        if proj.ndim == 2:
            concat_planes = len(self.mc_result.mmap_files)
        else:
            concat_planes = 1

        if norm_medw is not None:
            proj = preprocess_proj_for_seed(proj, med_w=norm_medw, border=border, concat_planes=concat_planes)

        seed_params_extra['concat_planes'] = concat_planes
        seed_params_extra['border'] = border
        return proj, seed_params_extra
    

    def get_plane_projections(self, projection_params: Union[str, dict], motion_corrected=True, exclude_border=True) -> list[np.ndarray]:
        if isinstance(projection_params, str):
            proj, border = self.get_projection_and_border(projection_params, motion_corrected=motion_corrected, need_border=exclude_border)
        else:
            proj, seed_params_extra = self.get_projection_for_seed(motion_corrected=motion_corrected, **projection_params)
            border = seed_params_extra['border']

        plane_projs = np.split(proj, self.metadata['num_planes'], axis=1)
        if exclude_border:
            plane_projs = [plane[BorderSpec.equal(border).slices(plane.shape)] for plane in plane_projs]
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

            item = self.get_gridsearch_results().iloc[ind]
            output_path: Path = item.cnmf.get_output_path()
            proj_path = output_path.parent / 'projection_for_seed.npy'
            if proj_path.exists():
                full_proj = np.load(proj_path)
                plane_projs = np.split(full_proj, self.metadata['num_planes'], axis=1)
                if exclude_border:
                    cnmf = load_CNMFExt(item.cnmf.get_output_path(), quiet=True)
                    border = cnmf.params.patch['border_pix']
                    plane_projs = [plane[BorderSpec.equal(border).slices(plane.shape)] for plane in plane_projs]
            else:
                # fall back to recomputing based on seed_params
                logging.warning(f'projection_for_seed not saved - recomputing with params {self.seed_params}')
                plane_projs = self.get_plane_projections(self.seed_params, exclude_border=exclude_border)
        else:
            plane_projs = self.get_plane_projections(projection_params, exclude_border=exclude_border)

        return np.max(plane_projs, axis=0)


    #--------------------------------  CNMF  ----------------------------------------#


    def check_patch_params(self, ax=None):
        """Use view_quilt to check the current CNMF patch parameters"""
        patch_width = self.cnmf_params.patch['rf'] * 2 + 1
        patch_overlap = self.cnmf_params.patch['stride'] + 1
        patch_stride = patch_width - patch_overlap
        print(f'Patch width: {patch_width}, stride: {patch_stride}, overlap: {patch_overlap}')

        # make correlation image of selected plane
        corr_image = self.get_projection('corr')

        patch_ax = view_quilt(corr_image, patch_stride, patch_overlap,
                              vmin=float(np.percentile(np.ravel(corr_image), 50)),
                              vmax=float(np.percentile(np.ravel(corr_image), 99.5)),
                              ax=ax, figsize=(4, 4))

        patch_ax.set_title(f'width={patch_width}\noverlap={patch_overlap}')


    def get_cnmf_run_params(self, is3D=False, seeded=False) -> CNMFParams:
        """Make a copy of the current CNMF params and make changes that need to be made between motion correction and CNMF"""
        if self.mc_result is None:
            raise RuntimeError('No MC results; cannot run CNMF')
        mc_res_fn = self.mc_result.mmap_file_transposed
        file_size = get_file_size(mc_res_fn)[0]
        
        if is3D:
            # rename mmap to undo reshape
            h, w = file_size
            file_size = (h, w // self.metadata['num_planes'], self.metadata['num_planes'])
            mc_res_fn_reshaped = re.sub(f'd2_{w}_d3_1', f'd2_{file_size[1]}_d3_{file_size[2]}', mc_res_fn)
            if not os.path.exists(mc_res_fn_reshaped):
                os.link(mc_res_fn, mc_res_fn_reshaped)
            mc_res_fn = mc_res_fn_reshaped

        default_params = CNMFParams()

        new_params = {
            'data': {'fnames': [mc_res_fn], 'dims': file_size},
            # this is overridden when creating the CNMF object anyway
            'patch': {
                'n_processes': default_params.patch['n_processes'],
                'border_pix': self.mc_result.border_to_0
                },
            'online': {
                # paths that may be different on different machines
                'movie_name_online': default_params.online['movie_name_online'],
                'path_to_model': default_params.online['path_to_model'],
                'init_batch': default_params.online['init_batch']
                }
            }

        # correct patch parameters based on is3D
        curr_patch = self.cnmf_params.patch
        default_patch_z = {'rf': self.metadata['num_planes'], 'stride': 0}
        for patch_key, default_z in default_patch_z.items():
            curr_param = curr_patch[patch_key]
            if is3D:
                if np.isscalar(curr_param):
                    new_params['patch'][patch_key] = [curr_param, curr_param, default_z]
                elif len(curr_param) == 2:
                    new_params['patch'][patch_key] = [*curr_param, default_z]
            elif isinstance(curr_param, (Sequence, np.ndarray)) and len(curr_param) == 3:
                new_params['patch'][patch_key] = curr_param[:2]
        
        # correct gSig based on is3D
        curr_gSig = self.cnmf_params.init['gSig']
        if is3D:
            if np.isscalar(curr_gSig) or len(curr_gSig) < 3:
                raise ValueError('Must pass gSig as length-3 list for 3D CNMF')
        elif isinstance(curr_gSig, (Sequence, np.ndarray)) and len(curr_gSig) == 3:
            new_params['init'] = {'gSig': curr_gSig[:2]}
        
        if is3D:
            # CNN does not support 3D CNMF
            logging.info('Disabling CNN for 3D CNMF')
            new_params['quality'] = {'use_cnn': False}

            # Set custom structuring element for binary closing step to avoid eroding top and bottom planes
            if self.cnmf_params.spatial['se'] is None:
                new_params['spatial']['se'] = np.ones((3, 3, 1), dtype=np.uint8)

        if seeded:
            # settings for seeded run - cannot use patches
            new_params['patch']['rf'] = None
            new_params['patch']['only_init'] = False

        run_params = deepcopy(self.cnmf_params)
        run_params.change_params(new_params)
        return run_params
    

    def make_spatial_seed(self, seed_params: dict, proj: Optional[np.ndarray] = None) -> sparse.csc_array:
        """
        Make a spatial seed (initialization for spatial components) based on a correlation image or projection
        seed_params:
            - 'type': one of ['mean', 'max', 'std', 'corr']  (default: 'mean')
            - 'blur_size': size of Gaussian kernel to use to blur movie in 2D before taking projection
                           (SD = blur_size // 2)
            - 'norm_medw': width of median filter for brightness normalization (default: skip this step)
            - 'border': border pixels to exclude from each plane, either an int to use the same on all sides or a BorderSpec
            - 'gSig': sigma for neuron spatial template
            - other params: see caiman.base.rois.extract_binary_masks_from_structural_channel
        proj: (optional) a pre-computed projection; in this case seed_params should be the extra ones returned from get_projection_for_seed.
        """
        if proj is None:
            proj, seed_params_extra = self.get_projection_for_seed(**seed_params)
        else:
            seed_params_extra = seed_params

        concat_planes = seed_params_extra.pop('concat_planes')
        border: BorderSpec = seed_params_extra.pop('border')
        
        # get masks from each plane separately
        if concat_planes > 1:
            planes = np.split(proj, concat_planes, axis=1)
        else:
            planes = [proj]

        Ain_planes: list[sparse.csc_array] = []
        for plane in planes:
            center_slices = border.slices(plane.shape)
            plane_center = plane[center_slices]
            Ain_plane, _ = cmcustom.my_extract_binary_masks_from_structural_channel(plane_center, **seed_params_extra)
            # fix rows to take border into account
            ind_array = np.arange(plane.size, dtype=int).reshape(plane.shape, order='F')
            inds_used = ind_array[center_slices].ravel(order='F')
            Ain_plane.resize((plane.size, Ain_plane.shape[1]))  # expand shape to take border into account
            Ain_plane.indices = inds_used[Ain_plane.indices]  # offset indices to take border into account
            Ain_planes.append(Ain_plane)

        return sparse.block_diag(Ain_planes, format='csc')  # type: ignore


    def do_cnmf(self, do_refit=True, force=False, cluster_args: Optional[dict] = None,
                is3D=False, seed_params: Optional[dict] = None, snr_type: Optional[Literal['normal', 'gamma']] = None) -> None:
        """
        Do or load CNMF.
        This function now does the same thing as do_cnmf_with_mescore - loading a
        CNMF run that was not run with mesmerize-core is no longer supported
        Generally it is better to use update_params to set cluster_args, seed_params, and snr_type instead of passing them here
        """
        if snr_type is None:
            snr_type = self.snr_type

        uuid, _ = self.start_cnmf_with_mescore(is3D=is3D, seed_params=seed_params, force=force,
                                               backend='local', wait=True, do_refit=do_refit, cluster_args=cluster_args)
        self.select_gridsearch_run(uuid, quiet=True)
        assert self.cnmf_fit is not None
        logging.info(f'Selected CNMF run with UUID {uuid}')

        redo_eval = False
        if self.crossplane_merge_thr is not None and not is3D and self.metadata['num_planes'] > 1:
            logging.info('Doing crossplane merging')
            n_merged = self.cnmf_fit.estimates.merge_components_crossplane(
                n_planes=self.metadata['num_planes'], params=self.cnmf_params, thr=self.crossplane_merge_thr)
            logging.info(f'Merged {n_merged} sets of components')
            if n_merged > 0:
                logging.info('Redoing evaluation after merging')
                redo_eval = True
        else:
            n_merged = 0
        
        if snr_type != 'normal':
            # redo evaluation after switching SNR type
            if n_merged == 0:
                logging.info(f'Redoing evaluation with {snr_type} SNR type')
            redo_eval = True
            
        if redo_eval:
            self.do_cnmf_evaluation(snr_type=snr_type, cluster_args=cluster_args)
        
        self.make_df_over_f()

        if self.downsample_factor is not None:
            assert self.frames_per_trial is not None, 'frames per trial should be set during conversion'
            assert self.cnmf_fit is not None, 'CNMF not successful?'
            logging.info(f'Upsampling (interpolating) results by a factor of {self.downsample_factor}')
            self.cnmf_fit.estimates.interpolate_t(self.downsample_factor, self.frames_per_trial)
        
        self.save(save_cnmf=True)


    def do_cnmf_with_mescore(self, force=False, is3D=False, seed_params: Optional[dict] = None,
                             do_refit=True, cluster_args: Optional[dict] = None) -> None:
        """
        Run a CNMF run through mesmerize-core, so results will be in the batch dataframe.
        Waits for process to finish and then selects the run to make it active for further processing.
        force: whether to just load the last run in the batch file if there is one.
        is3D: whether to do 3D CNMF.
        """
        logging.warning('do_cnmf_with_mescore is deprecated, you can just use do_cnmf with the same keyword arguments')
        self.do_cnmf(do_refit=do_refit, force=force, cluster_args=cluster_args,
                     is3D=is3D, seed_params=seed_params)


    def start_cnmf_with_mescore(self, is3D=False, seed_params: Optional[dict] = None, force=True,
                                backend='local_async', wait=False, do_refit=True, cluster_args: Optional[dict] = None,
                                **run_args) -> tuple[str, Waitable]:
        """
        Start a CNMF run through mesmerize-core, so results will be in the batch dataframe. Defaults to running in background.
        is3D: whether to do 3D CNMF.
        seed_params: if None, usees default seeed params (see caiman_params.py). Use empty dict to use non-seeded CNMF.
        do_refit only takes effect if seed_params is empty ({}) (no need for refit with seeded CNMF)
        cluster_args are used when setting up the cluster if the backend is 'local' or 'local_async'
        run_args are set to series.caiman.run in mescore.
        Returns the UUID and process, which may still be running depending on whether 'wait' is set to True in run_args.
        """
        if os.name == "nt":
            cnmf_cache.set_maxsize(0)
        
        if seed_params is None:
            seed_params = copy(self.seed_params)

        # Get params to pass to mescore
        batch = self.get_gridsearch_results(allow_create=True)
        item_name = batch.paths.get_batch_path().stem
        run_params = self.get_cnmf_run_params(is3D=is3D, seeded=bool(seed_params))
        
        # Use CNMF params to make seed if necessary
        if bool(seed_params):
            seed_params['border'] = run_params.patch['border_pix']
            Ain_name = get_spatial_seed_name(seed_params)
            if 'gSig' not in seed_params:  # should not be included in name
                seed_params['gSig'] = run_params.init['gSig'][0]
        else:
            Ain_name = None

        params_cnmf = {
            'main': run_params.to_dict(),
            'refit': do_refit and not seed_params,
            'Ain_path': f'{Ain_name}.npy' if Ain_name is not None else None
        }

        if not force:
            completed_runs = batch.loc[[out is not None and out['success'] for out in batch.outputs], :]
            for _, row in completed_runs.iterrows():
                row = cast(MescoreSeries, row)
                params_diffs: Sequence[dict[str, Any]] = get_params_diffs([params_cnmf, row.at['params']])
                # ignore quality parameters when checking for match; don't need to re-run CNMF for different quality params
                if all(param_name.startswith('quality.') or param_name in EXCLUDE_FROM_DIFFS
                       for param_name in params_diffs[0]):
                    logging.info('Found matching CNMF run')
                    if len(params_diffs[0]) > 0:
                        # load in order to compare quality params
                        cnmf_found = load_CNMFExt(str(row.cnmf.get_output_path()), quiet=True)
                        quality_params_actual = cnmf_found.params.quality
                        quality_diffs = get_params_diffs([params_cnmf['main']['quality'], quality_params_actual])
                        if len(quality_diffs[0]) > 0:
                            logging.warning('Found run has different ROI quality settings:')
                            quality_diffs_df = pd.DataFrame.from_dict(quality_diffs, dtype=object)  # type: ignore
                            quality_diffs_df.rename(index={0: 'Current settings', 1: 'Saved settings'}, inplace=True)
                            print(quality_diffs_df)

                    return str(row.at['uuid']), DummyProcess()
            logging.info('No previous matching CNMF run found - starting process')

        batch.caiman.add_item(
            algo='cnmf',
            item_name=item_name,
            input_movie_path=params_cnmf['main']['data']['fnames'][0],
            params=params_cnmf
        )

        item = cast(MescoreSeries, batch.iloc[-1])
        uuid = str(item.uuid)
        if bool(seed_params):
            # actually make and save seed
            output_dir = batch.paths.get_batch_path().parent.joinpath(uuid).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            proj, seed_params_extra = self.get_projection_for_seed(**seed_params)
            np.save(output_dir / 'projection_for_seed.npy', proj)
            Ain = self.make_spatial_seed(seed_params_extra, proj=proj)
            np.save(output_dir / f'{Ain_name}.npy', np.array(Ain))  # saves as object array

        if 'dview' not in run_args and backend in ['local', 'local_async']:
            if cluster_args is not None:
                cluster.start(**cluster_args)
            run_args['dview'] = cluster.dview

        proc = item.caiman.run(backend=backend, wait=wait, **run_args)
        return uuid, proc


    def do_cnmf_gridsearch(self, gridsearch_params: Union[ParamGrid, Sequence[ParamGrid]], wait=True,
                           backend='local_async') -> list[Waitable]:
        """
        Test CNMF with every combination of given parameters, using mesmerize-core. This function runs on the local host.
        Each key of gridsearch_params should be a tuple (group, name) specifying a specific CNMF parameter.
        Each entry is a list of parameter values to test (can be of length 1 to set value for all runs).
        Runs on a remote cluster if host is given and matches an entry in host_info.py (e.g. login.tj).
        To run locally, use 'localhost'.
        If wait is false, launches grid search process but does not wait for it to finish or validate success.

        The following specal entries of gridsearch_params are allowed (and will be removed before passing to CNMFParams):
        - ('special', '3D'): (bool) - set to True to do 3D CNMF. Note gSig should also be set accordingly.
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
    

    def make_df_over_f(self, force: bool = False, denoised: Optional[bool] = None):
        """
        Calculates delta F over F for all ROIs using our default settings and
        saves to estimates.F_dff and estimates.F_dff_denoised. Set denoised to True or False
        to only calculate one or the other
        """
        if self.cnmf_fit is None:
            raise RuntimeError('CNMF fit not found')
        est = self.cnmf_fit.estimates   
    
        if denoised is None or not denoised:
            if force or est.F_dff is None:
                logging.info('Calculating df/f')
                est.F_dff = calc_df_over_f(est, use_residuals=True)
            else:
                logging.info('Found df/f - not recalculating')
        
        if denoised is None or denoised:
            if force or est.F_dff_denoised is None:
                logging.info('Calculating denoised df/f')
                est.F_dff_denoised = calc_df_over_f(est, use_residuals=False)
            else:
                logging.info('Found denoised df/f - not recalculating')


    #------------------------- ROI EVALUATION -----------------------------#


    def do_cnmf_evaluation(self, new_quality_params=None, snr_type: Optional[Literal['normal', 'gamma']] = None,
                           recalc=False, cluster_args: Optional[dict] = None):
        """
        Recalculates idx_components based on automatic criteria.
        new_quality_params: Unused, throws an error if not None, use self.cnmf_params.change_params({'quality': <new params>}) instead
        snr_type: which type of SNR to use; if left as None, uses self.snr_type. Otherwise updates self.snr_type.
        recalc: If true, recomputes all metrics; otherwise reuses them if they already exist.
        """
        if self.mc_result is None:
            raise RuntimeError('Cannot do CNMF evaluation without input file (motion correction result)')

        if self.cnmf_fit is None:
            raise RuntimeError('CNMF fit not found')
        
        if new_quality_params is not None:
            raise RuntimeError('Use self.cnmf_params.change_params({"quality": <new params>}) instead of setting new_quality_params')
        new_quality_params = self.cnmf_params.quality
        
        if snr_type is None:
            snr_type = self.snr_type
        elif snr_type != self.snr_type:
            self.snr_type = snr_type

        if cluster_args is not None:
            cluster.start(**cluster_args)

        logging.info('Doing CNMF component evaluation')
        evaluate_cnmf(self.cnmf_fit, self.mc_result.mmap_file_transposed, snr_type=snr_type,
                      new_quality_params=new_quality_params, recalc=recalc, dview=cluster.dview)

        # re-save cnmf object with evaluation data added
        logging.info('Re-saving CNMF object with evaluation data')
        self.save(save_cnmf=True)


    def pick_gt_rois(self):
        """
        Interactively select ROIs that should be recognized as neurons from the correlation plot.
        Saves ROIs in sparse CSC format as {mouseid}_{sessid}_{tag}_gtroi.npz.
        """
        if mpl.get_backend().lower() == 'agg':
            raise RuntimeError('Must have interactive backend to pick ROIs')
        
        # get and plot correlations
        corr_concat = self.get_projection('corr')
        fig, ax = plt.subplots()
        ax.imshow(corr_concat, interpolation=None,
                  vmin=float(np.percentile(corr_concat[~np.isnan(corr_concat)], 1)),
                  vmax=float(np.percentile(corr_concat[~np.isnan(corr_concat)], 99)))
        
        # prompt for ROIs
        rois = MultiRoi(fig=fig, ax=ax)
        # figure will close automatically

        # make CSC matrix and save
        n_rois = len(rois.rois)
        masks_lil = sparse.lil_array((n_rois, corr_concat.size), dtype=bool)
        b_keep = np.ones(n_rois, dtype=bool)  # for excluding empty masks
        for i, roi in enumerate(rois.rois.values()):
            mask = roi.get_mask(corr_concat).ravel(order='F')
            if np.any(mask):
                masks_lil[i, :] = mask
            else:
                b_keep[i] = False

        masks_csc = sparse.csr_array(masks_lil[b_keep, :]).T
        tagstr = f'_{self.tag}' if self.tag is not None else ''
        mask_file = os.path.join(self.data_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_gtroi.npz')
        sparse.save_npz(mask_file, masks_csc)


    def get_roi_gt_score(self, gt_file: Optional[Union[str, Path]] = None, binarize=True, nrgthr=0.9, accepted_only=False) -> float:
        """
        Score CNMF ROIs against ground-truth file (see caiman_analysis.get_roi_gt_score)
        """
        # get estimates and ground truth
        cnmf_fit = self.cnmf_fit
        if cnmf_fit is None:
            raise RuntimeError('No CNMF results to use')
        est = cnmf_fit.estimates
        if not isinstance(est.dims, tuple):
            raise RuntimeError('CNMF has not been run')

        if gt_file is None:
            tagstr = f'_{self.tag}' if self.tag is not None else ''
            gt_file = os.path.join(self.data_dir, f'{self.mouse_id}_{self.sess_id:03d}{tagstr}_gtroi.npz')

        # make dims that correspond to actual spatial layout
        dims = self.get_dims_in_um()
        return get_roi_gt_score(cnmf_fit, gt_file, dims, binarize=binarize, nrgthr=nrgthr,
                                accepted_only=accepted_only, dview=cluster.dview)

    
    #--------------------- ACCESSING CNMF RESULTS -----------------------------#
            
    
    def get_gridsearch_results(self, allow_create=False) -> MescoreBatch:
        if self.gridsearch_batch_path is None:
            # See if a batch file already exists
            df = get_batch_for_session(
                self.mouse_id, self.sess_id, tag=self.tag, data_dir=self.data_dir, create=allow_create)
            self.gridsearch_batch_path = str(df.paths.get_batch_path())
        else:
            df = cast(MescoreBatch, mc.load_batch(self.gridsearch_batch_path))

        # fix input_movie_path for new directory structure - make relative to parent raw data path
        any_changed = False
        for _, row in df.iterrows():
            abs_path_from_data_dir = Path(self.data_dir) / str(row['input_movie_path'])
            if abs_path_from_data_dir.exists():
                new_rel_path = abs_path_from_data_dir.relative_to(mc.get_parent_raw_data_path())
                row['input_movie_path'] = str(PurePosixPath(new_rel_path))
                any_changed = True
        if any_changed:
            df.caiman.save_to_disk()            
        return df


    def get_gridsearch_diffs(self, batch: Optional[MescoreBatch] = None, add_gt_scores=False, gt_file=None) -> pd.DataFrame:
        """
        Get table of parameter differences between all the gridsearch runs that have been run so far
        If batch is given, use this batch dataframe instead of calling get_gridsearch_results()
        If add_gt_scores is true, also adds ground-truth Jaccard scores to the table (computed by get_roi_gt_score).
        """
        if batch is None:
            df = self.get_gridsearch_results()
        else:
            df = batch.reset_index(drop=True)
        
        df_diffs = df.caiman.get_params_diffs('cnmf', df.item_name.iat[0])

        # filter out unimportant parameters (in case they aren't fixed by get_cnmf_run_params)
        df_diffs = df_diffs.loc[:, ~np.isin(df_diffs.columns, EXCLUDE_FROM_DIFFS)]

        # add UUIDs
        df_diffs = pd.concat([df_diffs, df.loc[:, 'uuid']], axis=1)

        if add_gt_scores:
            gt_scores = pd.Series(np.zeros(len(df)), index=df.index, name='gt_score')
            for i, uuid in enumerate(df.uuid.array):
                self.select_gridsearch_run(uuid=uuid, quiet=True)
                gt_scores.at[i] = self.get_roi_gt_score(gt_file)
            df_diffs = pd.concat([gt_scores, df_diffs], axis=1)
            df_diffs.sort_values(by='gt_score', axis=0, inplace=True, ascending=False)

        return df_diffs
            

    def select_gridsearch_run(self, uuid=None, *_, index=None, quiet=False, force_reload=True) -> str:
        """
        Select a CNMF gridsearch run by either UUID or index in the dataframe, and make it the current run
        Sets self.cnmf_fit, self.cnmf_fit_filename, and self.cnmf_params to match the selected run
        Returns the UUID of the selected run.
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
        assert self.gridsearch_batch_path is not None, 'Gridsearch batch path should be set after get_gridsearch_results'

        if index is not None:
            try:
                row = df.iloc[index]
                uuid = row.uuid
            except KeyError:
                raise ValueError(f'Index {index} not present in gridsearch dataframe')
        else:
            assert uuid is not None
            rows = df.loc[df.uuid == uuid]
            if len(rows) != 1:
                raise ValueError(f'UUID {uuid} not present in gridsearch dataframe')
            row = rows.iloc[0]

        if row.outputs is None or not row.outputs['success']:
            raise RuntimeError('Run has not completed or did not succeed')
        
        if row.algo != 'cnmf':
            raise RuntimeError('Selected item is not a CNMF run')
        
        # OK, we have a CNMF run and it has completed successfully. Set fields.
        self.cnmf_fit_filename = str(Path(self.gridsearch_batch_path).parent / row.outputs['cnmf-hdf5-path'])
        if cluster.dview is None:
            setup_cluster()
        cnmf_fit = load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=quiet)
        self.cnmf_fit = cnmf_fit

        params_from_row = row.params['main']
        # update specifically quality parameters from CNMF object because they could be changed post-hoc
        params_from_row['quality'] = cnmf_fit.params.quality
        # also blank out fnames to avoid path normalization issues
        params_from_row['data']['fnames'] = None
        self.cnmf_params = CNMFParams(params_dict=params_from_row)

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
        completed_inds = [i for i, out in enumerate(batch.outputs) if out is not None and out['success']]
        completed_runs = batch.loc[completed_inds, :].reset_index(drop=True)
        run_filenames = [os.path.split(outputs['cnmf-hdf5-path'])[1] for outputs in completed_runs.outputs]
        selected_filename = os.path.split(self.cnmf_fit_filename)[1]
        if selected_filename in run_filenames:
            return completed_inds[run_filenames.index(selected_filename)]
        else:
            return None
        
    def get_selected_uuid(self) -> Optional[str]:
        index = self.get_selected_index()
        if index is not None:
            batch = self.get_gridsearch_results()
            return batch.at[index, 'uuid']
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


    def get_xy_footprints(self, normalize=True, binarize=False, **binarize_kwargs) -> sparse.csc_matrix:
        """
        Get footprints (A) of selected CNMF run compressed to just X/Y coordinates
        added across the z dimension and normalized).
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
        xy_footprints = footprints.collapse_footprints_to_xy(A, n_planes, binarize=binarize, **binarize_kwargs)
        if normalize and not binarize:
            xy_footprints = footprints.normalize_footprints(xy_footprints)
        return xy_footprints
    

    def get_footprints_per_plane(self, normalize=False, binarize=False, **binarize_kwargs) -> list[sparse.csc_matrix]:
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
        cnmf_fit = load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=True)
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
        filename_meta = filename_base + f'_metadata.pkl'
        
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
            inds = range(est.A.shape[1])
        
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
        image_dims = self.cnmf_params.data['dims']

        for i, (comp_s, comp_t) in enumerate(zip(est.b.T, est.f)):
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
            self.cnmf_fit = load_CNMFExt(self.cnmf_fit_filename, dview=cluster.dview, quiet=True)

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

        if roi_ids is None:
            roi_ids = range(est.A.shape[1])

        if any([id not in range(est.A.shape[1]) for id in roi_ids]):
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
        A = sparse.csc_array(est.A) if not isinstance(est.A, MaybeSparse) else est.A
        coms = rois.com(A[:, roi_ids], *dims)

        # get projection
        if isinstance(proj_type, str):
            proj = self.get_projection(proj_type)
        else:
            proj, _ = self.get_projection_for_seed(**proj_type)
        
        fig, ax = plt.subplots()
        thumbnail = np.empty((box_size, box_size), dtype=proj.dtype)

        for roi_id, path, (com_y, com_x) in zip(roi_ids, save_paths, tqdm(coms, desc='Saving thumbnails...', unit='ROI')):
            if not force and os.path.exists(path):
                logging.info(f'Skipping {roi_id} since it is already saved')
                continue

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
            ax.imshow(thumbnail, interpolation=None, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_axis_off()

            # save as PNG
            fig.savefig(path, bbox_inches='tight', pad_inches=0)

        return save_paths


def get_session_analysis_file_pattern(mouse_id: Union[str, int], sess_id: int,
                                      tag: Optional[str], downsample_factor: Optional[int] = None) -> str:
    tagstr = '_' + tag if tag else ''
    if downsample_factor is not None:
        tagstr = tagstr + f'_ds{downsample_factor}'
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
        sessdata._cnmf_fit = load_cnmf(sessdata.cnmf_fit_filename, metadata=sessdata.metadata, quiet=quiet)
        if sessdata._cnmf_fit is None:
            sessdata.cnmf_fit_filename = None
    return sessdata


def load_cnmf(cnmf_filename: str, metadata: dict, quiet=True) -> Optional[CNMFExt]:
    """
    Tries to load CNMF results
    """
    if not quiet:
        logging.info(f'Loading CNMF results from {cnmf_filename}')
    try:
        cnmf_obj = load_CNMFExt(cnmf_filename, dview=cluster.dview, quiet=quiet)
    except FileNotFoundError:
        logging.warning(f'CNMF file could not be found; not loaded')
        return None
    else:
        # set n_processes since refit reads from it and passing this to load_CNMF doesn't actually do anything
        # also blank out fnames to avoid check; will set it before actually running any analysis
        cnmf_obj.params.change_params({'data': {'fnames': None}, 'patch': {'n_processes': cluster.ncores}})
        return cnmf_obj


def load_latest(mouse_id: Union[int, str], sess_id: int, rec_type: str = 'learning_ppc',
                tag: Optional[str] = None, downsample_factor: Optional[int] = None,
                quiet=True, lazy=True) -> SessionAnalysis:
    """Load latest saved analysis for given mouse/session/tag"""
    data_dir = paths.get_processed_dir(mouse_id, rec_type=rec_type)
    file_pattern = get_session_analysis_file_pattern(mouse_id, sess_id, tag=tag, downsample_factor=downsample_factor)
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


def evaluate_cnmf(cnmf_fit: CNMFExt, mc_res_path_or_images: Union[str, np.ndarray], snr_type: Literal['normal', 'gamma'] = 'gamma',
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


def get_roi_gt_score(cnmf_fit: cnmf.CNMF, gt_file: Union[str, Path], dims: Optional[Sequence[Union[int, slice, np.ndarray]]] = None,
                     binarize=True, nrgthr=0.9, accepted_only=False, dview=None) -> float:
    """
    Score CNMF ROIs against ground-truth file by computing the average Jaccard index
    between each ground-truth ROI and the identified ROI with the nearest center of mass.
    By default, ground-truth ROIs are loaded from the same path as pick_gt_rois saves to.
    If dims is specified, each entry (y, x[, z]) specifies either the number of pixels in this dimension or
        the spatial location of each pixel in this dimension, for accurate distance calculation.
    If binarize is True, binarize the identified ROIs before computing Jaccard index.
    If nrgthr is <1, use only up this fraction of cumulative energy (from highest- to lowest-value pixels)
    If accepted_only is true, uses the subset of identified ROIs that were accepted by the quality criteria.
    """
    # get estimates and ground truth
    est = cnmf_fit.estimates
    assert est.A is not None and isinstance(est.dims, tuple), 'Estimates elements should not be None - has CNMF been run?'
    A = sparse.csc_array(est.A)

    if accepted_only:
        if est.idx_components is None:
            raise RuntimeError('Cannot score accepted components - no CNMF evaluation has been run')
        A = A[:, est.idx_components]

    gt_masks: sparse.csc_array = sparse.load_npz(gt_file).astype(est.A.dtype)
    n_gt = gt_masks.shape[1]

    if dims is None:
        dims = est.dims

    if len(dims) != len(est.dims):
        raise ValueError('Incorrect number of dimensions given')

    gt_planes = np.zeros(n_gt)
    xy_size = est.dims[0] * est.dims[1]
    if len(est.dims) == 3:
        # ensure each ground-truth mask is only on one plane
        gt_masks = gt_masks.tolil()  # allows changing sparsity structure more efficiently
        for col in range(n_gt):
            # get plane that most (usually ALL) pixels are in for this ground-truth ROI
            kplane = mode(gt_masks[:, [col]].nonzero()[0] // xy_size)[0]
            gt_masks[:kplane*xy_size, col] = 0
            gt_masks[(kplane+1)*xy_size:, col] = 0
            gt_planes[col] = kplane

    # map gt ROIs to nearest identified ROI
    coms_gt = cmcustom.my_com(gt_masks, *dims)
    coms_id = cmcustom.my_com(A, *dims)

    # builds tree for nearest-neighbor lookups, then queries with the gt ROI COMs
    nn_tree_id = KDTree(coms_id)
    _, nn_inds = nn_tree_id.query(coms_gt)
    A_closest = A[:, nn_inds]

    # process each ROI using threshold_components function
    if nrgthr < 1:
        thresh_args = {
            'medw': (1,) * len(est.dims),  # disable median filtering
            'thr_method': 'nrg',
            'nrgthr': nrgthr,
            # morphological closing makes sense after threhsolding, but don't do it in Z
            'se': np.ones((3, 3, 1)[:len(est.dims)], dtype=int),
            'dview': dview
        }
        A_closest = sparse.csc_array(threshold_components(A_closest, est.dims, **thresh_args))
    
    if binarize:
        A_closest = A_closest.astype(bool)

    # just take pixels on plane of each ROI
    if len(est.dims) == 3:
        A_closest = A_closest.tolil()  # allows changing sparsity structure more efficiently
        for col, kplane in enumerate(gt_planes):
            A_closest[:kplane*xy_size, col] = 0
            A_closest[(kplane+1)*xy_size:, col] = 0

    if not binarize:
        # normalize so that pixels have average value of 1 (not necessary if binarized)
        A_closest = A_closest.multiply(A_closest.getnnz(axis=0) / A_closest.sum(axis=0))

    # compute Jaccard index
    intersection = np.array(A_closest.minimum(gt_masks).sum(axis=0))
    union = np.array(A_closest.maximum(gt_masks).sum(axis=0))
    iou = intersection / union
    return float(np.mean(iou))


def identify_marked_rois(cnmf_obj: CNMFExt, cnmf_filename: Optional[str], A_structural: sparse.csc_array,
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
        A_functional = sparse.csc_matrix(est.A)[:, used_comps]
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
        load.cache_clear()
    else:
        logging.warning('CNMF not saved - no filename')


def calc_df_over_f(est: Estimates, use_residuals=True, roi_subset: Optional[Sequence[int]] = None,
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
        # index to just the component(s) requested
        if isinstance(est.A, (sparse.coo_matrix, sparse.coo_array)):
            A = sparse.csc_matrix(est.A)[:, roi_subset]
        else:
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