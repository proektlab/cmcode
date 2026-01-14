"""
Motion correction utilities
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Optional, Union, Generator, ParamSpec, TypedDict, Any, cast

import caiman as cm
from caiman.base.movies import get_file_size
from caiman.motion_correction import MotionCorrect, get_patch_centers
from caiman.paths import decode_mmap_filename_dict, memmap_frames_filename
import cv2
import holoviews as hv
from mesmerize_core.algorithms._utils import Cluster, save_c_order_mmap_parallel
from mesmerize_core.utils import Border
import numpy as np

from cmcode import caiman_analysis as cma, caiman_params as cmp
from cmcode.util import paths
from cmcode.util.image import BorderSpec
from cmcode.util.types import NoMatchingResultError


@dataclass
class PiecewiseMCInfo:
    """Deprecated, used only to allow unpickling previous results"""
    shifts_els: np.ndarray
    patch_xy_inds: Optional[list[tuple[int, int]]] = None

    def __post_init__(self):
        if self.patch_xy_inds is not None and any(inds is None for inds in self.patch_xy_inds):
            # occurs if shifts_opencv is True
            self.patch_xy_inds = None

@dataclass
class MCResult(paths.CustomPathMappable):
    mmap_files: list[str]
    border_to_0: int
    border_asym: list[BorderSpec]  # border on each side (old results just repeat border_to_0)
    shifts_rig: list[np.ndarray]
    shifts_els: Optional[list[np.ndarray]] = None
    dims: Optional[tuple[int, int]] = None
    motion_params: Optional[dict] = None
    mmap_file_transposed: Optional[str] = None  # deprecated, keep for unpickling

    # cached datasets
    _shifts_rig_hv: Optional[hv.Dataset] = field(init=False, default=None)
    _shifts_els_hv: Optional[hv.Dataset] = field(init=False, default=None)

    def __getstate__(self):
        """Avoid saving cache fields when pickling"""
        state = self.__dict__.copy()
        del state['_shifts_rig_hv']
        del state['_shifts_els_hv']
        return state
    
    def __setstate__(self, state):
        """Deal with old versions of MCResult when unpickling"""
        if 'shifts' in state:
            logging.debug('Converting old version of MCResult object')
            if len(state['shifts']) == 0 or state['shifts'][0].ndim == 2:
                # rigid
                state['shifts_rig'] = state['shifts']
                state['shifts_els'] = None
            else:
                logging.warning('Rigid shifts not saved from piecewise - using mean instead')
                state['shifts_rig'] = [np.mean(shifts, axis=2) for shifts in state['shifts']]
                state['shifts_els'] = state['shifts']
            del state['shifts']
        if 'piecewise_info' in state:
            logging.debug('Converting old version of MCResult object')
            if state['piecewise_info'] is None:
                state['shifts_els'] = None
            else:
                state['shifts_els'] = [pw_info.shifts_els for pw_info in state['piecewise_info']]
            del state['piecewise_info']

        for hv_field in ['_shifts_rig_hv', '_shifts_els_hv']:
            if hv_field not in state: 
                state[hv_field] = None

        if 'border_asym' not in state:
            state['border_asym'] = [BorderSpec.equal(state['border_to_0'])] * len(state['mmap_files'])

        self.__dict__.update(state)

    def __setattr__(self, name, val):
        """invalidate caches when necessary"""
        if name == 'shifts_rig':
            self._shifts_rig_hv = None
        elif name == 'shifts_els':
            self._shifts_els_hv = None
        super().__setattr__(name, val)
    
    def __getattribute__(self, name):
        """Make sure we don't read mmap_file_transposed"""
        if name == 'mmap_file_transposed':
            raise AttributeError('MCResult.mmap_file_transposed is deprecated, read from SessionAnalysis')
        return object.__getattribute__(self, name)

    @property
    def shifts_rig_hv(self) -> hv.Dataset:
        """Make HoloViews dataset from rigid shifts"""
        if self._shifts_rig_hv is None:
            shifts_all = np.stack(self.shifts_rig)
            nplanes, ndims, nframes = shifts_all.shape
            assert ndims == 2, 'Only 2D shifts supported'
            data_dims = {
                'plane': range(nplanes),
                'dim': ['y', 'x'],
                'frame': range(nframes),
                'shift': shifts_all,
            }
            self._shifts_rig_hv = hv.Dataset(data_dims, ['frame', 'dim', 'plane'], 'shift')
        return self._shifts_rig_hv

    @property
    def shifts_els_hv(self) -> Optional[hv.Dataset]:
        """Make HoloViews dataset from piecewise shifts, if they exist"""
        if self.shifts_els is None:
            return None

        if self._shifts_els_hv is None:
            if self.dims is None or self.motion_params is None:
                raise RuntimeError('Must set dims and motion_params before getting shifts_els as HoloView dataset')

            # find patch locations
            patch_centers_y, patch_centers_x = get_patch_centers(
                self.dims, strides=self.motion_params['strides'], overlaps=self.motion_params['overlaps'],
                upsample_factor_grid=self.motion_params['upsample_factor_grid'], shifts_opencv=self.motion_params['shifts_opencv'])
            
            npatch_y = len(patch_centers_y)
            npatch_x = len(patch_centers_x)

            shifts_all_els = np.stack(self.shifts_els)
            nplanes, ndims, nframes, _ = shifts_all_els.shape
            assert ndims == 2, 'Only 2D shifts supported'

            # unravel shifts into X/Y grid
            shifts_all_els = shifts_all_els.reshape(shifts_all_els.shape[:3] + (npatch_y, npatch_x), order='C')

            data_dims = {
                'plane': range(nplanes),
                'dim': ['y', 'x'],
                'frame': range(nframes),
                'shift': shifts_all_els,
                'ypatch': patch_centers_y,
                'xpatch': patch_centers_x
            }
            self._shifts_els_hv = hv.Dataset(data_dims, ['xpatch', 'ypatch', 'frame', 'dim', 'plane'], 'shift')
        return self._shifts_els_hv

    @property
    def n_planes(self) -> int:
        return len(self.mmap_files)

    @property
    def is_piecewise(self) -> bool:
        return self.shifts_els is not None

    P = ParamSpec('P')
    def apply_path_mapper(self, path_mapper: paths.PathMapper[P], *args: P.args, **kwargs: P.kwargs) -> 'MCResult':
        """Implements CustomPathMappable by normalizing paths to memmap files"""
        res_norm = deepcopy(self)
        for field in ['mmap_files', 'mmap_file_transposed']:
            norm_path = path_mapper(getattr(res_norm, field), *args, **kwargs)
            setattr(res_norm, field, norm_path)  # type: ignore
        return res_norm

    def has_same_shifts_as(self, other: 'MCResult') -> bool:
        """Test whether this result and another have the same shifts. Allows for one to have more frames than the other."""
        if len(self.shifts_rig) != len(other.shifts_rig):
            return False
        
        if self.is_piecewise != other.is_piecewise:
            return False
        
        this_shifts = getattr(self, 'shifts_els' if self.is_piecewise else 'shifts_rig')
        other_shifts = getattr(other, 'shifts_els' if other.is_piecewise else 'shifts_rig')
        n_frames = min(this_shifts[0].shape[1], other_shifts[0].shape[1])
        return all(np.all(this_plane_shifts[:, :n_frames] == other_plane_shifts[:, :n_frames])
                   for this_plane_shifts, other_plane_shifts in zip(this_shifts, other_shifts))

    def recreate_mcorr_objects(self) -> list[MotionCorrect]:
        """Make a motion correction object like the one these results were derived from, good enough to use apply_shifts_movie."""
        if self.motion_params is None:
            raise RuntimeError('Must set motion params to recreate MotionCorrect object')

        mcorr_objs: list[MotionCorrect] = []
        
        for kplane in range(len(self.shifts_rig)):  
            # use dummy input movie
            mcorr_obj = MotionCorrect(np.array([]), **self.motion_params, dview=cma.cluster.dview)
            # assign results from motion correction
            mcorr_obj.shifts_rig = list(self.shifts_rig[kplane].T)
            if self.shifts_els is not None:
                shifts_els = self.shifts_els[kplane]
                mcorr_obj.x_shifts_els = list(shifts_els[0])
                mcorr_obj.y_shifts_els = list(shifts_els[1])
                if self.motion_params['is3D']:
                    mcorr_obj.z_shifts_els = list(shifts_els[2])
            mcorr_objs.append(mcorr_obj)
        return mcorr_objs


def _build_motion_correct_basename(filepath: str, is_piecewise=True, with_dt=False) -> str:
    """
    Determine correct motion correction base name for given input file path (without the dims, T, order parts)
    If with_dt is true, returns a template for a filename with a timestamp (see paths.make_timestamped_filename)
    """
    file_path = Path(filepath)
    base_name = file_path.stem
    if with_dt:
        base_name += '_%dt'
    if is_piecewise:
        base_name += '_els_'
    else:
        base_name += '_rig_'
    return str(file_path.parent.parent / 'mcorr' / base_name)

def _build_motion_correct_path(filepath: str, is_piecewise=True, with_dt=False) -> Path:
    """
    Determine what caiman will save the motion-corrected movie as (to find if previously calculated)
    If with_dt is true, returns a template for a filename with a timestamp (see paths.make_timestamped_filename)
    """
    base_name = _build_motion_correct_basename(filepath, is_piecewise=is_piecewise, with_dt=with_dt)

    dims, T = get_file_size(filepath)
    assert isinstance(T, int), 'T should be int when taking file size of one movie'

    fname_tot = memmap_frames_filename(base_name, dims, T, order='F')
    return Path(fname_tot)


def _make_hardlink_with_dt(file_path: str) -> str:
    """
    Make a hard link to the given file with the current date/time added before the extension.
    This is used to help do potentially multiple versions of motion correction from a single
    TIF file, since MotionCorrect doesn't support saving the mmap file with a different name.
    """
    start, ext = os.path.splitext(file_path)
    path_template = start + '_%dt' + ext
    dir, name_template = os.path.split(path_template)
    link_path = os.path.join(dir, paths.make_timestamped_filename(name_template))
    os.link(src=file_path, dst=link_path)
    return link_path


@contextmanager
def set_output_location(output_path: Union[str, Path]) -> Generator[None, None, None]:
    """
    Context manager that sets the temp directory to the output location
    to save to the given output file. Idempotent.
    """
    mcorr_dir = os.path.split(output_path)[0]
    if not os.path.exists(mcorr_dir):
        os.makedirs(mcorr_dir, exist_ok=True)

    prev_temp_dir = os.environ['CAIMAN_TEMP'] if 'CAIMAN_TEMP' in os.environ else None
    try:
        os.environ['CAIMAN_TEMP'] = mcorr_dir
        yield
    finally:
        if prev_temp_dir is not None:
            os.environ['CAIMAN_TEMP'] = prev_temp_dir
        else:
            del os.environ['CAIMAN_TEMP']


class PlaneMcorrResult(TypedDict):
    mmap_path: str                    # path to corrected movie
    shifts_rig: np.ndarray            # rigid shifts
    shifts_els: Optional[np.ndarray]  # nonrigid shifts, if using
    border_to_0: int                  # max border on any side
    border_asym: BorderSpec           # max border on each side


def compute_border_asym(shifts: np.ndarray) -> BorderSpec:
    """
    Given shifts array with dimension along the first axis, compute asymmetric border
    (max border on each side, rounding up to the nearest integer).
    """
    max_top = max(0, int(np.ceil(np.max(shifts[0]))))
    max_bottom = max(0, -int(np.ceil(np.max(-shifts[0]))))
    max_left = max(0, int(np.ceil(np.max(shifts[1]))))
    max_right = max(0, -int(np.ceil(np.max(-shifts[1]))))
    return BorderSpec(top=max_top, bottom=max_bottom, left=max_left, right=max_right)


def compute_adjusted_indices(params_for_mcorr: cmp.UpToMcorrParamDict) -> Optional[tuple[slice, slice]]:
    """
    Compute indices (sub-region to motion correct) corrected for crop, ndead and offset (to exclude dead pixels)
    """
    indices: tuple[slice, slice] = params_for_mcorr['motion']['indices']
    ndead = params_for_mcorr['conversion'].odd_row_ndead
    offset = params_for_mcorr['conversion'].odd_row_offset
    crop = params_for_mcorr['conversion'].crop
    
    # compute left border and exclude, if not already cropped out
    ndead_max = 0 if ndead is None else max(ndead)
    shift_max = 0 if offset is None else math.ceil(abs(offset) / 2)
    n_to_clip = ndead_max + shift_max
    curr_x_indices = indices[1]
    curr_start = 0 if curr_x_indices.start is None else int(curr_x_indices.start)
    n_clipped = curr_start + crop.left  # number of pixels currently removed from original image

    if n_to_clip > n_clipped:
        # figure out how to modify slice to clip out n_to_clip pixels
        # while maintaining the same phase if step != 1
        diff = n_to_clip - n_clipped  # minimum number of pixels to add to indices[1].start
        step = 1 if curr_x_indices.step is None else curr_x_indices.step
        new_start = curr_start + step * math.ceil(diff / step)
        new_x_indices = slice(new_start, curr_x_indices.stop, curr_x_indices.step)
        return indices[:1] + (new_x_indices,) + indices[2:]


def get_candidate_mcorr_result_files(tif_path: str, is_piecewise: bool) -> list[str]:
    """Get a list of possible filenames for motion correct results"""
    path_pattern_withdate = _build_motion_correct_path(tif_path, is_piecewise=is_piecewise, with_dt=True)
    path_nodate = _build_motion_correct_path(tif_path, is_piecewise=is_piecewise, with_dt=False)
    files_to_try = paths.get_all_timestamped_files(path_pattern_withdate.parent, path_pattern_withdate.name)
    if path_nodate.exists():
        files_to_try.append(str(path_nodate))
    return files_to_try

def load_mcorr_result(mmap_path: str) -> PlaneMcorrResult:
    info_file = os.path.splitext(mmap_path)[0] + '.npz'
    with np.load(info_file, allow_pickle=True) as info:
        shifts_rig = cast(np.ndarray, info['shifts_rig'])
        border_to_0 = int(info['border_to_0'].item())
        
        if 'shifts_els' in info:
            shifts_els = info['shifts_els']
            if shifts_els.ndim == 0:
                shifts_els = shifts_els.item()
            shifts_els = cast(np.ndarray, shifts_els)
        elif 'piecewise_info' in info:
            logging.warning('PiecewiseMCInfo will be removed soon, making this field un-unpicklable')
            piecewise_info: Optional[PiecewiseMCInfo] = info['piecewise_info'].item()
            shifts_els = piecewise_info.shifts_els if piecewise_info is not None else None
        else:
            shifts_els = None
        
        if 'border_asym' in info:
            border_asym = cast(BorderSpec, info['border_asym'].item())
        else:
            border_asym = None
    
    # compute border_asym if None
    if border_asym is None:
        # compute from shifts
        if shifts_els is None:
            border_asym = compute_border_asym(shifts_rig)
        else:
            border_asym = compute_border_asym(shifts_els)
    
    return PlaneMcorrResult(
        mmap_path=mmap_path, shifts_rig=shifts_rig, shifts_els=shifts_els,
        border_to_0=border_to_0, border_asym=border_asym
    )


def motion_correct_file(tif_file: str, motion_params: dict[str, Any], dview: Optional[Cluster] = None) -> PlaneMcorrResult:
    """Runs motion correction on the given file (does not attempt to load)"""
    # First, make a link to the tif_file with the current date, so that the mmap file will have it too
    tif_file_link = _make_hardlink_with_dt(tif_file)

    # Get path to output file we want to be created in the mcorr folder
    expected_file = _build_motion_correct_path(tif_file_link, is_piecewise=motion_params['pw_rigid'])

    # whether to first fit to subwindow and then apply to whole movie
    with set_output_location(expected_file):
        mcorr_obj = MotionCorrect(tif_file_link, **motion_params, dview=dview)
        # if we have indices, first compute using indices, then apply to the original movie.
        if any(s != slice(None) for s in motion_params['indices']):
            mcorr_obj.motion_correct(save_movie=False)
            actual_file = apply_mcorr_to_file(mcorr_obj, tif_file_link)
            if expected_file != actual_file:
                logging.debug(f'apply_mcorr_to_file expected to save to {expected_file}, but saved to {actual_file} instead')
        else:
            mcorr_obj.motion_correct(save_movie=True)
            actual_file = expected_file

    # extract shifts
    shifts_rig = np.array(mcorr_obj.shifts_rig).T  # transpose to dims x frames
    if motion_params["pw_rigid"] == True:
        x_shifts = mcorr_obj.x_shifts_els
        y_shifts = mcorr_obj.y_shifts_els
        shifts = [x_shifts, y_shifts]
        if hasattr(mcorr_obj, 'z_shifts_els'):
            shifts.append(mcorr_obj.z_shifts_els)
        shifts_els = np.array(shifts)
        border_asym = compute_border_asym(shifts_els)
    else:
        shifts_els = None
        border_asym = compute_border_asym(shifts_rig)
    
    info_file = actual_file.parent / (actual_file.stem + '.npz')
    np.savez(info_file,
             shifts_rig=shifts_rig,
             border_to_0=mcorr_obj.border_to_0,
             shifts_els=np.array(shifts_els),
             border_asym=np.array(border_asym))

    return PlaneMcorrResult(
        mmap_path=str(expected_file), shifts_rig=shifts_rig, shifts_els=shifts_els,
        border_to_0=mcorr_obj.border_to_0, border_asym=border_asym
    )


def apply_mcorr_to_file(mcorr_obj: MotionCorrect, input_file: str) -> Path:
    """Apply shifts from a MotionCorrect object to the given input file (returns output filename)"""
    # First, make a link to the tif_file with the current date, so that the mmap file will have it too
    file_link = _make_hardlink_with_dt(input_file)
    basename = _build_motion_correct_basename(file_link, is_piecewise=mcorr_obj.pw_rigid)

    saved_file = mcorr_obj.apply_shifts_movie(
        input_file, save_memmap=True, save_base_name=basename, remove_min=False)
    assert isinstance(saved_file, str), 'path returned when save_memmap is true'
    return Path(saved_file)


# ------------- transposition step ------------------- #


def get_transposed_mmap_name(orig_mmap_names: list[str], trans_params: cmp.TranspositionParams) -> str:
    if len(orig_mmap_names) > 1:
        # remove the _planeN part of the name b/c we're concatenating
        orig_mmap_name = re.sub(r'_plane\d+(_[^/\\]*)$', r'\1', orig_mmap_names[0])
    else:
        orig_mmap_name = orig_mmap_names[0]

    mmap_dir, mmap_basename = os.path.split(orig_mmap_name)
    mmap_t_basename = mmap_basename.replace('__', '_')
    mmap_t_basename = mmap_t_basename.replace('order_F', 'order_C')
    fn_params = decode_mmap_filename_dict(mmap_t_basename)

    # increase d2 (X) to reflect # of planes
    if len(orig_mmap_names) > 1:
        new_d2 = fn_params['d2'] * len(orig_mmap_names)
        mmap_t_basename = re.sub(r'd2_\d+_d3_\d+', f'd2_{new_d2}_d3_1', mmap_t_basename)
    
    # collect param strings to add to the filename to disambiguate
    # note this is just for convenience; actual decision for whether it can be used is from the params file
    extra_param_strings = []

    if trans_params.blur_kernel_size != 1:
        extra_param_strings.append(f'blur{trans_params.blur_kernel_size}')

    if trans_params.highpass_cutoff != 0:
        extra_param_strings.append(f'highpass{trans_params.highpass_cutoff:g}')      
        if trans_params.highpass_order != 4:  # only relevant if we are doing highpass filter
            extra_param_strings.append(f'order{trans_params.highpass_order}')
    
    if trans_params.add_to_mov != 0:
        extra_param_strings.append(f'add{trans_params.add_to_mov:g}')

    if len(extra_param_strings) > 0:
        # insert strings for non-default params into filename
        mmap_t_basename = re.sub(r'^(.*)_d1_', '\\1_' + '_'.join(extra_param_strings) + '_d1_', mmap_t_basename)

    return os.path.join(mmap_dir, mmap_t_basename)


def blur_forder_movie(input_mmap_path: str, output_mmap_path: str, ksize: int):
    """Gaussian-blur each frame of the given input F-order mmap file, saving to another mmap file"""
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError('ksize must be an odd positive integer')
    elif ksize == 1:
        logging.warning('blur call with ksize = 1 should be eliminated')
        # just make hard link to output file
        os.link(input_mmap_path, output_mmap_path)
        return

    input_mov: cm.movie = cm.load(input_mmap_path)
    T, *dims = input_mov.shape
    n_pix = int(np.prod(dims))

    # create output file
    output_mmap = np.memmap(output_mmap_path, dtype=np.float32, mode='w+', shape=(n_pix, T), order='F')
    for i, frame in enumerate(input_mov):
        sm_frame = cv2.GaussianBlur(
            frame, ksize=(ksize, ksize), sigmaX=ksize / 4, sigmaY=ksize / 4, borderType=cv2.BORDER_REPLICATE)
        output_mmap[:, i] = sm_frame.ravel(order='F')
    
    output_mmap.flush()
    del output_mmap


@contextmanager
def blurred_movies(input_mmap_files: list[str], ksize=1) -> Generator[list[str], None, None]:
    """
    Context manager that just yields the input files if ksize (kernel size) == 1. If ksize is an odd integer
    greater than 1, it puts blurred versions of each input movie into temporary files and
    yields the paths of these files, which are deleted when the context manager exits.
    """
    if ksize == 1:
        yield input_mmap_files
    else:
        output_paths: list[str] = []
        for input_path in input_mmap_files:
            logging.info(f'Using Gaussian blur of size {ksize}')
            file = tempfile.NamedTemporaryFile(suffix='.mmap', delete=False)
            output_path = file.name
            output_paths.append(output_path)
            file.close()
            blur_forder_movie(input_path, output_path, ksize=ksize)
        
        yield output_paths
        # on exit, delete each file
        for path in output_paths:
            os.remove(path)         


def transpose_flatten_mc_mmap(
        mc_result: MCResult, trans_params: cmp.TranspositionParams, fr: float,
        dview: Optional[Cluster] = None) -> str:
    """
    Saves motion-corrected data, flattened from 3D to 2D and transposed to iterate over time first (C-order).
    Note: I am breaking the usual rule in software that each function should do one thing for space and time efficiency.
    Since this involves iterating over and re-saving the entire post-motion-correction movie, it is the best time
        to do any other operations on the movie that work better on chunks of frames than patches of pixels.
    These also change the name of the output, so that extra data isn't normally saved, but if multiple versions of the
        transpose operation are run, they will be saved individually.

    These operations are:
        - Gaussian blur, enabled by setting blur_kernel_size > 1.
        - High-pass filtering, enabled by setting highpass_cutoff (in Hz) > 0.
    """
    mmap_files = mc_result.mmap_files
    highpass_cutoff = trans_params.highpass_cutoff
    highpass_order = trans_params.highpass_order
    add_to_movie = trans_params.add_to_mov

    expected_file = get_transposed_mmap_name(mmap_files, trans_params)
    logging.info(f'Saving transposed memmap to {os.path.basename(expected_file)}')

    with blurred_movies(mmap_files, ksize=trans_params.blur_kernel_size) as mmap_files:
        dims, T = get_file_size(mmap_files)
        if not isinstance(T, int):  # returns tuple of T for each file
            if any(t != T[0] for t in T):
                raise RuntimeError('Files should all have the same number of frames')
            T = int(T[0])

        n_planes = len(mmap_files)
        pixels_per_plane = int(np.prod(dims))
        n_pix = pixels_per_plane * n_planes

        # create output file for transposed data to allocate disk space, then immediately close
        big_mov = np.memmap(expected_file, dtype=np.float32, mode='w+', shape=(n_pix, T), order='C')
        bytes_per_pixel = big_mov.dtype.itemsize
        big_mov.flush()
        del big_mov

        for k_plane, input_path in enumerate(mmap_files):
            byte_offset = pixels_per_plane * k_plane * bytes_per_pixel * T
            # use plane-specific border if possible
            if mc_result.border_asym is not None:
                border = cast(Border, asdict(mc_result.border_asym[k_plane]))
            else:
                border = mc_result.border_to_0

            save_c_order_mmap_parallel(
                movie_path=input_path,
                base_name="",  # unused
                dview=dview,
                fr=fr,
                add_to_movie=add_to_movie,
                border_pixels=border,
                highpass_cutoff=highpass_cutoff,
                highpass_order=highpass_order,
                existing_output_path=expected_file,
                existing_output_offset=byte_offset
            )

    return expected_file


def do_or_load_transpose(
        mc_result: MCResult, params: cmp.UpToTransposeParamDict, fr: float, metadata: dict[str, Any],
        dview: Optional[Cluster] = None, load: Optional[bool] = None) -> str:
    """
    Either load existing result or do the transpose, saving a params file along with it

    load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
    """
    if load != False:   # try to load existing results
        expected_file = get_transposed_mmap_name(mc_result.mmap_files, params['transposition'])
        params_file = paths.params_file_for_result(expected_file)
        try:
            loaded_params = cmp.read_params_up_to_stage(cmp.AnalysisStage.TRANSPOSE, params_file)
            if cmp.do_params_match(params, loaded_params, metadata=metadata):
                if load is None:  # only log if we were unsure whether to load
                    logging.info('Using existing transposed file: ' + expected_file)
                return expected_file
        except FileNotFoundError:
            pass
        
    if load == True:
        raise NoMatchingResultError('Cannot find matching transposed file.')
    else:
        # we are doing the transpose
        res_file = transpose_flatten_mc_mmap(mc_result, params['transposition'], fr=fr, dview=dview)
        # write params file as well
        params_file = paths.params_file_for_result(res_file)
        cmp.write_params(params, params_file)
        return res_file
