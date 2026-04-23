"""
Motion correction utilities
"""
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict, replace
from functools import cached_property
import logging
import math
from multiprocessing.pool import Pool
import os
from pathlib import Path
import re
import tempfile
from typing import Optional, Union, ParamSpec, Any, cast
from typing_extensions import Self

import caiman as cm
from caiman.base.movies import get_file_size
from caiman.motion_correction import MotionCorrect, get_patch_centers
from caiman.paths import decode_mmap_filename_dict, memmap_frames_filename
from caiman.source_extraction.cnmf.params import MotionParams
import cv2
import holoviews as hv
from mesmerize_core.algorithms._utils import Cluster, save_c_order_mmap_parallel, make_projection_parallel
import numpy as np
import optype.numpy as onp
import scipy.interpolate as interp
import scipy.ndimage as ndi
import suite2p
from suite2p.registration import register as s2p_register, nonrigid, get_pc_metrics
import torch

from cmcode import caiman_analysis as cma, caiman_params as cmp
from cmcode.util import paths, footprints
from cmcode.util.image import BorderSpec, preprocess_proj_for_seed
from cmcode.util.types import NoMatchingResultError, Array4D


@dataclass(frozen=True)
class PlaneMcorrResult:
    """Shifts and borders from running mcorr on a single plane. Shifts *must* be corrected for any stride that was applied."""
    mmap_path: str                     # path to corrected movie
    shifts_rig: onp.Array2D            # rigid shifts of shape ({Y, X, [Z]}, frames)
    shifts_els: Optional[onp.Array3D]  # nonrigid shifts, if using, of shape ({Y, X, [Z]}, frames, patches)
    border_to_0: int                   # max border on any side
    border_asym: BorderSpec            # max border on each side

    def save(self, npz_path, **additional_fields):
        np.savez(
            npz_path, allow_pickle=True, shifts_rig=self.shifts_rig, shifts_els=np.asarray(self.shifts_els),
            border_to_0=self.border_to_0, border_asym=np.asarray(self.border_asym),
            **{name: np.asarray(val) for name, val in additional_fields.items()})


@dataclass(frozen=True)
class MCResult(paths.CustomPathMappable):
    """Single object that contains result paths of a motion correction run & allows access to shifts & borders through cached properties"""
    mmap_files: tuple[str, ...]
    dims: Optional[tuple[int, int]] = None
    motion_params: Optional[MotionParams] = None
    suite2p_reg_params: Optional[cmp.Suite2pRegistrationParams] = None
    mmap_file_transposed: Optional[str] = field(default=None, repr=False)  # deprecated, keep for unpickling

    # cache for plane results to lazy load other fields
    _loaded_results: list[Optional[PlaneMcorrResult]] = field(default_factory=lambda: [], repr=False)


    def __post_init__(self):
        if len(self._loaded_results) == 0:
            # fill with Nones
            for _ in range(len(self.mmap_files)):
                self._loaded_results.append(None)
        elif len(self._loaded_results) != len(self.mmap_files):
            raise TypeError('_loaded_results must have length matching # of planes')

        if self.mmap_file_transposed is not None:  # should only ever be set to non-none when unpickling
            raise ValueError('mmap_file_transposed must be None')

    def __getstate__(self):
        """Avoid saving cached results and properties when pickling"""
        state = self.__dict__.copy()
        keys = list(state.keys())
        for key in keys:
            if key in ('_loaded_results', 'piecewise_info'):
                del state[key]
            elif hasattr(type(self), key) and isinstance(getattr(type(self), key), cached_property):
                del state[key]
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

        if all(f in state for f in ('shifts_rig', 'shifts_els', 'border_to_0')):
            # use to initialize _loaded_results
            if 'border_asym' not in state:
                # infer
                if state['shifts_els'] is None:
                    state['border_asym'] = [compute_border_asym(shifts) for shifts in state['shifts_rig']]
                else:
                    state['border_asym'] = [compute_border_asym(shifts) for shifts in state['shifts_els']]
            
            shifts_els = [None] * len(state['mmap_files']) if state['shifts_els'] is None else state['shifts_els']
            
            state['_loaded_results'] = [
                PlaneMcorrResult(
                    mmap_path=path, shifts_rig=rig, shifts_els=els, border_asym=border, border_to_0=state['border_to_0']
                ) for path, rig, els, border in zip(state['mmap_files'], state['shifts_rig'], shifts_els, state['border_asym'])
            ]
        else:
            state['_loaded_results'] = [None] * len(state['mmap_files'])
        
        for f in ('shifts_rig', 'shifts_els', 'border_to_0', 'border_asym'):
            if f in state:
                del state[f]
        
        if 'motion_params' in state and isinstance(state['motion_params'], dict):
            state['motion_params'] = MotionParams(**state['motion_params'])

        # convert non-hashable types
        state['mmap_files'] = tuple(state['mmap_files'])

        self.__dict__.update(state)


    @property
    def used_suite2p(self) -> bool:
        return self.suite2p_reg_params is not None

    @property
    def info_files(self) -> list[str]:
        return [os.path.splitext(mmap_path)[0] + '.npz' for mmap_path in self.mmap_files]

    @property
    def n_planes(self) -> int:
        return len(self.mmap_files)

    @property
    def is_piecewise(self) -> bool:
        return self.shifts_els is not None

    @cached_property
    def shifts_rig(self) -> list[onp.Array2D]:
        """shifts from rigid step - (dims, frames) for each plane"""
        shifts: list[onp.Array2D] = []
        for loaded_res, info_file in zip(self._loaded_results, self.info_files):
            if loaded_res is not None:
                shifts.append(loaded_res.shifts_rig)
            else:
                with np.load(info_file, allow_pickle=True) as info:
                    shifts.append(info['shifts_rig'])
        return shifts

    @cached_property
    def shifts_els(self) -> Optional[list[onp.Array3D]]:
        """shifts *including* nonrigid - (dims, frames, patches) for each plane"""
        shifts: Optional[list[onp.Array3D]] = None

        for i, (loaded_res, info_file) in enumerate(zip(self._loaded_results, self.info_files)):
            if loaded_res is not None:
                this_shifts = loaded_res.shifts_els
            else:
                with np.load(info_file, allow_pickle=True) as info:
                    if 'shifts_els' in info:
                        this_shifts = info['shifts_els']
                        if this_shifts.ndim == 0:
                            this_shifts = this_shifts.item()
                    elif 'piecewise_info' in info:
                        logging.warning('PiecewiseMCInfo will be removed soon, making this field un-unpicklable')
                        piecewise_info: Optional[PiecewiseMCInfo] = info['piecewise_info'].item()
                        this_shifts = piecewise_info.shifts_els if piecewise_info is not None else None
                    else:
                        this_shifts = None

            if i == 0 and this_shifts is not None:
                shifts = []
            elif (this_shifts is not None) != (shifts is not None):
                raise RuntimeError('Expected all or none of the planes to have shifts_els')

            if this_shifts is not None and shifts is not None:
                shifts.append(this_shifts)
        return shifts

    
    @cached_property
    def border_to_0(self) -> int:
        """Integer maximal border among all dimensions/planes"""
        max_border: int = 0
        for loaded_res, info_file in zip(self._loaded_results, self.info_files):
            if loaded_res is not None:
                max_border = max(max_border, loaded_res.border_to_0)
            else:
                with np.load(info_file, allow_pickle=True) as info:
                    max_border = max(max_border, int(info['border_to_0'].item()))
        return max_border

    @cached_property
    def border_asym(self) -> list[BorderSpec]:
        """Border on each side, if known"""
        borders: list[BorderSpec] = []
        for kplane, (loaded_res, info_file) in enumerate(zip(self._loaded_results, self.info_files)):
            if loaded_res is not None:
                borders.append(loaded_res.border_asym)
            else:
                with np.load(info_file, allow_pickle=True) as info:
                    if 'border_asym' in info:
                        borders.append(info['border_asym'].item())
                    else:
                        # compute from shifts
                        if self.shifts_els is None:
                            borders.append(compute_border_asym(self.shifts_rig[kplane]))
                        else:
                            borders.append(compute_border_asym(self.shifts_els[kplane]))
        return borders


    @cached_property
    def shifts_rig_hv(self) -> hv.Dataset:
        """Make HoloViews dataset from rigid shifts"""
        shifts_all: onp.Array3D = np.stack(self.shifts_rig)
        nplanes, ndims, nframes = shifts_all.shape
        assert ndims == 2, 'Only 2D shifts supported'
        data_dims = {
            'plane': range(nplanes),
            'dim': ['y', 'x'],
            'frame': range(nframes),
            'shift': shifts_all,
        }
        return hv.Dataset(data_dims, ['frame', 'dim', 'plane'], 'shift')

    @cached_property
    def shifts_els_hv(self) -> Optional[hv.Dataset]:
        """Make HoloViews dataset from piecewise shifts, if they exist"""
        if self.shifts_els is None:
            return None

        if self.dims is None or self.motion_params is None:
            raise RuntimeError('dims and motion_params must be set to get shifts_els as HoloView dataset')

        # find patch locations
        patch_centers = get_full_movie_patch_centers(self.motion_params, self.suite2p_reg_params, self.dims)
        npatch_y = len(patch_centers['y'])
        npatch_x = len(patch_centers['x'])

        shifts_all_els: Array4D = np.stack(self.shifts_els)
        nplanes, ndims, nframes, _ = shifts_all_els.shape
        assert ndims == 2, 'Only 2D shifts supported'

        # unravel shifts into X/Y grid
        shifts_all_els = shifts_all_els.reshape(shifts_all_els.shape[:3] + (npatch_y, npatch_x), order='C')

        data_dims = {
            'plane': range(nplanes),
            'dim': ['y', 'x'],
            'frame': range(nframes),
            'shift': shifts_all_els,
            'ypatch': patch_centers['y'],
            'xpatch': patch_centers['x']
        }
        return hv.Dataset(data_dims, ['xpatch', 'ypatch', 'frame', 'dim', 'plane'], 'shift')


    def get_plane_result(self, plane: int) -> PlaneMcorrResult:
        return PlaneMcorrResult(
            mmap_path=self.mmap_files[plane],
            shifts_rig=self.shifts_rig[plane],
            shifts_els=self.shifts_els[plane] if self.shifts_els else None,
            border_to_0=self.border_asym[plane].ceil_scalar(),
            border_asym=self.border_asym[plane]
        )


    P = ParamSpec('P')
    def apply_path_mapper(self, path_mapper: paths.PathMapper[P], *args: P.args, **kwargs: P.kwargs) -> Self:
        """Implements CustomPathMappable by normalizing paths to memmap files"""
        return replace(self, mmap_files=tuple(path_mapper(list(self.mmap_files), *args, **kwargs)))

    def has_same_shifts_as(self, other: Self) -> bool:
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

    def get_mcorr_object(self, plane: int) -> MotionCorrect:
        """
        Make a motion correction object like the one these results were derived from,
        good enough to use apply_shifts_movie, for the given plane.
        """
        if self.used_suite2p:
            raise RuntimeError('Cannot create CaImAn MotionCorrect object from suite2p registration results')
        
        # try loading from info file
        with np.load(self.info_files[plane], allow_pickle=True) as info:
            if 'mcorr_obj' in info:
                return info['mcorr_obj'].item()
        
        # create from shifts
        if self.motion_params is None:
            raise RuntimeError('Must set motion params to recreate MotionCorrect object')

        # use dummy input movie
        mcorr_obj = MotionCorrect(np.array([]), **self.motion_params, dview=cma.cluster.dview)

        # here we have to correct for change in scale, if any (undo correction from creating MCResult)
        shifts_rig = self.shifts_rig[plane].copy()
        shifts_els = self.shifts_els[plane].copy() if self.shifts_els is not None else None
        for kdim, inds in enumerate(self.motion_params.indices):
            if inds.step is not None and inds.step != 1:
                shifts_rig[kdim] /= inds.step
                if shifts_els is not None:
                    shifts_els[kdim] /= inds.step

        # assign results from motion correction
        mcorr_obj.shifts_rig = list(shifts_rig.T)
        if shifts_els is not None:
            mcorr_obj.x_shifts_els = list(shifts_els[0])
            mcorr_obj.y_shifts_els = list(shifts_els[1])
            if self.motion_params.is3D:
                mcorr_obj.z_shifts_els = list(shifts_els[2])

        return mcorr_obj


    def get_pc_metrics(self, plane: int) -> tuple[onp.Array2D[np.floating], Array4D[np.floating]]:
        """
        Compute principal components and averages of top/bottom-weighted frames
            of motion-corrected movie for a single plane, using suite2p
        See: https://suite2p.readthedocs.io/en/latest/api/registration/#suite2p.registration.metrics.get_pc_metrics
        Outputs:
            - tPC:      Temporal PC weights of shape (n_samples, nPC), describing how each PC varies
                        across the subsampled frames.
            - regPC:    Average of top and bottom weighted frames for each PC, shape (2, nPC, Ly_crop, Lx_crop)
                        where index 0 is pclow and index 1 is pchigh.
        """
        if plane < 0 or plane >= self.n_planes:
            raise ValueError(f'Plane must be from 0 to {self.n_planes - 1}, inclusive.')

        mov = cm.load(self.mmap_files[plane])
        _T, *dims = mov.shape
        slices = self.border_asym[plane].slices(dims)
        mov_center = mov[:, *slices]

        if torch.cuda.is_available():
            device = torch.device('cuda')  # type: ignore
        else:
            device = torch.device('cpu')  # type: ignore
        
        return get_pc_metrics(mov_center, device=device)[:2]


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


def compute_border_asym(shifts: np.ndarray) -> BorderSpec:
    """
    Given shifts array with dimension along the first axis, compute asymmetric border
    (max border on each side).
    """
    max_top = max(0, np.max(shifts[0]))
    max_bottom = max(0, -np.max(-shifts[0]))
    max_left = max(0, np.max(shifts[1]))
    max_right = max(0, -np.max(-shifts[1]))
    return BorderSpec(top=max_top, bottom=max_bottom, left=max_left, right=max_right)


def compute_adjusted_indices(params_for_mcorr: cmp.UpToMcorrParamStruct) -> Optional[tuple[slice, ...]]:
    """
    Compute indices (sub-region to motion correct) corrected for crop, ndead and offset (to exclude dead pixels)
    """
    indices = params_for_mcorr.motion.indices
    ndead = params_for_mcorr.conversion.odd_row_ndead
    offset = params_for_mcorr.conversion.odd_row_offset
    crop = params_for_mcorr.conversion.crop
    
    # compute left border and exclude, if not already cropped out
    ndead_max = 0 if ndead is None else max(ndead)
    shift_max = 0 if offset is None else math.ceil(abs(offset) / 2)
    n_to_clip = ndead_max + shift_max
    curr_x_indices = indices[1]
    curr_start = 0 if curr_x_indices.start is None else int(curr_x_indices.start)
    n_clipped = curr_start + crop['left']  # number of pixels currently removed from original image

    if n_to_clip > n_clipped:
        # figure out how to modify slice to clip out n_to_clip pixels
        # while maintaining the same phase if step != 1
        diff = n_to_clip - n_clipped  # minimum number of pixels to add to indices[1].start
        step = 1 if curr_x_indices.step is None else curr_x_indices.step
        new_start = curr_start + step * math.ceil(diff / step)
        new_x_indices = slice(new_start, curr_x_indices.stop, curr_x_indices.step)
        return indices[:1] + (new_x_indices,) + indices[2:]


def get_full_movie_patch_centers(
    motion_params: MotionParams, suite2p_reg_params: Optional[cmp.Suite2pRegistrationParams], dims: tuple[int, int]
    ) -> dict[str, list[float]]:
    """Get list of nonrigid patch center locations along each dimension (assumes a regular grid)"""
    if suite2p_reg_params is not None:
        yblocks, xblocks, nblocks, _ = get_suite2p_blocks_for_full_movie(
            block_size=suite2p_reg_params.block_size, dims=dims, indices=motion_params.indices
        )
        return {
            'y': [np.mean(yblock).item() for yblock in yblocks[::nblocks[1]]],  # take first in each row
            'x': [np.mean(xblock).item() for xblock in xblocks[:nblocks[1]]]   # take first in each column
        }
    else:
        effective_dims = tuple(len(range(d)[dim_inds]) for d, dim_inds in zip(dims, motion_params.indices))

        patch_centers_orig = get_patch_centers(
            effective_dims, strides=motion_params.strides, overlaps=motion_params.overlaps,
            upsample_factor_grid=motion_params.upsample_factor_grid, shifts_opencv=motion_params.shifts_opencv)

        # if only a portion of the original image was used, offset/multiply patch centers to now apply to the whole movie
        # compute one dimension at a time. Copied from caiman.motion_correction.apply_shifts_movie.
        patch_centers: dict[str, list[float]] = {}
        for dim_inds, dim_centers_orig, dim in zip(motion_params.indices, patch_centers_orig, ['y', 'x']):
            start = dim_inds.start if dim_inds.start is not None else 0
            step = dim_inds.step if dim_inds.step is not None else 1
            patch_centers[dim] = [float(start + step * center) for center in dim_centers_orig]
        return patch_centers


def get_candidate_mcorr_result_files(tif_path: str, is_piecewise: bool) -> list[str]:
    """Get a list of possible filenames for motion correct results"""
    path_pattern_withdate = _build_motion_correct_path(tif_path, is_piecewise=is_piecewise, with_dt=True)
    path_nodate = _build_motion_correct_path(tif_path, is_piecewise=is_piecewise, with_dt=False)
    files_to_try = paths.get_all_timestamped_files(path_pattern_withdate.parent, path_pattern_withdate.name)
    if path_nodate.exists():
        files_to_try.append(str(path_nodate))
    return files_to_try


def motion_correct_file(
    tif_file: str, mcorr_params: cmp.McorrParamStruct, dview: Optional[Cluster] = None) -> PlaneMcorrResult:
    if mcorr_params.mcorr_extra.use_suite2p:
        return motion_correct_file_suite2p(tif_file, mcorr_params.suite2p_register, mcorr_params.motion)
    else:
        return motion_correct_file_caiman(tif_file, mcorr_params.motion, dview=dview)


def motion_correct_file_caiman(tif_file: str, motion_params: MotionParams, dview: Optional[Cluster] = None) -> PlaneMcorrResult:
    """Runs motion correction on the given file (does not attempt to load)"""
    # First, make a link to the tif_file with the current date, so that the mmap file will have it too
    with paths.linked_timestamped_path(tif_file) as tif_file_link:
        # Get path to output file we want to be created in the mcorr folder
        expected_file = _build_motion_correct_path(tif_file_link, is_piecewise=motion_params.pw_rigid)

        # whether to first fit to subwindow and then apply to whole movie
        with set_output_location(expected_file):
            mcorr_obj = MotionCorrect(tif_file_link, **motion_params, dview=dview)
            # if we have indices, first compute using indices, then apply to the original movie.
            if any(s != slice(None) for s in motion_params.indices):
                mcorr_obj.motion_correct(save_movie=False)
                actual_file = apply_mcorr_to_file_caiman(tif_file_link, mcorr_obj)
                if expected_file != actual_file:
                    logging.debug(f'apply_mcorr_to_file expected to save to {expected_file}, but saved to {actual_file} instead')
            else:
                mcorr_obj.motion_correct(save_movie=True)
                actual_file = expected_file

    # extract shifts
    shifts_rig = np.array(mcorr_obj.shifts_rig).T  # transpose to dims x frames
    if motion_params.pw_rigid:
        # note: x_shifts_els and y_shifts_els are swapped!!
        y_shifts = mcorr_obj.x_shifts_els
        x_shifts = mcorr_obj.y_shifts_els
        shifts = [y_shifts, x_shifts]
        if hasattr(mcorr_obj, 'z_shifts_els'):
            shifts.append(mcorr_obj.z_shifts_els)
        shifts_els = np.array(shifts)
    else:
        shifts_els = None

    # correct for change in scale, if any
    for kdim, inds in enumerate(motion_params.indices):
        if inds.step is not None and inds.step != 1:
            shifts_rig[kdim] *= inds.step
            if shifts_els is not None:
                shifts_els[kdim] *= inds.step
    
    border_asym = compute_border_asym(shifts_els if shifts_els is not None else shifts_rig)
    border_to_0 = border_asym.ceil_scalar()

    result = PlaneMcorrResult(
        mmap_path=str(expected_file), shifts_rig=shifts_rig, shifts_els=shifts_els,
        border_to_0=border_to_0, border_asym=border_asym
    )

    info_file = actual_file.parent / (actual_file.stem + '.npz')
    result.save(info_file, mcorr_obj=mcorr_obj) # this is complicated, save the original MotionCorrect object just in case

    return result


def motion_correct_file_suite2p(tif_file: str, reg_params: cmp.Suite2pRegistrationParams, motion_params: MotionParams) -> PlaneMcorrResult:
    """Motion correct one plane using Suite2p registration"""
    reg_settings: dict = suite2p.default_settings()['registration']
    reg_settings.update(asdict(reg_params))

    indices = motion_params.indices

    f_raw = cm.load(tif_file)  # loads TIF as memmap
    T, ny, nx = f_raw.shape  # full movie size
    f_raw = f_raw[(slice(None),) + indices]
    _, ny_cropped, nx_cropped = f_raw.shape

    # get filename of output mmap
    with paths.linked_timestamped_path(tif_file) as input_file_plus_dt:
        output_path = _build_motion_correct_path(input_file_plus_dt, is_piecewise=reg_params.nonrigid)
    
    # make mmap to hold output data
    # if we are using subindices, we will have to recompute this afterwards, but we still need a
    # place to hold corrected data during registration_wrapper (e.g, if two_step_registration is used)
    output_mmap, _, _ = cm.load_memmap(str(output_path), mode='w+')  # pixels x time
    assert isinstance(output_mmap, np.memmap)
    f_reg = output_mmap.T.reshape((T, ny, nx), order='F')
    f_reg = f_reg[(slice(None),) + indices]

    res = suite2p.registration_wrapper(f_reg, f_raw, settings=reg_settings)
    output_mmap.flush()
    del output_mmap, f_reg

    shifts_rig = np.stack([res['yoff'], res['xoff']])
    if reg_params.nonrigid:
        nonrigid_offsets = np.stack([res['yoff1'], res['xoff1']])
        # combine rigid shifts with nonrigid offsets to get nonrigid shifts
        shifts_els = nonrigid_offsets + shifts_rig[:, :, np.newaxis]
    else:
        shifts_els = None

    # convert xrange/yrange to my border format
    border_asym = BorderSpec(
        left=res['xrange'][0],
        right=nx_cropped - res['xrange'][1],
        top=res['yrange'][0],
        bottom=ny_cropped - res['yrange'][1]
    )

    # correct for change in scale, if any
    for kdim, inds in enumerate(indices):
        if inds.step is not None and inds.step != 1:
            shifts_rig[kdim] *= inds.step
            if shifts_els is not None:
                shifts_els[kdim] *= inds.step
            if kdim == 0:
                border_asym *= BorderSpec(left=1, right=1, top=inds.step, bottom=inds.step)
            else:
                border_asym *= BorderSpec(left=inds.step, right=inds.step, top=1, bottom=1)
    
    border_asym = BorderSpec.min(border_asym, BorderSpec.maximal(shape=(ny, nx)))

    result = PlaneMcorrResult(
        mmap_path=str(output_path), shifts_rig=shifts_rig, shifts_els=shifts_els,
        border_asym=border_asym, border_to_0=border_asym.ceil_scalar()
    )

    # now apply to full movie if necessary
    if any(inds != slice(None) for inds in indices):
        logging.info('Applying shifts to full movie using Suite2p')
        path = apply_mcorr_to_file_suite2p(tif_file, result, motion_params, reg_params)
        assert str(path) == result.mmap_path, 'Should save to same path'

    # save shifts
    info_file = output_path.parent / (output_path.stem + '.npz')
    result.save(info_file)
    
    return result    


def apply_mcorr_to_file_caiman(input_file: str, mcorr_obj: MotionCorrect) -> Path:
    """Apply shifts from a MotionCorrect object to the given input file (returns output filename)"""
    tif_path_with_timestamp = paths.add_timestamp_to_path(input_file)
    basename = _build_motion_correct_basename(tif_path_with_timestamp, is_piecewise=mcorr_obj.pw_rigid)

    saved_file = mcorr_obj.apply_shifts_movie(
        input_file, save_memmap=True, save_base_name=basename, remove_min=False)
    assert isinstance(saved_file, str), 'path returned when save_memmap is true'
    return Path(saved_file)


def apply_mcorr_to_file_suite2p(
    input_file: str, result: PlaneMcorrResult, motion_params: MotionParams,
    reg_params: Optional[cmp.Suite2pRegistrationParams]) -> Path:
    """
    Use suite2p to apply shifts from the given plane to the given input movie,
    assuming it is the same size as the original movie (before cropping), and
    save at the appropriate mcorr output path (returns this path)
    This also works on shifts computed using CaImAn. See also apply_mcorr_to_file_caiman.
    """
    # open the input file
    f_raw = cm.load(input_file)  # loads TIF as memmap
    T, *dims = f_raw.shape  # full movie size
    if len(dims) != 2:
        raise ValueError('Input movie should be 2-dimensional')
    ny, nx = dims

    # open the output file
    is_piecewise = reg_params.nonrigid if reg_params is not None else motion_params.pw_rigid
    with paths.linked_timestamped_path(input_file) as input_file_plus_dt:
        output_path = _build_motion_correct_path(input_file_plus_dt, is_piecewise=is_piecewise)
    
    # make mmap to hold output data
    output_mmap, _, _ = cm.load_memmap(str(output_path), mode='w+')  # pixels x time
    f_reg = output_mmap.T.reshape((T, ny, nx), order='F')

    # get blocks to use
    if result.shifts_els is None:
        blocks = None
        yoff1 = None
        xoff1 = None
    else:
        if reg_params is not None:
            blocks = get_suite2p_blocks_for_full_movie(
                block_size=reg_params.block_size, dims=(ny, nx), indices=motion_params.indices
            )
        else:
            patch_centers = get_full_movie_patch_centers(motion_params, reg_params, (ny, nx))
            blocks = get_suite2p_blocks_from_centers(patch_centers)

        # convert shifts_els to offsets (deviations from rigid)
        yoff1 = result.shifts_els[0] - result.shifts_rig[0][:, :, np.newaxis]
        xoff1 = result.shifts_els[0] - result.shifts_rig[0][:, :, np.newaxis]
    
    s2p_register.shift_frames_and_write(
        f_raw, f_reg, yoff=result.shifts_rig[0], xoff=result.shifts_rig[1],
        yoff1=yoff1, xoff1=xoff1, blocks=blocks)

    return output_path


def get_suite2p_blocks_for_full_movie(
    block_size: tuple[int, int], dims: tuple[int, int], indices: tuple[slice, ...] = (slice(None), slice(None))
    ) -> tuple[list[onp.Array1D[np.intp]], list[onp.Array1D[np.intp]], list[int], tuple[int, int]]:
    """
    Get block definitions for suite2p registration to use to apply shifts to the full movie (of shape `dims`)
    (before subsampling pixels according to `indices`, if applicable).
    Only the first 4 outputs of `make_blocks` are returned (these are all that are needed for `shift_frames`).
    """
    # determine indices used for registration along each axis
    y_full, x_full = dims
    yinds_reg = np.arange(y_full)[indices[0]]
    xinds_reg = np.arange(x_full)[indices[1]]

    # get blocks for sub-sampled movie
    yblocks_reg, xblocks_reg, nblocks, block_size_reg, *_ = nonrigid.make_blocks(len(yinds_reg), len(xinds_reg), block_size)

    # adjust to be relative to full movie
    block_size = (
        block_size_reg[0] * (indices[0].step if indices[0].step else 1),
        block_size_reg[1] * (indices[1].step if indices[1].step else 1)
    )

    yblocks = [np.array([yinds_reg[yblock[0]], yinds_reg[yblock[1]-1]+1]) for yblock in yblocks_reg]
    xblocks = [np.array([xinds_reg[xblock[0]], xinds_reg[xblock[1]-1]+1]) for xblock in xblocks_reg]

    return yblocks, xblocks, nblocks, block_size


def get_suite2p_blocks_from_centers(
    patch_centers: dict[str, list[float]], block_size: tuple[int, int] = (0, 0)
    ) -> tuple[list[onp.Array1D[np.intp]], list[onp.Array1D[np.intp]], list[int], tuple[int, int]]:
    """
    Get block definitions for suite2p registration from just the patch centers along each axis (not the full grid).
    Only the first 4 outputs of `make_blocks` are returned. The block size doesn't matter for `shift_frames`, so
    for simplicity it is 0 by default, but this can be changed by setting block_size.
    """
    yblocks: list[onp.Array1D] = []
    xblocks: list[onp.Array1D] = []
    # iterate in row-major order
    for y_center in patch_centers['y']:
        for x_center in patch_centers['x']:
            yblocks.append(np.array([y_center - block_size[0] / 2, y_center + block_size[0] / 2]))
            xblocks.append(np.array([x_center - block_size[1] / 2, x_center + block_size[1] / 2]))
    
    nblocks = [len(patch_centers['y']), len(patch_centers['x'])]
    return yblocks, xblocks, nblocks, block_size


# ------------- transposition step ------------------- #


def get_transposed_mmap_name(orig_mmap_names: Sequence[str], trans_params: cmp.TranspositionParams) -> str:
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
    # TODO maybe just date the files, like the mcorr results, rather than putting parameters in the filename?
    # but then it would be less clear which transposed file(s) comes from which original plane files, if that matters
    extra_param_strings = []

    if trans_params.blur_kernel_size != 1:
        extra_param_strings.append(f'blur{trans_params.blur_kernel_size}')

    if trans_params.highpass_cutoff != 0:
        extra_param_strings.append(f'highpass{trans_params.highpass_cutoff:g}')
     
        if trans_params.highpass_order != 4:  # only relevant if we are doing highpass filter
            extra_param_strings.append(f'order{trans_params.highpass_order}')
    
    if trans_params.add_to_mov != 0:
        extra_param_strings.append(f'add{trans_params.add_to_mov:g}')

    if trans_params.remove_bg_mean:
        extra_param_strings.append('bgremoved')

        if trans_params.bg_filter_size != 20:  # only relevant if background-remove step is on
            extra_param_strings.append(f'filtersize{trans_params.bg_filter_size}')

    if trans_params.remove_bg_component:
        extra_param_strings.append('bgcompremoved')
        
    if (trans_params.remove_bg_mean or trans_params.remove_bg_component) and trans_params.bg_scale != 1.:
        extra_param_strings.append(f'scale{trans_params.bg_scale:g}')

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
    T, *dims = input_mov.shape  # type: ignore
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
def blurred_movies(input_mmap_files: Sequence[str], ksize=1) -> Generator[Sequence[str], None, None]:
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


def estimate_bg_mean(
    plane_mean: onp.Array2D[np.floating], border: BorderSpec, filter_size: int, return_center_only=True,
    dview: Optional[Cluster] = None) -> onp.Array2D[np.floating]:
    """Make estimate of local background level based on min-filtering the mean projection"""
    # get mean projection
    slices = border.slices(plane_mean.shape)
    center = plane_mean[slices]
    center_filtered = ndi.minimum_filter(center, filter_size)
    center_filtered = ndi.gaussian_filter(center_filtered, filter_size // 2)
    if return_center_only:
        return center_filtered
    else:
        filtered = np.zeros_like(plane_mean)
        filtered[slices] = center_filtered
        return filtered


def _bg_component_helper(
    args: tuple[str, slice, onp.Array1D[np.bool_]]) -> tuple[onp.Array1D[np.float32], onp.Array1D[np.float32]]:
    """Compute mean f and b on slice, for pixels in given mask"""
    mmap_path, time_slice, pixel_mask = args
    Y, _, _ = cm.load_memmap(mmap_path, 'r')
    chunk = Y[pixel_mask, time_slice]
    f = np.mean(chunk, axis=0)
    b = chunk @ f
    return b, f


def estimate_bg_component(
    plane_mmap_path: str, border: BorderSpec, corrected_mean_proj: onp.Array2D[np.floating],
    cellpose_params: cmp.CellposeParams, chunk_size=1000, dview: Optional[Cluster] = None,
    ) -> tuple[onp.Array2D[np.floating], onp.Array1D[np.floating]]:
    """Make estimate of mean background spatial and temporal components"""
    dims, T = get_file_size(plane_mmap_path)
    assert isinstance(T, int), 'Should be scalar for scalar input'
    assert len(dims) == 2, 'Plane is 2-dimensional'

    # first identify non-cell pixels using cellpose, then find mean activation across movie
    bg_seed_params = cmp.SeedParams(use_cellpose=True, cellpose_params=cellpose_params)
    seed = footprints.make_spatial_seed(corrected_mean_proj, bg_seed_params)
    not_cell_pixels = (seed.sum(axis=1) == 0) & border.flatmask(dims)  # outside cells, within borders
    cell_pixels = ~not_cell_pixels & border.flatmask(dims)  # inside cells, within borders

    coords = np.mgrid[0:dims[0], 0:dims[1]]
    Y_flat = coords[0].ravel(order='F')
    X_flat = coords[1].ravel(order='F')
    not_cell_coords = np.stack([Y_flat[not_cell_pixels], X_flat[not_cell_pixels]], axis=1)
    cell_coords = np.stack([Y_flat[cell_pixels], X_flat[cell_pixels]], axis=1)
    
    # estimate component in the same way as caiman - first get temporal mean, then full spatial component
    args = [
        (plane_mmap_path, slice(start, min(start + chunk_size, T)), not_cell_pixels)
        for start in range(0, T, chunk_size)
    ]
    if dview is None:
        map_fn = map
    elif isinstance(dview, Pool):
        map_fn = dview.map
    else:
        map_fn = dview.map_sync
    
    bf_each_chunk = map_fn(_bg_component_helper, args)
    f = np.concatenate([res[1] for res in bf_each_chunk])
    b_not_cell = np.sum([res[0] for res in bf_each_chunk], axis=0)
    b_not_cell /= np.linalg.norm(f) ** 2  # normalize so that if a row i of Y == f, b[i] = 1


    # interpolate to fill in holes where cells are - first linear, then nearest
    interp_lin = interp.LinearNDInterpolator(not_cell_coords, b_not_cell)
    b_cell = interp_lin(cell_coords)
    is_outside_hull = np.isnan(b_cell)
    if np.any(is_outside_hull):
        # fill in pixels outside convex hull with nearest interpolator
        outside_hull_coords = cell_coords[is_outside_hull, :]
        interp_nearest = interp.NearestNDInterpolator(not_cell_coords, b_not_cell)
        b_cell[is_outside_hull] = interp_nearest(outside_hull_coords)
    
    b = np.zeros(dims, dtype=np.float32)
    b[not_cell_coords[:, 0], not_cell_coords[:, 1]] = b_not_cell
    b[cell_coords[:, 0], cell_coords[:, 1]] = b_cell

    return b, f


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

    expected_file = get_transposed_mmap_name(mmap_files, trans_params)
    logging.info(f'Saving transposed memmap to {os.path.basename(expected_file)}')

    with blurred_movies(mmap_files, ksize=trans_params.blur_kernel_size) as mmap_files:
        dims, T = get_file_size(list(mmap_files))  # note this *has* to be a list and not a tuple
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

        for k_plane, (input_path, border) in enumerate(zip(mmap_files, mc_result.border_asym)):
            # offset by bytes already written in previous planes
            byte_offset = pixels_per_plane * k_plane * bytes_per_pixel * T

            add_to_movie = trans_params.add_to_mov
            add_to_movie_temporal = None
            if trans_params.remove_bg_mean or trans_params.remove_bg_component:
                plane_mean = make_projection_parallel(input_path, 'mean', dview=dview)
                filter_size = trans_params.bg_filter_size
                bg_mean = estimate_bg_mean(plane_mean, border, filter_size, dview=dview, return_center_only=False)

                if trans_params.remove_bg_mean:
                    # implement remove_bg_mean by setting add_to_movie to a 2D array (can be size of whole plane or just center)
                    add_to_movie = add_to_movie - trans_params.bg_scale * bg_mean

                if trans_params.remove_bg_component:
                    # technically not mutually exclusive with remove_bg_mean, but probably it should be
                    corrected_mean_proj = plane_mean - bg_mean
                    bg_mean_spatial, add_to_movie_temporal = estimate_bg_component(
                        input_path, border, corrected_mean_proj, trans_params.bg_cellpose_params, dview=dview)
                    add_to_movie = add_to_movie - trans_params.bg_scale * bg_mean_spatial

            save_c_order_mmap_parallel(
                movie_path=input_path,
                base_name="",  # unused
                dview=dview,
                fr=fr,
                add_to_movie=add_to_movie,
                add_to_movie_temporal=add_to_movie_temporal,
                border_pixels=border.ceil(),
                highpass_cutoff=highpass_cutoff,
                highpass_order=highpass_order,
                existing_output_path=expected_file,
                existing_output_offset=byte_offset
            )

    return expected_file


def do_or_load_transpose(
        mc_result: MCResult, params: cmp.UpToTransposeParamStruct, fr: float, metadata: dict[str, Any],
        dview: Optional[Cluster] = None, load: Optional[bool] = None) -> str:
    """
    Either load existing result or do the transpose, saving a params file along with it

    load: Whether to try loading previously-computed results.
            None: use previous results if params match, otherwise compute anew
            True: use previous results if params match, otherwise raise NoMatchingResultError
            False: recompute results even if they already exist.
    """
    if load is not False:   # try to load existing results
        expected_file = get_transposed_mmap_name(mc_result.mmap_files, params.transposition)
        params_file = paths.params_file_for_result(expected_file)
        try:
            loaded_params = cmp.UpToTransposeParamStruct.read_from_file(params_file)
            if loaded_params.do_params_match(params, metadata=metadata, stage=cmp.AnalysisStage.TRANSPOSE):
                if load is None:  # only log if we were unsure whether to load
                    logging.info('Using existing transposed file: ' + expected_file)
                return expected_file
        except FileNotFoundError:
            pass
        
    if load is True:
        raise NoMatchingResultError('Cannot find matching transposed file.')
    else:
        # we are doing the transpose
        res_file = transpose_flatten_mc_mmap(mc_result, params.transposition, fr=fr, dview=dview)
        # write params file as well
        params_file = paths.params_file_for_result(res_file)
        params.write_params(params_file, stage=cmp.AnalysisStage.TRANSPOSE)
        return res_file


@dataclass
class PiecewiseMCInfo:
    """Deprecated, used only to allow unpickling previous results"""
    shifts_els: np.ndarray
    patch_xy_inds: Optional[list[tuple[int, int]]] = None

    def __post_init__(self):
        if self.patch_xy_inds is not None and any(inds is None for inds in self.patch_xy_inds):
            # occurs if shifts_opencv is True
            self.patch_xy_inds = None
