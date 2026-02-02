"""
Motion correction utilities
TODO integrate this with mesmerize-core to facilitate trying multiple parameter combinations
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from multiprocessing import parent_process
import os
import psutil
import re
import shutil
from typing import Optional, Generator, ParamSpec, cast

import caiman as cm
from caiman.base.movies import get_file_size
from caiman.motion_correction import MotionCorrect, get_patch_centers
from caiman.paths import decode_mmap_filename_dict, memmap_frames_filename, fn_relocated
from caiman.source_extraction.cnmf.params import CNMFParams
import cv2
import holoviews as hv
import numpy as np
from numpy.typing import DTypeLike
from scipy import signal

from cmcode import in_jupyter, caiman_analysis as cma
from cmcode.util.image import BorderSpec
from cmcode.util.paths import PathMapper, CustomPathMappable

if in_jupyter():
    from tqdm.notebook import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


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
class MCResult(CustomPathMappable):
    mmap_files: list[str]
    mmap_file_transposed: str
    border_to_0: int
    border_asym: list[BorderSpec]  # border on each side (old results just repeat border_to_0)
    shifts_rig: list[np.ndarray]
    shifts_els: Optional[list[np.ndarray]] = None
    dims: Optional[tuple[int, int]] = None
    motion_params: Optional[dict] = None

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
    def apply_path_mapper(self, path_mapper: PathMapper[P], *args: P.args, **kwargs: P.kwargs) -> 'MCResult':
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


def _build_motion_correct_filename(filepath: str, is_piecewise: bool = True) -> str:
    """Determine what caiman will save the motion-corrected movie as (to find if previously calculated)"""
    dims, T = get_file_size(filepath)
    assert isinstance(T, int), 'T should be int when taking file size of one movie'
    filedir, filename = os.path.split(filepath)
    base_name = os.path.splitext(filename)[0] + ('_els_' if is_piecewise else '_rig_')
    fname_tot = memmap_frames_filename(base_name, dims, T, order='F')
    dir_mcorr = re.sub(r'\bconversion\b', 'mcorr', filedir)
    fname_tot = os.path.join(dir_mcorr, fname_tot)
    return fname_tot


def _to_transposed_flattened_mmap_name(orig_mmap_names: list[str], do_transpose=True,
                                       blur_kernel_size=1, highpass_cutoff=0.) -> str:
    if len(orig_mmap_names) > 1:
        # remove the _planeN part of the name b/c we're concatenating
        orig_mmap_name = re.sub(r'_plane\d+(_[^/\\]*)$', r'\1', orig_mmap_names[0])
    else:
        orig_mmap_name = orig_mmap_names[0]
    orig_mmap_name = orig_mmap_name
    mmap_dir, mmap_basename = os.path.split(orig_mmap_name)
    mmap_t_basename = mmap_basename.replace('__', '_')
    if do_transpose:
        mmap_t_basename = mmap_t_basename.replace('order_F', 'order_C')
    fn_params = decode_mmap_filename_dict(mmap_t_basename)

    # increase d2 (X) to reflect # of planes
    if len(orig_mmap_names) > 1:
        new_d2 = fn_params['d2'] * len(orig_mmap_names)
        mmap_t_basename = re.sub(r'd2_\d+_d3_\d+', f'd2_{new_d2}_d3_1', mmap_t_basename)
    
    if blur_kernel_size != 1:
        mmap_t_basename = re.sub(r'^(.*)_d1_', f'\\1_blur{blur_kernel_size}_d1_', mmap_t_basename)

    if highpass_cutoff != 0:
        mmap_t_basename = re.sub(r'^(.*)_d1_', f'\\1_highpass{highpass_cutoff:.2g}_d1_', mmap_t_basename)

    return os.path.join(mmap_dir, mmap_t_basename)


@contextmanager
def set_output_location(output_path: str) -> Generator[None, None, None]:
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
    (max border on each side, rounding up to the nearest integer).
    """
    max_top = max(0, int(np.ceil(np.max(shifts[0]))))
    max_bottom = max(0, -int(np.ceil(np.max(-shifts[0]))))
    max_left = max(0, int(np.ceil(np.max(shifts[1]))))
    max_right = max(0, -int(np.ceil(np.max(-shifts[1]))))
    return BorderSpec(top=max_top, bottom=max_bottom, left=max_left, right=max_right)


def motion_correct_file(tif_file: str, params: CNMFParams, cluster_args: Optional[dict] = None, force: bool = False
                         ) -> tuple[str, np.ndarray, int, BorderSpec, Optional[np.ndarray], bool]:
    """
    Runs motion correction on the given file and returns:
        - path(s) to the mmap file(s),
        - rigid shifts
        - border pixels
        - BorderSpec for border on each side (border_asym)
        - nonrigid shifts (if doing pw_rigid)
        - whether a new result was computed (always True if force is True)
    """
    expected_file = _build_motion_correct_filename(tif_file, params.motion['pw_rigid'])
    expected_info_file = re.sub(r'.mmap$', '.npz', expected_file)

    if not force and os.path.exists(expected_file) and os.path.exists(expected_info_file):
        logging.info('Using existing motion correction results from ' + expected_file)
        with np.load(expected_info_file, allow_pickle=True) as info:
            shifts_rig = info['shifts_rig']
            border_to_0 = int(info['border_to_0'].item())
            if 'shifts_els' in info:
                shifts_els = info['shifts_els']
                if shifts_els.ndim == 0:
                    shifts_els = shifts_els.item()
            elif 'piecewise_info' in info:
                logging.warning('PiecewiseMCInfo will be removed soon, making this field un-unpicklable')
                piecewise_info: Optional[PiecewiseMCInfo] = info['piecewise_info'].item()
                shifts_els = piecewise_info.shifts_els if piecewise_info is not None else None
            else:
                shifts_els = None

            if 'border_asym' in info:
                border_asym = cast(BorderSpec, info['border_asym'].item())
            else:
                # compute from shifts
                if shifts_els is None:
                    border_asym = compute_border_asym(shifts_rig)
                else:
                    border_asym = compute_border_asym(shifts_els)              

        return expected_file, shifts_rig, border_to_0, border_asym, shifts_els, False
    
    if parent_process() is None and cluster_args is not None:
        cma.cluster.start(**cluster_args)

    # whether to first fit to subwindow and then apply to whole movie
    use_apply = any(s != slice(None) for s in params.motion['indices'])

    params.change_params({'data': {'fnames': [tif_file]}})

    with set_output_location(expected_file):
        mcorr_obj = MotionCorrect(tif_file, **params.motion, dview=cma.cluster.dview)
        if use_apply:
            mcorr_obj.motion_correct(save_movie=False)
            expected_file = apply_mcorr_to_file(mcorr_obj, tif_file)
        else:
            mcorr_obj.motion_correct(save_movie=True)

    # extract shifts
    shifts_rig = np.array(mcorr_obj.shifts_rig).T  # transpose to dims x frames
    if params.motion["pw_rigid"] == True:
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
    
    np.savez(expected_info_file, shifts_rig=shifts_rig, border_to_0=mcorr_obj.border_to_0,
             shifts_els=np.array(shifts_els), border_asym=np.array(border_asym))

    return expected_file, shifts_rig, int(mcorr_obj.border_to_0),  border_asym, shifts_els, True


def apply_mcorr_to_file(mcorr_obj: MotionCorrect, input_file: str) -> str:
    """Apply shifts from a MotionCorrect object to the given input file (returns output filename)"""
    expected_file = _build_motion_correct_filename(input_file, mcorr_obj.pw_rigid)
    with set_output_location(expected_file):
        base_name_temp = fn_relocated('MC')
        saved_file = mcorr_obj.apply_shifts_movie(
            input_file, save_memmap=True, save_base_name=base_name_temp, remove_min=False)
    assert isinstance(saved_file, str), 'path returned when save_memmap is true'
    if os.path.exists(expected_file):
        os.remove(expected_file)
    shutil.move(saved_file, expected_file)
    return expected_file


def make_highpass_filter(cutoff: float, sample_rate: float, dtype: Optional[DTypeLike] = None
                         ) -> tuple[tuple[np.ndarray, np.ndarray], int]:
    """
    Make high-pass filter for filtering movie across time.
        - cutoff: cutoff frequency in Hz
        - sample_rate: sample rate of movie in Hz
    Returns ((b, a), npad): coefficients in tf format and the number of samples to pad/trim the input.
    """
    cutoff_lam_samps = sample_rate / cutoff  # samples per cycle at cutoff frequency
    npad = round(cutoff_lam_samps)
    n_taps = 2 * npad + 1
    b = signal.firwin(n_taps, cutoff, pass_zero=False, fs=sample_rate)
    a = np.array([1.])
    if dtype is not None:
        b = b.astype(dtype)
        a = a.astype(dtype)
    return (b, a), npad


def filter_worker(args: tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallel kernel for high-pass filtering.
    Inputs:
        - filter_coeffs: (b, a) coefficients in tf format
        - Yr_in: pixels x time chunk of data
        - zi: initial conditions
    
    Outputs:
        - Yr_out: pixels x time filtered data
        - zf: final conditions
    """
    filter_coeffs, Yr, z = args
    Yr, z = signal.lfilter(*filter_coeffs, Yr, axis=1, zi=z)
    return Yr, z


def transpose_flatten_mc_mmap(mmap_files: list[str], mc_border: int, sample_rate: float, force: bool = False,
                              do_transpose=True, blur_kernel_size=1, highpass_cutoff=0.) -> str:
    """
    Saves motion-corrected data, flattened from 3D to 2D and transposed to iterate over time last.
    Note: I am breaking the usual rule in software that each function should do one thing for space and time efficiency.
    Since this involves iterating over and re-saving the entire post-motion-correction movie, it is the best time
        to do any other operations on the movie that work better on chunks of frames than patches of pixels.
    These also change the name of the output, so that extra data isn't normally saved, but if multiple versions of the
        transpose operation are run, they will be saved individually.

    These operations are:
        - Gaussian blur, enabled by setting blur_kernel_size > 1.
        - High-pass filtering, enabled by setting highpass_cutoff (in Hz) > 0.
    """
    expected_file = _to_transposed_flattened_mmap_name(
        mmap_files, do_transpose=do_transpose, blur_kernel_size=blur_kernel_size, highpass_cutoff=highpass_cutoff)

    if not force and os.path.exists(expected_file):
        logging.info('Using existing transposed file: ' + expected_file)
    else:
        logging.info(f'Saving transposed memmap to {os.path.basename(expected_file)}')
        logging.info(f'Using border of {mc_border} pixels')
        if blur_kernel_size != 1:
            logging.info(f'Using Gaussian blur of size {blur_kernel_size}')

        # do in chunks to avoid running out of memory
        dims, T = get_file_size(mmap_files[0])
        assert isinstance(T, int), 'get_file_size with 1 input should return int for T'
        synchronous_limit = psutil.virtual_memory()[1]
        chunks_serial = int(np.ceil(np.prod(dims) / synchronous_limit * T * 8))  # order of ops is important to avoid overflow!!
        chunks_parallel = chunks_serial * cma.cluster.ncores

        if len(mmap_files) == 1 and do_transpose and blur_kernel_size == 1 and highpass_cutoff == 0:  # (only supports C order)
            base_name = os.path.basename(expected_file)
            base_name = base_name[:base_name.index('_d1')]

            n_chunks = chunks_parallel
            while True:
                try:
                    saved_name = cm.save_memmap_join(mmap_files, base_name=base_name, save_npz=False, border_to_0=mc_border,
                                                     n_chunks=n_chunks, dview=cma.cluster.dview)
                    break
                except MemoryError:
                    logging.info('Doubling number of chunks and trying again')
                    n_chunks *= 2

            shutil.move(saved_name, expected_file)  # necessary to flatten - shape interpretation becomes (d1, d2*d3, 1) based on filename
        else:
            # manually concatenate along non-time axis while saving

            # make filter if filtering
            if highpass_cutoff != 0:
                filter_coeffs, delay = make_highpass_filter(highpass_cutoff, sample_rate, dtype=np.float32)
            else:
                filter_coeffs = None
                delay = 0

            # create file to save to
            rows_per_file = int(np.prod(dims))
            rows_total = len(mmap_files) * rows_per_file
            mmap_out = np.memmap(expected_file, dtype=np.float32, mode='w+', shape=(rows_total, T),
                                 order=('C' if do_transpose else 'F'))

            try:
                # make chunks of frames
                chunk_boundaries = np.unique(np.linspace(0, T, chunks_serial + 1, dtype=int))
                chunks = [slice(b1, b2) for b1, b2 in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

                # for filter parallel processing
                pixel_chunk_boundaries = np.unique(np.linspace(0, rows_per_file, cma.cluster.ncores, dtype=int))
                pixel_chunks = [slice(b1, b2) for b1, b2 in zip(pixel_chunk_boundaries[:-1], pixel_chunk_boundaries[1:])]

                for idx, mmap_file in tqdm(enumerate(mmap_files), total=len(mmap_files), unit='plane'):
                    Yr, dims, _ = cm.load_memmap(mmap_file)
                    mmap_out_plane = mmap_out[(rows_per_file*idx):(rows_per_file*(idx+1))]

                    if filter_coeffs is not None:
                        # prepare for filtering
                        zi_base = signal.lfilter_zi(*filter_coeffs)
                        zis = [zi_base[np.newaxis, :] * Yr[pixel_chunk, [0]] for pixel_chunk in pixel_chunks]
                        chunks.append(slice(T, T + delay))
                    else:
                        zis = None

                    for chunk in tqdm(chunks, unit='chunk', leave=False):
                        if chunk.stop > T:
                            # zeros for last bit of filtering
                            assert filter_coeffs is not None and chunk.start == T, f'Unexpected chunk out of bounds (T = {T}, chunk = {(chunk.start, chunk.stop)})'
                            Yr_chunk = np.zeros((Yr.shape[0], delay), dtype=Yr.dtype)
                        else:
                            Yr_chunk = Yr[:, chunk].copy(order='K') # load chunk into memory (incl. transfer from NAS) all at once

                        if filter_coeffs is not None:
                            assert zis is not None
                            # filter
                            dview = cma.cluster.dview
                            if dview is None:
                                assert len(pixel_chunks) == 1, 'Multiple pixel chunks but no dview?'
                                Yr_chunk, zis[0] = signal.lfilter(*filter_coeffs, Yr_chunk, axis=1, zi=zis[0])
                            else:
                                arg_list = [(filter_coeffs, Yr_chunk[pixel_chunk, :], zi) for pixel_chunk, zi in zip(pixel_chunks, zis)]
                                Yr_chunk = np.empty_like(Yr_chunk)  # allocate output buffer
                                pbar = tqdm(range(len(pixel_chunks)), desc='High-pass filtering', unit='chunk', leave=False)
                                map_fn = dview.imap if 'multiprocessing' in str(type(dview)) else dview.map_async
                                for i, res in zip(pbar, map_fn(filter_worker, arg_list)):
                                    Yr_chunk[pixel_chunks[i], :], zis[i] = res
                                pbar.close()
                            
                            # account for delay when writing
                            write_slice = slice(chunk.start - delay, chunk.stop - delay)
                            if write_slice.stop <= 0:
                                # discard the whole chunk
                                continue

                            if write_slice.start < 0:
                                # discard up to the sample corresponding to 0 in the output
                                Yr_chunk = Yr_chunk[:, -write_slice.start:]
                                write_slice = slice(0, write_slice.stop)
                        else:
                            write_slice = chunk

                        if mc_border == 0 and blur_kernel_size == 1:
                            # don't have to reshape
                            mmap_out_plane[:, write_slice] = Yr_chunk
                        else:
                            border = BorderSpec.equal(mc_border)
                            center_slices = border.slices(dims)
                            rows, cols = dims                            
                            chunk_out_3d = mmap_out_plane[:, write_slice].reshape((rows, cols, -1), order='F')
                            chunk_out_3d[:mc_border] = 0
                            chunk_out_3d[rows-mc_border:] = 0
                            chunk_out_3d[:, :mc_border] = 0
                            chunk_out_3d[:, cols-mc_border:] = 0

                            chunk_in_3d = Yr_chunk.reshape((rows, cols, -1), order='F')
                            center_3d = chunk_in_3d[center_slices]

                            if blur_kernel_size == 1:
                                chunk_out_3d[center_slices] = center_3d
                            else:
                                center_out = np.empty_like(center_3d)
                                for k_frame in tqdm(range(chunk_out_3d.shape[2]), unit='frame', leave=False):
                                    center_out[..., k_frame] = cv2.GaussianBlur(
                                        center_3d[..., k_frame], ksize=(blur_kernel_size, blur_kernel_size),
                                        sigmaX=blur_kernel_size//4, sigmaY=blur_kernel_size//4, borderType=cv2.BORDER_REPLICATE)
                                chunk_out_3d[center_slices] = center_out
                mmap_out.flush()
            except:
                # write failed, cleanup
                del mmap_out
                if os.path.exists(expected_file):
                    os.remove(expected_file)
                raise
    return expected_file
