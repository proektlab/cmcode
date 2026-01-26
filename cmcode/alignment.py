from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
import logging
import math
import os
import pickle
from typing import Optional, Sequence, Literal, Union, cast

import caiman as cm
from caiman.base.rois import com, distance_masks, find_matches
from caiman.motion_correction import (
    MotionCorrect, register_translation_3d, tile_and_correct, get_patch_centers, interpolate_shifts)
from caiman.paths import fn_relocated
from caiman.utils import sbx_utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas._libs.missing import NAType
from scipy import sparse, ndimage, signal, optimize

from cmcode import caiman_analysis as cma
from cmcode.cmcustom import compute_matching_performance
from cmcode.util import footprints
from cmcode.util.sbx_data import average_raw_frames, find_sess_sbx_files, get_trial_numbers_from_files
from cmcode.util.image import BorderSpec, invert_mapping, remap_points, remap_points_from_df
from cmcode.util.naming import make_sess_name, make_sess_names
from cmcode.util.paths import get_root_data_dir, get_processed_dir, make_timestamped_filename, get_latest_timestamped_file
from cmcode.util.scaled import ScaledDataFrame, ScaledSeries, make_um_df, make_pixel_df
from cmcode.util.types import NoMultisessionResults, MaybeSparse, BadFitError

# TODO can move to util.sbx_data if needed in other files
@dataclass
class SbxShape:
    n_chan: int
    n_x: int
    n_y: int
    n_planes: int
    n_frames: int

    @classmethod
    def from_file(cls, filename: str, info: Optional[dict] = None):
        """Get shape from a sbx file directly"""
        return cls(*sbx_utils.sbx_shape(filename, info))
    

def is_bidi(sbx_filename: Optional[str] = None, info: Optional[dict] = None) -> bool:
    """Simple helper to determine whether a recording is bidirectional"""
    if info is None:
        if sbx_filename is None:
            raise RuntimeError('Must provide either filename or info object')
        file, ext = os.path.splitext(sbx_filename)
        if ext == '.sbx':
            sbx_filename = file
        
        if ext != '.mat':
            sbx_filename = sbx_filename + '.mat'
        
        info = sbx_utils.loadmat_sbx(sbx_filename)
    return info['scanmode'] == 0
    

def get_stack_file(mouse_id: Union[int, str], sess_id: int, run_id: Optional[int] = None, stack_rec_type: str = 'dlx_calibration') -> str:
    """Find sbx file with stack (defaulting to last run/trial)"""
    if run_id is not None:
        found_files = find_sess_sbx_files(mouse_id=mouse_id, sess_id=sess_id, trials_to_include=[run_id],
                                          rec_type=stack_rec_type, remove_ext=True)
    else:
        # find all files and take the one with highest trial number
        all_files = find_sess_sbx_files(mouse_id=mouse_id, sess_id=sess_id, rec_type=stack_rec_type, remove_ext=True)
        trial_numbers, has_number = get_trial_numbers_from_files(all_files)
        all_files = [f for f, valid in zip(all_files, has_number) if valid]
        all_files_sorted = [f for _, f in sorted(zip(trial_numbers, all_files))]
        found_files = all_files_sorted[-1:]
    
    if len(found_files) == 0:
        raise RuntimeError('Stack file not found')
    found_file = found_files[0]
    return found_file


def load_zstack(filename: str, plane: Union[int, slice, None] = None, channel=1, average=True, skip_odd=False, knobby_frames_to_discard=2,
                info: Optional[dict] = None, shape: Optional[SbxShape] = None, odd_row_offset=0, crop_dead=True
                ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        """
        Returns the data for a single plane along with um per pixel in x, y, and z and relative z-positions
        Shape of result is [T x ]Y x X [x Z_planes] x Z, where z is defined by the z-stack rather than etl planes.
        
        Args:
            filename (str):
                file to load, without .sbx suffix
            plane (int | slice | None):
                which plane(s) to load, or load all planes along 2nd-to-last dimension if None
            channel (int):
                which channel to use (0-based)
            average (bool):
                whether to average over time (otherwise time is first dimension)
            skip_odd (bool):
                whether to only use even rows (in Y) - um_per_pixel is adjusted accordingly
            knobby_frames_to_discard (int):
                how many frames after each knobby movement to discard due to delay/ringing
            info (dict | None):
                can be passed in to save time, but will be inferred from the file metadata if not.
            shape (SbxShape | None):
                can be passed in to save time, but will be inferred from the file metadata if not.
            odd_row_offset (int):
                how many pixels odd rows are shifted relative to even rows (if correction is needed)
            crop_dead (bool):
                whether to crop out dead pixels in odd rows (for bidirectional recordings). only takes
                effect if skip_odd is false.
        """
        if info is None:
            info = sbx_utils.loadmat_sbx(filename + '.mat')
        
        if shape is None:
            shape = SbxShape.from_file(filename, info)

        # figure out z-stack indices
        knobby_table = info['config']['knobby']['schedule']
        z_diff = knobby_table[:, 2]
        if not all(z_diff[1:] == z_diff[1]):
            raise RuntimeError(f'File {filename} seems to contain a non-uniform stack')
        z_pos = np.cumsum(z_diff)
        
        discard_per_plane = math.ceil(knobby_frames_to_discard / shape.n_planes)
        step_starts = list(knobby_table[:, 4] // shape.n_planes + discard_per_plane)
        step_ends = list(knobby_table[1:, 4] // shape.n_planes) + [shape.n_frames]

        if any(end_past_nframes := [end > shape.n_frames for end in step_ends]):
            first_step_to_discard = np.flatnonzero(end_past_nframes)[0]
            logging.warning(f'Number of frames does not match knobby table; using only first {first_step_to_discard} steps')
            step_starts = step_starts[:first_step_to_discard]
            step_ends = step_ends[:first_step_to_discard]
            z_pos = z_pos[:first_step_to_discard]

        if skip_odd:
            subinds_y = slice(0, None, 2) 
            crop_dead = False  # no point in cropping if we're skipping odd rows anyway
        else:
            subinds_y = slice(None)  

        subinds_z = slice(None)
        if plane is not None and not isinstance(plane, int):
            # put plane selection into subindices instead
            subinds_z = plane
            plane = None

        subinds_spatial = (subinds_y, slice(None), subinds_z)
        um_per_pixel = {'x': info['dxcal'], 'y': info['dycal'] * 2 if skip_odd else info['dycal'], 'z': float(abs(z_diff[1]))}

        if average:
            stack_planes = [
                average_raw_frames([filename], frames=slice(start, end), channel=channel, subinds_spatial=subinds_spatial,
                                   crop_dead=crop_dead, plane=plane, to32=True, quiet=True, odd_row_offset=odd_row_offset)
                for start, end in zip(step_starts, step_ends)
            ]
        else:
            stack_planes = [
                sbx_utils.sbxread(filename, subindices=(slice(start, end),) + subinds_spatial, channel=channel,
                                  plane=plane, odd_row_ndead=0, odd_row_offset=odd_row_offset, interp=False,
                                  dview=cma.cluster.dview, quiet=True, to32=True)
                for start, end in zip(step_starts, step_ends)
            ]
        stack = np.stack(stack_planes, axis=-1)
        return stack, um_per_pixel, z_pos


def load_n_zstacks(stack_files: Sequence[str], planes: Union[int, Sequence[int]], channel=1, average=True, skip_odd=False,
                   knobby_frames_to_discard=2, center_and_norm=False, infos: Optional[Sequence[Optional[dict]]] = None,
                   shapes: Optional[Sequence[Optional[SbxShape]]] = None) -> tuple[list[np.ndarray], dict[str, float], np.ndarray]:
    """
    Load each z-stack in the list stack_files (optionally averaging across time).
    Shapes may be truncated in the X and Z dimensions in order to ensure all outputs are the same shape
    (except possibly different numbers of frames if average is false). 
    If different numbers of z positions were captured, the expected z-offsets of each stack to the first one
    (if stacks are otherwise aligned) are given in the last output (otherwise this is typically all zeros).

    See load_plane_zstack for inputs.

    Outputs: 
        - list of loaded stacks, each [T x ]Y x X x Z
        - pixel size, which should be the same for each stack
        - array of accounted-for z offsets (see above) 
    """
    if isinstance(planes, int):
        planes = [planes] * len(stack_files)
    elif len(planes) != len(stack_files):
        raise ValueError('Number of given planes does not match number of files')

    if infos is None:
        infos = [None] * len(stack_files)
    
    if shapes is None:
        shapes = [None] * len(stack_files)

    # first load all the stacks without truncating
    zstack_infos = [
        load_zstack(file, plane=plane, channel=channel, average=average, skip_odd=skip_odd,
                          knobby_frames_to_discard=knobby_frames_to_discard, info=info, shape=shape)
        for file, plane, info, shape in zip(stack_files, planes, infos, shapes)
    ]

    zstacks = [info[0] for info in zstack_infos]
    pixel_sizes = [info[1] for info in zstack_infos]
    z_positions = [info[2] for info in zstack_infos]

    if not all(eq := [sz == pixel_sizes[0] for sz in pixel_sizes[1:]]):
        raise RuntimeError('Cannot compare stacks from the following files due to mismatched pixel size:\n' + 
                           '  - ' + stack_files[0] + '\n  - ' + stack_files[eq.index(False) + 1])
    pixel_size = pixel_sizes[0]
    
    # trim stacks if necessary
    if not average:
        # shift time dimension to end temporarily
        zstacks = [zstack.transpose(*range(1, zstack.ndim), 0) for zstack in zstacks]
    n_planes_to_use = min(len(zpos) for zpos in z_positions)
    n_x_to_use = min(zstack.shape[1] for zstack in zstacks)  # in case different numbers of dead pixels were removed
    # remove extra planes from bottom b/c a mismatch is probably due to not capturing enough frames
    zstacks = [zstack[:, -n_x_to_use:, :n_planes_to_use] for zstack in zstacks]
    z_positions = np.stack([pos[:n_planes_to_use] for pos in z_positions])
    accounted_for_z_offset = z_positions[:, 0] - z_positions[0, 0]  # just find offset from top plane

    if not average:
        # shift time dimension back to front
        zstacks = [zstack.transpose(-1, *range(0, zstack.ndim-1)) for zstack in zstacks]
    
    if center_and_norm:
        zstacks = [zstack - np.min(zstack) for zstack in zstacks]
        zstacks = [zstack / np.max(zstack) for zstack in zstacks]
    
    return zstacks, pixel_size, accounted_for_z_offset


def fix_offset(offset: ScaledDataFrame, skip_odd_y: bool, accounted_for_z_offset: Union[float, np.ndarray]) -> ScaledDataFrame:
    """
    Correct an inferred session-to-session offset to take two things into account:
    - whether odd rows were skipped due to bidirectional scanning
    - z offset that is already accounted for due to comparing planes at different relative depths within the stacks
      (i.e. that is not reflected in the actual functional recordings)
    pixel_size should be what is returned from load_plane_zstack (i.e. with 2x value for y if skip_odd_y is true)
    """
    offset = offset.copy(deep=True)

    if np.any(accounted_for_z_offset != 0):
        z_orig = cast(ScaledSeries, offset.z)
        z_unit = z_orig.unit
        if z_unit is None:
            raise RuntimeError('Z has heterogneous units - cannot correct')

        offset.z = (z_orig.to_um() - accounted_for_z_offset).to_unit(z_unit)

    if skip_odd_y:
        # in original movie pixel density is 2x along y axis
        y_orig = cast(ScaledSeries, offset.y)
        if (umpp_orig := y_orig.um_per_pixel) is None or y_orig.unit is None:
            raise RuntimeError('Cannot correct y pixel size - no pixel size provided')

        y_um = y_orig.to_um()
        y_um.um_per_pixel = umpp_orig / 2
        offset.y = y_um.to_unit(y_orig.unit)
    return offset


def stack_3d_mcorr(mouse_id: int, sessions: list[int], plane: int, stack_rec_type='dlx_calibration', channel=1,
                   runs: Optional[Sequence[Optional[int]]] = None, max_shift_xy=30, unit: Literal['um', 'pixels'] = 'um'
                   ) -> tuple[MotionCorrect, list[int], ScaledDataFrame]:
    """
    Use CaImAn's 3D motion correction to infer offsets between stacks from different sessions.
    Returns mcorr, frames_per_session, offsets:
    - mcorr is the MotionCorrection object to potentially use for further inspection
    - frames_per_session = number of frames in each session's stack
    - offsets is a ScaledDataFrame of average offsets for each session (should be centered around 0).
    """
    if runs is None:
        runs = [None] * len(sessions)

    stack_files = [get_stack_file(mouse_id, sess, run, stack_rec_type) for sess, run in zip(sessions, runs)]

    # skip odd rows if any of them are bidirectional
    infos = [sbx_utils.loadmat_sbx(stack_file + '.mat') for stack_file in stack_files]
    skip_odd = any(info['scanmode'] == 0 for info in infos)

    # load the stack from each session, without averaging
    zstacks, pixel_size, accounted_for_z_offset = load_n_zstacks(
        stack_files, planes=plane, channel=channel, average=False, skip_odd=skip_odd, infos=infos
    )

    # record the frames we have from each session, then concatenate in time
    frames_per_session = [len(zstack) for zstack in zstacks]
    zstack_cat = np.concatenate(zstacks, axis=0)
    # hacky workaround so that file will save correctly
    zstack_cat = cm.movie(zstack_cat, file_name=fn_relocated('tmp_mov_mot_corr.hdf5'))

    # do 3D motion correction
    n_planes = zstack_cat.shape[3]
    max_shifts = (max_shift_xy, max_shift_xy, n_planes // 2 + 1)
    mcorr = MotionCorrect(zstack_cat, max_shifts=max_shifts, pw_rigid=False, is3D=True,
                          indices=(slice(None), slice(None), slice(None)))
    mcorr.motion_correct(save_movie=False)

    # get average offsets for each session
    shifts = mcorr.shifts_rig  # frames x dims
    offsets = make_pixel_df(np.zeros((len(sessions), 3)), dim_names=['y', 'x', 'z'], pixel_size=pixel_size, index=sessions)
    frames_read = 0

    for sess_id, sess_frames in zip(sessions, frames_per_session):
        session_shifts = shifts[frames_read:frames_read+sess_frames]
        frames_read += sess_frames
        mean_session_shifts = np.mean(session_shifts, axis=0)
        
        # update 11/30, the shifts are actually in Y, X, Z order
        offsets.loc[sess_id, ['y', 'x', 'z']] = -mean_session_shifts

    # fix offsets for skip_odd and accounted_for_z_offset
    offsets_fixed = fix_offset(offsets, skip_odd_y = skip_odd, accounted_for_z_offset=accounted_for_z_offset)
    offsets_unit = offsets_fixed.to_unit(unit)
    return mcorr, frames_per_session, offsets_unit


def align_sessions_from_stacks(mouse_id: int, sess_base: int, sess_align: int, stack_rec_type='dlx_calibration',
                               channel=1, run_base: Optional[int] = None, run_align: Optional[int] = None,
                               planes: Optional[Sequence[Union[int, tuple[int, int]]]] = None, unit: Literal['um', 'pixels'] = 'um',
                               knobby_frames_to_discard=2, max_shift_xy=30) -> ScaledDataFrame:
    """
    Find the offsets of planes from one recording session from those of another recording session based
    on stack recordings (by default the recordings with the highest run (trial) numbers under the "dlx_calibration" folder).
    The output is a ScaledDataFrame where each row has the (x,y,z) offset for the corresponding planes.
    By default, the recordings are assumed to have the same number of planes and they are
    matched 1-to-1, but this can be changed by supplying a different sequence of (base, align) plane indices
    to use in the planes argument.
    """
    # Step 1: find files    
    stack_file_base = get_stack_file(mouse_id, sess_base, run_base, stack_rec_type)
    stack_file_align = get_stack_file(mouse_id, sess_align, run_align, stack_rec_type)

    # Step 2: get metadata and figure out what planes to align
    info_base = sbx_utils.loadmat_sbx(stack_file_base + '.mat')
    info_align = sbx_utils.loadmat_sbx(stack_file_align + '.mat')
    # if either file is bidirectional, skip odd lines
    skip_odd = any(is_bidi(info=info) for info in (info_base, info_align))

    shape_base = SbxShape.from_file(stack_file_base, info_base)
    shape_align = SbxShape.from_file(stack_file_align, info_align)

    if planes is None:
        if shape_base.n_planes != shape_align.n_planes:
            raise RuntimeError('Stacks have different number of planes; cannot match automatically - provide "planes" argument')
        planes = [(p, p) for p in range(shape_base.n_planes)]
    else:
        planes = [(p, p) if isinstance(p, int) else p for p in planes]
        for plane_base, plane_align in planes:
            if plane_base >= shape_base.n_planes:
                raise RuntimeError(f'Plane {plane_base} does not exist in base stack with {shape_base.n_planes} planes')
            if plane_align >= shape_base.n_planes:
                raise RuntimeError(f'Plane {plane_align} does not exist in stack to align with {shape_align.n_planes} planes')

    # Step 3: load and separate z-stacks
    plane_offsets: list[ScaledDataFrame] = []
    for plane_pair in planes:
        z_stacks, um_per_pixel, z_offsets = load_n_zstacks(
            [stack_file_base, stack_file_align], plane_pair, average=True,
            channel=channel, skip_odd=skip_odd, knobby_frames_to_discard=knobby_frames_to_discard,
            center_and_norm=True, infos=[info_base, info_align], shapes=[shape_base, shape_align]
        ) 
        n_planes = z_stacks[0].shape[2]
        accounted_for_z_offset = z_offsets[1]

        # OK now finally do the registration
        # reverse inputs for this function so we get the aligned stack relative to the reference
        # update 11/30/24: confirmed this order is correct
        shifts = register_translation_3d(z_stacks[1], z_stacks[0], upsample_factor=10,
                                         max_shifts=(max_shift_xy, max_shift_xy, n_planes // 2 + 1))[0]

        shifts_pix = make_pixel_df(shifts, dim_names=['y', 'x', 'z'], pixel_size=um_per_pixel)
        shifts = fix_offset(shifts_pix, skip_odd_y=skip_odd, accounted_for_z_offset=accounted_for_z_offset).to_unit(unit)
        plane_offsets.append(shifts)

    offset_df = cast(ScaledDataFrame, pd.concat(plane_offsets, axis=0))
    offset_df.index = pd.MultiIndex.from_tuples(planes) # can index by (p0, p1) tuples
    return offset_df


def get_offset_of_stack_from_ref(mouse_id: Union[int, str], ref_sess: int, stack_sess: int,
                                 ref_trial=0, stack_trial: Optional[int] = None, ref_rec_type='learning_ppc_dlx',
                                 stack_rec_type='dlx_calibration', top_n_planes: Optional[int] = None,
                                 knobby_frames_to_discard=2,  plot=False, xy_offset: Optional[ScaledDataFrame] = None,
                                 bad_fit_behavior: Literal['warning', 'error'] = 'warning', allow_stack_as_backup=True
                                 ) -> ScaledDataFrame:
    """
    Estimate X/Y/Z offset in um of the center of a Dlx stack recording from a reference Dlx recording
    (i.e., returns the inverse of the MATLAB function align_stack_to_ref, also not adjusted for Knobby 90deg rotation).
    If xy_offset is provided, the X/Y estimation step is skipped and these offsets are returned back.
    If bad_fit_behavior is 'error', raises a BadFitError instead of warning and returning the best guess when goodness of fit is bad.
    """
    # Load the data
    stack_file = get_stack_file(mouse_id=mouse_id, sess_id=stack_sess, run_id=stack_trial, stack_rec_type=stack_rec_type)
    info_stack = sbx_utils.loadmat_sbx(stack_file)

    # get Y x X x planes reference
    ref_files = find_sess_sbx_files(mouse_id=mouse_id, sess_id=ref_sess, trials_to_include=[ref_trial],
                                    rec_type=ref_rec_type, remove_ext=True)
    if len(ref_files) == 1:
        ref_file = ref_files[0]
        skip_odd = is_bidi(sbx_filename=ref_file) or is_bidi(info=info_stack)
        subinds_y = slice(0, None, 2) if skip_odd else slice(None)
        ref = average_raw_frames(
            [ref_file], frames=slice(None), channel=1, subinds_spatial=(subinds_y, slice(None), slice(0, top_n_planes)),
            crop_dead=False, to32=True, quiet=True)

    elif not allow_stack_as_backup:
        raise RuntimeError('Reference file not found')
    else:
        # try to use stack instead
        logging.warning('Reference file not found; using stack at z = 0 instead')
        if ref_sess == stack_sess:
            # just return zeroes
            _, um_per_pixel, _ = load_zstack(stack_file, skip_odd=False, crop_dead=False, plane=0, average=True,
                                             info=info_stack)
            return make_um_df({'x': 0., 'y': 0., 'z': 0.}, pixel_size=um_per_pixel)
        
        ref_file = get_stack_file(mouse_id=mouse_id, sess_id=ref_sess, stack_rec_type=stack_rec_type)
        skip_odd = is_bidi(sbx_filename=ref_file) or is_bidi(info=info_stack)
        # Y x X x planes x Z stack
        stack_ref, _, z_pos_ref = load_zstack(
            ref_file, skip_odd=skip_odd, crop_dead=False, knobby_frames_to_discard=knobby_frames_to_discard,
            plane=slice(0, top_n_planes), average=True)
        at_0 = z_pos_ref == 0
        if not np.any(at_0):
            raise RuntimeError('Could not use z stack because there is no step at 0!')
        zero_ind = np.flatnonzero(at_0)[0]
        ref = stack_ref[:, :, :, zero_ind]

    # get Y x X x planes x Z stack
    stack, um_per_pixel, z_pos = load_zstack(
        stack_file, skip_odd=skip_odd, crop_dead=False, knobby_frames_to_discard=knobby_frames_to_discard,
        plane=slice(0, top_n_planes), average=True, info=info_stack)
    

    def norm2d(img: np.ndarray) -> np.ndarray:
        img_centered = img - np.mean(img, axis=(0, 1), keepdims=True)
        return img_centered / np.std(img, axis=(0, 1), keepdims=True)

    # Step 1: X/Y offset estimation, if not provided
    if xy_offset is None:
        # cross-correlate top planes of stack and reference
        ref_topplane = norm2d(ref[:, :, [0]])  # keep 3D
        stack_topplane = norm2d(stack[:, :, 0])
        xcorr = signal.correlate(ref_topplane, stack_topplane)
        # get indices of peak correlation
        max_y, max_x, max_z = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        y_shift = max_y - (stack.shape[0] - 1)
        x_shift = max_x - (stack.shape[1] - 1)
        xcorr_best = xcorr[:, :, max_z]
    else:
        xy_offset_pix = xy_offset.to_pixels()
        x_shift = round(xy_offset_pix.at[0, 'x'])
        y_shift = round(xy_offset_pix.at[0, 'y'])
        xcorr_best = max_x = max_y = None
    
    # shift matrices 
    if x_shift >= 0:
        stack = stack[:, :stack.shape[1]-x_shift]  # avoid ":-0"
        ref = ref[:, x_shift:]
    else:
        stack = stack[:, -x_shift:]
        ref = ref[:, :x_shift]
    
    if y_shift >= 0:
        stack = stack[:stack.shape[0]-y_shift]  # avoid ":-0"
        ref = ref[y_shift:]
    else:
        stack = stack[-y_shift:]
        ref = ref[:y_shift]

    # Step 2: Z offset estimation
    # do correlation broadcasting across Z axis, using all planes
    ref_norm = norm2d(ref)
    stack_norm = norm2d(stack)
    corr = np.sum(ref_norm[..., np.newaxis] * stack_norm, axis=(0, 1, 2))

    # fit linear-offset gaussian to correlation across Z
    def gauss_offset_lin(z, loga, center, logsigma, m, b):
        gauss = np.exp(loga - ((z - center) / np.exp(logsigma)) ** 2)
        lin = m * z + b
        return gauss + lin
    
    try:
        popt, _, info = optimize.curve_fit(gauss_offset_lin, xdata=z_pos, ydata=corr, p0=(12, 0, 3, 0, 0),
                                        full_output=True)[:3]
        # compute R^2 as goodness-of-fit measure
        tss = np.sum((corr - np.mean(corr)) ** 2)
        rss = np.sum((info['fvec'] - np.mean(info['fvec'])) ** 2)
        r2 = 1 - rss / tss
        if r2 < 0.7:
            popt = None
    except RuntimeError:
        if bad_fit_behavior == 'warning':
            popt = None  # don't care if optimization failed, assume the fit is maximally bad
        else:
            raise BadFitError('Optimization failed')

    if popt is not None:
        z_shift_um = popt[1]
    else:
        if bad_fit_behavior == 'warning':
            logging.warning('Low goodness of fit; taking best tested point instead of using fit')
            z_shift_um = z_pos[np.argmax(corr)]
        else:
            raise BadFitError('Low goodness of fit')
    
    # convert to um
    # here we reverse x, y, and z shift since we're getting the offset of the stack, not
    # the amount we need to shift the stack. Also no Knobby weirdness.
    x_um = -x_shift * um_per_pixel['x']
    y_um = -y_shift * um_per_pixel['y']
    if skip_odd:
        # change um_per_pixel to be accurate for the original movie
        um_per_pixel['y'] /= 2

    offset = make_um_df({'x': x_um, 'y': y_um, 'z': -z_shift_um}, pixel_size=um_per_pixel) 

    if plot:
        fig = plt.figure()
        if xcorr_best is not None and max_x is not None and max_y is not None:  # plot X/Y cross-correlation in top subplot
            gs = fig.add_gridspec(2, 1)
            xy_ax = fig.add_subplot(gs[0, 0])
            z_ax = fig.add_subplot(gs[0, 0])
            im = xy_ax.imshow(xcorr_best)
            xy_ax.plot(max_x, max_y, 'r.')
            xy_ax.set_title('Cross-correlation of best plane')
            xy_ax.set_xlabel('X (pixels)')
            xy_ax.set_ylabel('Y (pixels)')
            fig.colorbar(im)
        else:
            z_ax = fig.add_subplot()
        
        zaxis = np.arange(z_pos[-1], z_pos[0], 0.1)
        z_ax.plot(z_pos, corr, 'b.', label='data')
        if popt is not None:
            z_ax.plot(zaxis, gauss_offset_lin(zaxis, *popt), 'r', label='fitted curve')
        z_ax.legend()
        z_ax.set_xlabel('Z offset (um)')
        z_ax.set_ylabel('Correlation to reference')
        if top_n_planes is None:
            z_ax.set_title('Corr. of Dlx image to ref as function of z')
        else:
            z_ax.set_title(f'Corr. of top {top_n_planes} to ref as function of z')

        fig.tight_layout()
        plt.show()
    
    return offset


def get_daily_offsets_robust(mouse_id: Union[int, str], sess_ids: Union[Sequence[int], np.ndarray],
                             method: Literal['direct', 'indirect'] = 'indirect', key_session: Optional[Union[int, date]] = None,
                             n_prev_to_match=5, verbose=False, allow_gaps=True, plot=True) -> ScaledDataFrame:
    """
    For a sequence of sessions (should be consecutive or at least monotonic in time), estimate the
    micron offset from each recording to the first through multiple applications of get_offset_of_stack_from_ref.
    
    Direct method:
     - offset-to-stack offset of the key session is computed. By default, this is the first session.
       (key_session has no effect if method != 'direct')
     - Each session's reference (which should be at the same position as the GCaMP recording) is matched to the key
       session's stack.

    Indirect method:
     - The offset of each session's stack from its reference recording is stored
     - Also, each session's reference is matched to N (up to n_prev_to_match) previous sessions' stacks.
       (n_prev_to_match has no effect if method != 'indirect')
     - The previous sessions' ref-to-stack offsets and the previously computed first-ref-to-other-ref offsets
       are used to obtain N first-ref-to-this-ref offset estimates.
     - This session's offset from the first is taken as the median of these estimates, and the cycle continues.
    
    Using verbose=True is encouraged; this prints all N estimates for each session and the median, so that
    you can check that it is a reasonable median and increase n_prev_to_match (or otherwise correct) as necessary.
    Alternatively, use plot=True to see these estimates in a box plot.

    The output will have the recording dates as the index. If allow_gaps is true, no error will be raised if some
    sess_ids have no reference recording. This way, all references from a range of recording sessions can be 
    processed without worrying about days with multiple GCaMP recordings and session IDs but just one reference.
    """
    sess_dates: dict[int, date] = {} # collect dates for each reference recording so they can be used as the index
    # (assumes only one reference recording per day which should be true for us)

    for sess in sess_ids:
        for rec_type in ['learning_ppc_dlx', 'dlx_calibration']:  # use stack as backup for ref
            try:
                sessinfo = cma.SessionAnalysis(mouse_id, sess_id=sess, rec_type=rec_type)
            except RuntimeError as e:
                if e.args[0] == 'No .sbx files found':
                    if allow_gaps or rec_type == 'learning_ppc_dlx':
                        continue
                    else:
                        raise RuntimeError(f'No reference or stack for session {sess} and allow_gaps is False') from e
                else:
                    raise

            scan_day = sessinfo.scan_day
            if scan_day is None:
                raise RuntimeError(f'Could not determine scan day for session {sess}')
            sess_dates[sess] = scan_day
            break
    
    sess_ids = list(sess_dates.keys())

    if method == 'direct':
        if key_session is None:
            key_session = sess_ids[0]
        
        if isinstance(key_session, date):
            key_sess_date = key_session
            key_session = None
            for sess, rec_date in sess_dates.items():
                if rec_date == key_session:
                    key_session = sess
            if key_session is None:
                raise RuntimeError(f'None of the requested sessions match key session date {key_sess_date}')
        else:
            key_sess_date = sess_dates[key_session]
    else:
        key_session = sess_ids[0]
        key_sess_date = sess_dates[key_session]

    # Initialize ref-to-stack and ref-to-ref dataframes
    ref_to_stack_offset_key = get_offset_of_stack_from_ref(mouse_id, ref_sess=key_session, stack_sess=key_session)
    ref_to_stack_offset = make_um_df(np.empty((len(sess_ids), 3)), pixel_size=ref_to_stack_offset_key.um_per_pixel,
                                     index=list(sess_dates.values()))  # pre-allocate dataframe for all sessions
    ref_to_stack_offset.loc[key_sess_date, :] = ref_to_stack_offset_key.loc[0]

    refkey_to_ref_offset = ref_to_stack_offset - ref_to_stack_offset  # initialize as first to first = 0

    # for plotting box-and-whisker:
    all_refkey_to_ref: list[dict] = [
        {'date': key_sess_date, 'aligned_to': key_sess_date, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    ]

    for i, (sess, rec_date) in enumerate(sess_dates.items()):
        if sess == key_session:
            continue

        if method == 'indirect':
            # compare to up to n_prev_to_match previous sessions
            all_prev_sessions = sess_ids[:i]
            prev_sessions = all_prev_sessions[-n_prev_to_match:]
            prev_dates = [sess_dates[other_sess] for other_sess in prev_sessions]

            these_ref0_to_refs = make_um_df(
                np.full((len(prev_sessions), 3), np.nan), pixel_size=refkey_to_ref_offset.um_per_pixel, index=prev_dates)

            # count number actually added to the dataframe, in case we have to skip some
            n_added = 0
            for other_sess in reversed(all_prev_sessions):
                if n_added >= n_prev_to_match:
                    break
                other_date = sess_dates[other_sess]

                # compute cross-session ref-to-stack offset
                try:
                    ref_to_stackprev = get_offset_of_stack_from_ref(
                        mouse_id, ref_sess=sess, stack_sess=other_sess, bad_fit_behavior='error')
                except BadFitError:
                    if verbose:
                        logging.warning(f'Bad fit between session {sess} and {other_sess}; skipping')
                    continue

                ref_to_stackprev.index = pd.Index([other_date])
                refprev_to_ref = ref_to_stack_offset.loc[[other_date], :] - ref_to_stackprev  # this ref from other session ref
                ref0_to_ref = refkey_to_ref_offset.loc[[other_date], :] + refprev_to_ref  # this ref from first session ref
                these_ref0_to_refs.loc[other_date, :] = ref0_to_ref.iloc[0]
                n_added += 1

                if plot:  # also save so we can plot all estimated offsets in the end
                    all_refkey_to_ref.append({
                        'date': rec_date, 'aligned_to': other_date,
                        'x': ref0_to_ref.x.item(), 'y': ref0_to_ref.y.item(), 'z': ref0_to_ref.z.item()
                    })

            # take median of offsets from ref 0
            med_ref0_to_ref = these_ref0_to_refs.median()

            if verbose:
                print(f'Offsets of session {sess} from session {key_session} based on previous {n_added} session(s):')
                print(these_ref0_to_refs)
                print('Median:')
                print(med_ref0_to_ref)

            refkey_to_ref_offset.loc[rec_date, :] = med_ref0_to_ref

            # compute offset of current stack from ref for use in future sessions
            stack_offset = get_offset_of_stack_from_ref(mouse_id, ref_sess=sess, stack_sess=sess)
            ref_to_stack_offset.loc[rec_date, :] = stack_offset.loc[0]
        
        elif method == 'direct':
            # match to key session
            ref_to_stackkey = get_offset_of_stack_from_ref(mouse_id, ref_sess=sess, stack_sess=key_session)
            refkey_to_ref = ref_to_stack_offset.loc[key_sess_date, :] - ref_to_stackkey.loc[0]  # this ref from key ref
            refkey_to_ref_offset.loc[key_sess_date, :] = refkey_to_ref
            if plot:
                all_refkey_to_ref.append({
                    'date': rec_date, 'aligned_to': key_sess_date,
                    'x': refkey_to_ref.x.item(), 'y': refkey_to_ref.y.item(), 'z': refkey_to_ref.z.item()
                })

        else:
            assert False, f'Unrecognized method {method}'
        
    # make relative to first session 
    if key_session != sess_ids[0]:
        ref0_to_ref_offset = refkey_to_ref_offset - refkey_to_ref_offset.loc[sess_dates[sess_ids[0]], :]
    else:
        ref0_to_ref_offset = refkey_to_ref_offset
    
    if plot:
        all_offsets_df = pd.DataFrame(all_refkey_to_ref)
        pd.plotting.boxplot(all_offsets_df, column=['x', 'y', 'z'], by='date', layout=(3, 1))
        
        fig = plt.gcf()
        for label in fig.axes[-1].get_xticklabels(which='major'):
            label.set(rotation=-60, rotation_mode='anchor', horizontalalignment='left', verticalalignment='center')

        for ax in fig.axes:
            ax.set_ylabel('Offset (um)')

        fig.suptitle(f'{mouse_id}: Estimated offsets from {sess_dates[key_session]}')
        plt.show()

    return ref0_to_ref_offset


def save_daily_offsets(mouse_id: Union[int, str], offsets: pd.DataFrame,
                       filename_fmt='{}_daily_offsets.csv', allow_overwrite=False):
    """
    Saves session offsets (e.g., those returned from get_daily_offsets_robust)
    to a csv file under 2p_data/session_alignment, where they fan be further edited by hand if necessary.
    filename_fmt is a format string for the filename under daily_offsets
        that takes mouse_id as the first argument.
    """
    multisession_dir = get_root_data_dir() / 'daily_offsets' 
    csv_file = multisession_dir / filename_fmt.format(mouse_id)
    mode = 'w' if allow_overwrite else 'x'
    offsets.to_csv(csv_file, mode=mode)


def load_daily_offsets(mouse_id: Union[int, str], filename_fmt='{}_daily_offsets.csv') -> pd.DataFrame:
    """
    loads the .csv  that has the x, y and z offsets of each session
    returns a dataframe indexed by recording date with x/y/z offsets in um
    filename_fmt is a format string for the filename under daily_offsets
        that takes mouse_id as the first argument.
    """
    multisession_dir = get_root_data_dir() / 'daily_offsets' 
    csv_file = multisession_dir / filename_fmt.format(mouse_id)
    offset_df = pd.read_csv(csv_file, index_col=0, parse_dates=True, comment='#', usecols=lambda c: c != 'end')
    return offset_df


def load_offsets_for_sessions(mouse_id: Union[int, str], sess_ids: Sequence[int], rec_type='learning_ppc',
                              tags: Union[None, Sequence[Optional[str]]] = None, filename_fmt='{}_daily_offsets.csv'
                              ) -> ScaledDataFrame:
    """
    Load um offsets from a file, select entries corresponding to the given sessions of the given rec_type, and
    convert to a ScaledDataFrame that has the pixel size. Each recording must have the same X/Y pixel size.
    """
    if tags is None:
        tags = [None] * len(sess_ids)

    # make sess_ids unique, since there's actually only one value per session
    sess_ids_uniq, sid_inds = np.unique(sess_ids, return_index=True)
    tags = [tags[i] for i in sid_inds]

    offsets_um_df = load_daily_offsets(mouse_id, filename_fmt)

    # load each SessionAnalysis to figure out how to index the offsets and convert from um to pixels
    rec_dates: list[date] = []
    um_per_pixel = None
    for sess_id, tag in zip(sess_ids_uniq, tags):
        sessinfo = cma.load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, quiet=True)
        this_um_per_pixel = {
            'x': sessinfo.metadata['um_per_pixel_x'],
            'y': sessinfo.metadata['um_per_pixel_y'],
            'z': None}
        if um_per_pixel is None:
            um_per_pixel = this_um_per_pixel
        elif um_per_pixel != this_um_per_pixel:
            raise RuntimeError(f'Pixel resolution mismatch between first session and session {sess_id}')
        
        rec_date = sessinfo.scan_day
        if rec_date is None:
            raise RuntimeError(f'Could not determine scan day for session {sess_id}')
        rec_dates.append(rec_date)
    
    daily_offsets_um = make_um_df(offsets_um_df.loc[rec_dates, :], pixel_size=um_per_pixel, index=sess_ids_uniq)
    # ensure it is relative to the first session
    daily_offsets_um -= daily_offsets_um.iloc[0]
    return daily_offsets_um



# ----------- Main alignment / ROI registration functions ---------- #
# - align_templates: just align 2 recordings based on provided template images
# - register_ROIs: optionally align and then match ROIs between 2 recordings
# - align_templates_multiple: like align_templates but for N >= 2 recordings
# - register_ROIs_multiple: like register_ROIs but for N >= 2 recordings
# - align_templates_multisession: like align_templates_multisession but loading from SessionAnalysis objects
# - register_ROIs_multisession: like register_ROIs_multiple but loading from SessionAnalysis objects


def align_templates(template1: np.ndarray, template2: np.ndarray, use_opt_flow=False,
                    align_options: Optional[dict] = None, n_planes=1, border: Union[BorderSpec, int] = 0,
                    template2_shift_guess: tuple[float, float] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Do just the template alignment step of registerROIs. This is just a single iteration of the
    motion-correction algorithm (or optical flow, if use_opt_flow is True) with some defaults
    that help align images that are not highly similar. Does not require CNMF results.
    The direction is template2 is aligned to fit template1. Return values are x_remap, y_remap
    to be fed into cv2.remap in order to warp session 2 images or ROIs to fit session 1.
    n_planes (positive int): If != 1, separate planes along x axis before aligning.
    border: border to exclude when doing alignment
    template2_shift_guess: (Y, X) amount to shift each plane of template 2 before attempting alignment.
        The resulting remaps will include this shift.
    
    Ouptupt: (x_remap, y_remap) to be used in remap_image, etc.
    """
    dims = template1.shape
    if template2.shape != dims:
        raise ValueError('Templates must have matching dimensions')
    
    if dims[1] % n_planes != 0:
        raise ValueError('Template width is not a multiple of the # of planes')

    if isinstance(border, int):
        border = BorderSpec.equal(border)
    
    x_plane_remaps = []
    y_plane_remaps = []
    planes1 = np.split(template1, n_planes, axis=1)
    planes2 = np.split(template2, n_planes, axis=1)
    for k_plane, (plane1, plane2) in enumerate(zip(planes1, planes2)):
        plane_dims = plane1.shape
        plane_inds = tuple(np.arange(0., dim).astype(np.float32) for dim in plane_dims)

        guess_y, guess_x = template2_shift_guess
        if guess_y != 0 or guess_x != 0:
            # shift plane2 based on shift guess and adjust borders
            plane2 = ndimage.shift(plane2, template2_shift_guess)
            if guess_y > 0:
                border.top += math.ceil(guess_y)
            else:
                border.bottom += math.ceil(-guess_y)
            if guess_x > 0:
                border.left += math.ceil(guess_x)
            else:
                border.right += math.ceil(-guess_x)

        # get just non-blacked out center to compute transform
        center_slices = border.slices(plane_dims)
        center1 = plane1[center_slices].copy()  # necessary to avoid mutating original array
        center2 = plane2[center_slices].copy()
        center_dims = center1.shape
        center_inds = (plane_inds[0][center_slices[0]], plane_inds[1][center_slices[1]])

        # scale center to range from 0 to 1
        center1 -= center1.min()
        center1 /= center1.max()
        center2 -= center2.min()
        center2 /= center2.max()

        if use_opt_flow:
            # don't think this is necessary
            # center1_norm = np.uint8(center1 * (center1 > 0) * 255)
            # center2_norm = np.uint8(center2 * (center2 > 0) * 255)
            flow = np.empty(center1.shape + (2,), dtype=np.float32)
            cv2.calcOpticalFlowFarneback(center1, center2, flow, 0.5, 3, 128, 3, 7, 1.5, 0)
            shifts_x_center = flow[:, :, 0]
            shifts_y_center = flow[:, :, 1]

            # repeat edge values to fill in border
            shifts_x_full = interpolate_shifts(shifts_x_center, center_inds, plane_inds)
            shifts_y_full = interpolate_shifts(shifts_y_center, center_inds, plane_inds)
        else:
            align_defaults = {
                "strides": (16, 16),
                "overlaps": (16, 16),
                "max_shifts": (10, 10),
                "shifts_opencv": True,
                "upsample_factor_grid": 4,
                "shifts_interpolate": True,
                "max_deviation_rigid": 2
                # any other argument to tile_and_correct can also be used in align_options
            }

            if align_options is not None:
                # override defaults with input options
                align_defaults.update(align_options)
            align_options = align_defaults

            shifts = tile_and_correct(center2, center1, **align_options)[1]

            if align_options["max_deviation_rigid"] == 0:
                # repeat rigid shifts to size of the image
                assert isinstance(shifts, tuple)
                shifts_x_center = np.full(center_dims, -shifts[1])
                shifts_y_center = np.full(center_dims, -shifts[0])

                # repeat edge values to fill in border
                shifts_x_full = interpolate_shifts(shifts_x_center, center_inds, plane_inds)
                shifts_y_full = interpolate_shifts(shifts_y_center, center_inds, plane_inds)
            else:
                # piecewise - interpolate from patches to get shifts per pixel
                patch_centers = get_patch_centers(
                    center_dims, overlaps=align_options["overlaps"], strides=align_options["strides"],
                    shifts_opencv=align_options["shifts_opencv"],
                    upsample_factor_grid=align_options["upsample_factor_grid"])
                # account for border
                patch_centers = (np.array(patch_centers[0]) + border.top,
                                 np.array(patch_centers[1]) + border.left)
                patch_grid = tuple(len(centers) for centers in patch_centers)
                _sh_ = np.stack(shifts, axis=0)

                # reshape each shift vector to (Y, X) shape
                shifts_x = np.reshape(_sh_[:, 1], patch_grid, order='C').astype(np.float32)
                shifts_y = np.reshape(_sh_[:, 0], patch_grid, order='C').astype(np.float32)

                # interpolate, also filling in edge values
                shifts_x_full = interpolate_shifts(-shifts_x, patch_centers, plane_inds)
                shifts_y_full = interpolate_shifts(-shifts_y, patch_centers, plane_inds)

        y_grid, x_grid = np.meshgrid(*plane_inds, indexing='ij')
        x_grid += k_plane * plane_dims[1]  # account for the fact that we're on the nth plane
        x_remap = (shifts_x_full + x_grid - guess_x).astype(np.float32)
        y_remap = (shifts_y_full + y_grid - guess_y).astype(np.float32)
        
        x_plane_remaps.append(x_remap)
        y_plane_remaps.append(y_remap)
    
    x_remap = np.concatenate(x_plane_remaps, axis=1)
    y_remap = np.concatenate(y_plane_remaps, axis=1)
    return x_remap, y_remap


def align_templates_multiple(
        templates: Sequence[np.ndarray], borders: Optional[Sequence[Union[BorderSpec, int]]] = None,
        use_opt_flow=False, align_options: Optional[dict] = None, n_planes=1,
        yx_position_guesses: Union[Sequence, np.ndarray, None] = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Align a series of templates in an iterative fashion (1 to 2, 2 to 3, ..., n-1 to n).
    Returns a list of n-1 tuples (x_remap, y_remap) which can be passed on to register_ROIs_multiple.
    yx_position_guesses: relative position of each template in (y, x) image coordinates, used as a hint for alignment.
        Shape should be n_templates x 2.
    """
    n_templates = len(templates)

    if borders is None:
        borders = [0] * n_templates
    
    if yx_position_guesses is None:
        yx_position_guesses = np.zeros((n_templates, 2))
    else:
        yx_position_guesses = np.array(yx_position_guesses)
        if yx_position_guesses.shape[0] != n_templates:
            raise ValueError('Height of position guesses matrix does not match number of templates')
        if yx_position_guesses.shape[1] != 2:
            raise ValueError('Width of position guesses matrix should be 2 (y, x)')
    
    # align each pair of sessions
    xy_remaps: list[tuple[np.ndarray, np.ndarray]] = []

    # always warp the first session, so it should be 'template2'
    for (template2, border2, pos2), (template1, border1, pos1) in pairwise(zip(templates, borders, yx_position_guesses)):
        border = BorderSpec.max(border1, border2)
        # the guess is how to shift template 2, so we want to subtract its current position and add the position of template 1
        template2_shift_guess = (pos1[0] - pos2[0], pos1[1] - pos2[1])
        xy_remaps.append(align_templates(
            template1, template2, use_opt_flow=use_opt_flow, align_options=align_options,
            n_planes=n_planes, border=border, template2_shift_guess=template2_shift_guess))
    return xy_remaps


def align_templates_allpairs(templates: Sequence[np.ndarray], borders: Optional[Sequence[Union[BorderSpec, int]]] = None,
                             use_opt_flow=False, align_options: Optional[dict] = None, n_planes=1,
                             yx_position_guesses: Union[Sequence, np.ndarray, None] = None,
                             precomputed_remaps: Optional[np.ndarray] = None, precomputed_mask: Sequence[bool] = ()) -> np.ndarray:
    """
    Align all pairs of given templates, similar to align_templates_multiple.
    The return value "remaps" is a 5-D NDArray of size (len(templates), len(templates)-1, 2) + templates[0].shape
    (assuming all templates have the same shape).
    remaps[i, j] contains the mapping for i -> j if i > j, i -> j+1 otherwise.
    Each entry remaps[i, j] contains the (x, y) mappings stacked in the first dimension.
    If precomputed_remaps is provided, should be a 5-D NDArray of the remaps between a subset of the given templates;
        these will not be recomputed. precomputed_mask should be provided along with it, indicating which of the templates
        were precomputed.
    """
    n_templates = len(templates)

    if borders is None:
        borders = [0] * n_templates
    
    if precomputed_remaps is not None:
        # validate precomputed remaps
        if len(precomputed_mask) != len(templates):
            raise ValueError('precomputed_mask of same length as templates must be provided along with precomputed_remaps')
        if sum(precomputed_mask) != precomputed_remaps.shape[0]:
            raise ValueError('Total number of precomputed sessions does not match size of precomputed_remaps')
    
    if yx_position_guesses is None:
        yx_position_guesses = np.zeros((n_templates, 2))
    else:
        yx_position_guesses = np.array(yx_position_guesses)
        if yx_position_guesses.shape[0] != n_templates:
            raise ValueError('Height of position guesses matrix does not match number of templates')
        if yx_position_guesses.shape[1] != 2:
            raise ValueError('Width of position guesses matrix should be 2 (y, x)')
    
    remaps = np.empty((n_templates, n_templates - 1, 2) + templates[0].shape, dtype=np.float32)
    # like above, we call the "from" template template2 and the "to" template template1
    for i_from, (template2, border2, pos2) in enumerate(zip(templates, borders, yx_position_guesses)):
        for j_to, template1, border1, pos1 in zip(range(i_from+1, n_templates), templates[i_from+1:], borders[i_from+1:],
                                                  yx_position_guesses[i_from+1:]):
            if precomputed_remaps is not None and precomputed_mask[i_from] and precomputed_mask[j_to]:
                # reuse precomputed remap
                i_precomputed = sum(precomputed_mask[:i_from])
                j_precomputed = sum(precomputed_mask[:j_to])  # j_to > i_from => j_precomputed > i_precomputed
                remaps[i_from, j_to-1] = precomputed_remaps[i_precomputed, j_precomputed-1]
                remaps[j_to, i_from] = precomputed_remaps[j_precomputed, i_precomputed]
            else:
                border = BorderSpec.max(border1, border2)
                # the guess is how to shift template 2, so we want to subtract its current position and add the position of template 1
                template2_shift_guess = (pos1[0] - pos2[0], pos1[1] - pos2[1])
                this_remap = align_templates(
                    template1, template2, use_opt_flow=use_opt_flow, align_options=align_options,
                    n_planes=n_planes, border=border, template2_shift_guess=template2_shift_guess
                )
                remaps[i_from, j_to-1] = this_remap
                remaps[j_to, i_from] = invert_mapping(*this_remap)
            
    return remaps


def load_remaps_allpairs(mapping_path: str, sess_names: Sequence[str]) -> tuple[Sequence[bool], np.ndarray]:
    """
    Load x/y mappings from an npz file, between the sessions specified in sess_names.
    The result is of shape from_session x to_session x (x,y) x source_x x source_y.
    Also, self-mappings are excluded, so mappings[i, j] is i -> j if j < i but i -> j+1 if j >= i.
    Returns (b_found, mappings) where b_found[i] is true if the i'th session was found in the file.
    """
    with np.load(mapping_path) as mapping_f:
        if np.array_equal(mapping_f['sess_names'], sess_names):
            b_found = [True] * len(sess_names)
            xy_remaps = mapping_f['xy_remaps']
        else:
            file_sess_names = list(mapping_f['sess_names'])
            b_found = [name in file_sess_names for name in sess_names]
            file_sess_inds = [file_sess_names.index(name) for name, found in zip(sess_names, b_found) if found]

            # rebuild mapping matrix by selectively copying from file
            xy_remaps = np.empty_like(mapping_f['xy_remaps'],
                                      shape=(len(file_sess_inds), len(file_sess_inds)-1) + mapping_f['xy_remaps'].shape[2:])
            for i, ind1 in enumerate(file_sess_inds):
                for j, ind2 in enumerate(file_sess_inds[:i] + file_sess_inds[i+1:]):  # j = index when excluding self-mapping
                    xy_remaps[i, j, ...] = mapping_f['xy_remaps'][ind1, ind2 if ind2 < ind1 else ind2-1]

    return b_found, xy_remaps


def load_or_compute_remaps_for_sessions(
        mouse_id: Union[int, str], sess_ids: Sequence[int], rec_type='learning_ppc', tags: Union[None, Sequence[Optional[str]]] = None,
        grouptag: Optional[str] = None, use_saved_mappings: Optional[bool] = None,
        save_mappings_with_grouptag: Optional[bool] = None, rigid_offsets: Optional[ScaledDataFrame] = None,
        max_initial_shift=50, max_additional_shift=10, max_deviation_rigid=6,
        projection_params: Optional[Union[str, dict]] = None) -> tuple[np.ndarray, str]:
    """
    Helper to load nonrigid mappings between sessions for the given sessions (based on sess_ids and tags), or compute them if they
    are not saved. If computing mappings, rigid_offsets will be used for the initial guesses if it is not None.
    See register_ROIs_multisession_3D for the other parameters.

    use_saved_mappings is a ternary flag:
        - If False, skips trying to load and just computes the mappings
        - If True, only loads and errors if they are not available
        - If None (default), tries to load, and computes if they are not available.

    Returns (mappings, mappings_file_path)
    """
    if tags is None:
        tags = [None] * len(sess_ids)

    sess_names = make_sess_names(sess_ids, tags)
    processed_dir = get_processed_dir(mouse_id, rec_type=rec_type)
    alignment_dir = os.path.join(processed_dir, 'alignment')

    untagged_fn_pattern = f'{mouse_id}_session_mappings_%dt.npz'
    tagged_fn_pattern = f'{mouse_id}_{grouptag}_session_mappings_%dt.npz' if grouptag else None

    mappings_found = [False] * len(sess_names)
    xy_remaps = None
    loaded_from_untagged = False
    mapping_load_path = None
    if use_saved_mappings == False:
        logging.info('Not trying to load mappings since use_saved_mappings is false')
    elif not os.path.exists(alignment_dir):
        logging.info('Alignment dir not found; creating')
        os.mkdir(alignment_dir)
    else:
        # find latest tagged or untagged saved file
        if tagged_fn_pattern is None or (mapping_load_path := get_latest_timestamped_file(alignment_dir, tagged_fn_pattern)) is None:
            mapping_load_path = get_latest_timestamped_file(alignment_dir, untagged_fn_pattern)
            if mapping_load_path is not None:
                loaded_from_untagged = True

        if mapping_load_path is not None:
            logging.info('Loading saved nonrigid X/Y mappings between every pair of sessions')
            if mapping_load_path.endswith('.npz'):
                mappings_found, xy_remaps = load_remaps_allpairs(mapping_load_path, sess_names)
                if all(mappings_found):
                    return xy_remaps, mapping_load_path

                logging.warning('The following sessions did not have saved mappings: ' +
                                ', '.join([sess for sess, found in zip(sess_names, mappings_found) if not found]))
            else:
                xy_remaps = np.load(mapping_load_path)
                if xy_remaps.shape[0] == len(sess_ids):
                    return xy_remaps, mapping_load_path
                
                logging.warning('Loading saved mappings failed: wrong number of mappings')
                xy_remaps = None
        else:
            logging.warning('No saved mappings found')

    if use_saved_mappings:
        raise RuntimeError('Failed to load mappings and use_saved_mappings is true, so not computing')
    elif use_saved_mappings is None:
        logging.info('Computing mappings for any/all missing sessions')

    logging.info('Loading z-stack for each session')
    templates, borders = get_zmax_templates_and_borders_multisession(
        mouse_id, sess_ids, rec_type=rec_type, tags=tags,
        include_dead_pixel_border=True, projection_params=projection_params)

    if rigid_offsets is not None:
        logging.info('Using saved rigid X/Y offsets')
        yx_pos_df = rigid_offsets.loc[list(sess_ids), :]
    else:
        logging.info('Estimating rigid X/Y offsets')
        yx_pos_df = guess_yx_positions_multiple(templates, borders=borders, max_shift=max_initial_shift)
        
    yx_position_guesses = yx_pos_df.loc[:, ['y', 'x']].to_pixels().to_numpy()

    # compute every pair
    logging.info('Estimating nonrigid X/Y mappings')
    align_options = {'max_shifts': (max_additional_shift, max_additional_shift),
                     'max_deviation_rigid': max_deviation_rigid}
    xy_remaps = align_templates_allpairs(
        templates, borders, align_options=align_options, yx_position_guesses=yx_position_guesses,
        precomputed_remaps=xy_remaps, precomputed_mask=mappings_found)
    
    # save for next time
    logging.info('Saving X/Y mappings')
    if save_mappings_with_grouptag is None:
        # decide whether to overwrite existing untagged file, if any
        if not loaded_from_untagged:
            save_mappings_with_grouptag = False
        else:
            assert mapping_load_path is not None
            # check which sessions are saved in the file and only overwrite if all of the saved sessions are present here
            with np.load(mapping_load_path) as untagged_f:
                save_mappings_with_grouptag = all(saved_sess in sess_names for saved_sess in untagged_f['sess_names'])

    save_file_pattern = tagged_fn_pattern if tagged_fn_pattern and save_mappings_with_grouptag else untagged_fn_pattern
    mapping_save_path = os.path.join(alignment_dir, make_timestamped_filename(save_file_pattern))
    np.savez(mapping_save_path, xy_remaps=xy_remaps, sess_names=np.array(sess_names))
    return xy_remaps, mapping_save_path


def align_templates_multisession(
        mouse_id: int, sess_ids: Sequence[int], rec_type='learning_ppc',
        tags: Union[None, str, Sequence[Optional[str]]] = None,
        align_options: Optional[dict] = None, template_params: Optional[dict] = None,
        yx_position_guesses: Union[Sequence, np.ndarray, None] = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Just align templates (projections) from the given sessions, in an iterative fashion (1 to 2, 2 to 3, ..., n-1 to n).
    Does not require CNMF results. Returns a list of n-1 tuples (x_remap, y_remap) which can be passed on to register_ROIs_multiple.
    template_params defaults to {'type': 'mean', 'norm_medw': 25}.
    yx_position_guesses: relative position of each session in (y, x) image coordinates, used as a hint for alignment.
        Shape should be n_sessions x 2.
    """
    if not isinstance(tags, Sequence):
            tags = [tags] * len(sess_ids)
        
    if template_params is None:
        template_params = {'type': 'mean', 'norm_medw': 25}
    
    if align_options is not None and 'use_opt_flow' in align_options:
        use_opt_flow = align_options.pop('use_opt_flow')
    else:
        use_opt_flow = False

    # make template for each session
    templates: list[np.ndarray] = []
    n_planes = None
    borders: list[int] = []

    for sess_id, tag in zip(sess_ids, tags):
        # load existing analysis object
        info = cma.load_latest(mouse_id=mouse_id, sess_id=sess_id, rec_type=rec_type, tag=tag)
        if n_planes is None:
            n_planes = info.metadata['num_planes']
        elif n_planes != info.metadata['num_planes']:
            raise RuntimeError(f'Mismatch in number of planes in session {sess_id}{tag if tag else ""}')
        
        if info.mc_result is None:
            raise RuntimeError(f'Motion correction not done in session {sess_id}{tag if tag else ""}')
        borders.append(info.mc_result.border_to_0)
        templates.append(info.get_projection_for_seed(**template_params)[0])
    assert n_planes is not None, 'No sessions processed'

    return align_templates_multiple(
        templates, borders, use_opt_flow=use_opt_flow, align_options=align_options, n_planes=n_planes,
        yx_position_guesses=yx_position_guesses)


def guess_yx_positions_multiple(templates: Sequence[np.ndarray], n_planes=1, max_shift=50,
                                borders: Optional[Sequence[Union[BorderSpec, int]]] = None) -> ScaledDataFrame:
    """
    Use rigid correction with generous max shift to get an initial estimate of relative X/Y 
    offsets of a sequence of templates. The output is a len(templates) x 2 matrix of relative
    positions (in pixels) that can be used as an input to align_templates_multiple, etc.
    """
    align_options = {'max_shifts': (max_shift, max_shift), 'max_deviation_rigid': 0}
    xy_remaps = align_templates_multiple(templates, borders, use_opt_flow=False, align_options=align_options,
                                         n_planes=n_planes)

    # collect results into matrix; take advantage of the fact that the remap matrices are uniform except for the offset
    # and the entry at (0, 0) has no offset.
    pos_guesses = [np.array([0., 0.])]
    for x_remap, y_remap in xy_remaps:
        # subtract guessed offset of last one relative to next one from position of last one
        pos_guesses.append(pos_guesses[-1] - np.array([y_remap[0, 0], x_remap[0, 0]]))
    
    return make_pixel_df(pos_guesses, dim_names=['y', 'x'])


@dataclass
class RegisterROIsResults:
    """Return value type for register_ROIs"""
    matched1: list[int]    # ROIs in first set that matched with other set
    unmatched1: list[int]  # ROIs in first set that did not match
    matched2: list[int]
    unmatched2: list[int]
    performance: dict[str, float] 
    A1: sparse.csc_matrix  # First set of ROIs (unchanged)
    A2: sparse.csc_matrix  # Second set of ROIs after mapping to first image (if align_flag==True)
    # in x_remap and y_remap, None indicates that there was no remapping done, while NA means the data is missing.
    x_remap: Union[Optional[np.ndarray], NAType] = pd.NA  # x-values of pixel remap function
    y_remap: Union[Optional[np.ndarray], NAType] = pd.NA  # y-values of pixel remap function
    A2_orig: Optional[sparse.csc_matrix] = None  # Second set of ROIs (unchanged)
    components_used: Optional[np.ndarray]  = None  # which of original components were used for registration


def threshold_masks(A: sparse.csc_matrix, max_thr: float):
    """Modify masks to apply given threshold (fraction of max value in each mask)"""
    remove_rows = np.array([], dtype=np.int32)
    remove_cols = np.array([], dtype=np.int32)
    for k in range(A.shape[1]):
        roi = A[:, [k]].toarray()
        below_thr = np.flatnonzero(np.squeeze(roi) < roi.max() * max_thr)
        remove_rows = np.concatenate((remove_rows, below_thr))
        remove_cols = np.concatenate((remove_cols, np.repeat(k, len(below_thr))))
    remove_data = np.repeat(True, len(remove_rows))
    remove_mask = sparse.coo_matrix((remove_data, (remove_rows, remove_cols)), shape=A.shape)
    A[remove_mask] = 0


def register_ROIs(
        A1: MaybeSparse,
        A2: MaybeSparse,
        dims: tuple[int, int],
        template1: Optional[np.ndarray] = None,
        template2: Optional[np.ndarray] = None,
        align_flag=True,
        xy_remap: Optional[tuple[np.ndarray, np.ndarray]] = None,
        com1: Optional[np.ndarray] = None,
        com2: Optional[np.ndarray] = None,
        D: Optional[list[np.ndarray]] = None,
        D_pow: Optional[np.ndarray] = None,
        max_thr: float = 0,
        use_opt_flow=False,
        thresh_cost=.7,
        max_dist=10.,
        enclosed_thr: Optional[float] = None,
        print_assignment=False,
        align_options: Optional[dict] = None,
        n_planes=1,
        border: Union[BorderSpec, int] = 0,
        template2_shift_guess: tuple[float, float] = (0, 0)) -> RegisterROIsResults:
    """
    See caiman.base.rois.register_ROIs for documentation
    Modified to return both A1 (filtered taking max_thr into account) and A2.
    xy_remap: tuple of (x_remap, y_remap) to map template2 onto template1.
        If provided, skips registration step but uses these to map A2.
    n_planes (positive int): if not equal to 1, separate planes along x axis before doing registration.
    border: border to exclude when doing registration
    D_pow_A2: if specified, raise distances (which are in range [0, 1]) to this power (should broadcast with D)
    """

    A1 = sparse.csc_matrix(A1)
    A2 = sparse.csc_matrix(A2)
    
    if xy_remap is None and align_flag and (template1 is not None and template2 is not None):
        if template1 is None or template2 is None:
            logging.warning('Templates not provided - skipping alignment')
        else:
            # first align ROIs from session 2 to the template from session 1
            xy_remap = align_templates(template1, template2, use_opt_flow=use_opt_flow,
                                        align_options=align_options, n_planes=n_planes, border=border,
                                        template2_shift_guess=template2_shift_guess)

    if xy_remap is not None:
        # Build new A2 by remapping each ROI
        A2_aligned = footprints.map_footprints(A2, xy_remap)
    else:
        A2_aligned = A2

    # apply max_thr
    for A in [A1, A2_aligned]:
        threshold_masks(A, max_thr)

    if D is None:
        if com1 is None:
            com1 = com(A1, *dims)
    
        if com2 is None:
            com2 = com(A2_aligned, *dims)
        elif xy_remap is not None:
            com2 = remap_points(com2, *xy_remap)

        A1_tr = (A1 > 0).astype(float)
        A2_tr = (A2_aligned > 0).astype(float)
        D = distance_masks([A1_tr, A2_tr], [com1, com2], max_dist, enclosed_thr=enclosed_thr)  # type: ignore

    if D_pow is not None:
        D = [d ** D_pow for d in D]

    # find best matches between A1 and A2 by solving linear sum assignment problem
    matches, costs = find_matches(D, print_assignment=print_assignment)
    matches = matches[0]
    costs = costs[0]

    # store indices
    idx_tp = np.flatnonzero(np.array(costs) < thresh_cost)
    matched_ROIs1 = list(matches[0][idx_tp])     # ground truth
    matched_ROIs2 = list(matches[1][idx_tp])     # algorithm - comp
    non_matched1 = list(np.setdiff1d(range(D[0].shape[0]), matches[0][idx_tp]))
    non_matched2 = list(np.setdiff1d(range(D[0].shape[1]), matches[1][idx_tp]))

    # compute precision and recall
    performance = compute_matching_performance(D[0].shape[0], D[0].shape[1], len(idx_tp))

    return RegisterROIsResults(
        matched1=matched_ROIs1,
        matched2=matched_ROIs2,
        unmatched1=non_matched1,
        unmatched2=non_matched2,
        performance=performance,
        A1=A1,
        A2_orig=A2,
        A2=A2_aligned,
        x_remap=xy_remap[0] if xy_remap else None,
        y_remap=xy_remap[1] if xy_remap else None
    )

def register_ROIs_multiple(
        A: Sequence[MaybeSparse],
        dims: tuple[int, int],
        templates: Optional[Sequence[np.ndarray]] = None,
        align_flag=True,
        max_thr: float = 0,
        use_opt_flow=False,
        thresh_cost=.7,
        max_dist=10,
        enclosed_thr=None,
        align_options: Optional[dict] = None,
        xy_remaps: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
        n_planes=1,
        borders: Optional[Sequence[Union[BorderSpec, int]]] = None,
        yx_position_guesses: Union[Sequence, np.ndarray, None] = None
        ) -> tuple[MaybeSparse, np.ndarray, list[np.ndarray], list[dict[str, float]], list[tuple]]:
    """
    Register ROIs across multiple sessions using an intersection over union metric
    and the Hungarian algorithm for optimal matching. Registration occurs by 
    aligning session 1 to session 2, keeping the union of the matched and 
    non-matched components to register with session 3 and so on.

    Args:
        A: list of ndarray or csc_matrix matrices # pixels x # of components
           ROIs from each session

        dims: list or tuple
            dimensionality of the FOV

        templates: list of ndarray matrices of size dims
            templates from each session

        align_flag: bool
            align the templates before matching (false if templates are not provided)

        max_thr: scalar
            max threshold parameter before binarization    

        use_opt_flow: bool
            use dense optical flow to align templates

        thresh_cost: scalar
            maximum distance considered

        max_dist: scalar
            max distance between centroids

        enclosed_thr: float
            if not None set distance to at most the specified value when ground 
            truth is a subset of inferred
        
        align_options: Optional[dict]
            motion correction options to override defaults for alignment in register_ROIs
            (only used if use_opt_flow is false)
        
        xy_remaps: Optional[list[tuple[ndarray, ndarray]]]
            list of pre-computed alignment maps, should be 1 less than the number of sessions, to
            bypass computing alignment
        
        n_planes: int
            number of planes for each session, used to divide images along X axis before registration
        
        borders: Sequence[Union[BorderSpec, int]]
            size of borders to ignore around each plane, across sessions. If a sequence of length 1 is
            passed, it is repeated for all sessions. Used for registration.
        
        yx_position_guesses: relative position of each template in (y, x) image coordinates, used as a hint for alignment.
            Shape should be n_templates x 2

    Returns:
        A_union: csc_matrix # pixels x # of total distinct components
            union of all kept ROIs 

        assignments: ndarray int of size # of total distinct components x # sessions
            element [i,j] = k if component k from session j is mapped to component
            i in the A_union matrix. If there is no much the value is NaN

        matchings: list of lists
            matchings[i][j] = k means that component j from session i is represented
            by component k in A_union

    """
    logger = logging.getLogger("caiman")

    if align_options is not None:
        use_opt_flow = align_options.pop('use_opt_flow', use_opt_flow)

    n_sessions = len(A)
    if n_sessions <= 1:
        raise Exception('number of sessions must be greater than 1')

    if xy_remaps is not None:
        if len(xy_remaps) != n_sessions - 1:
            raise ValueError('xy_remaps should have length 1 less than the # of sessions')
        xy_remaps_in = xy_remaps
        align_flag = False
    else:
        xy_remaps_in = [None] * (n_sessions - 1)

    # Do alignment on sequence if needed/requested
    if align_flag:
        if templates is None:
            logging.warning('Templates not provided - skipping alignment')
            align_flag = False
        elif len(templates) != n_sessions:
            raise ValueError('templates should have length equal to the # of sessions')
        else:
            xy_remaps_in = align_templates_multiple(
                templates, borders, use_opt_flow=use_opt_flow, align_options=align_options, n_planes=n_planes,
                yx_position_guesses=yx_position_guesses)

    # Do ROI matching on each pair
    A_union = deepcopy(A[0])
    matchings: list[np.ndarray] = []
    all_performance: list[dict[str, float]] = []
    xy_remaps_out: list[tuple] = []  # mappings from n-1 to n for n in range(1, n_sessions)
    matchings.append(np.arange(A_union.shape[-1], dtype=int))

    for sess, xy_remap in zip(range(1, n_sessions), xy_remaps_in):
        reg_results = register_ROIs(
            A[sess], A_union, dims, align_flag=False, max_thr=max_thr, thresh_cost=thresh_cost,
            max_dist=max_dist, enclosed_thr=enclosed_thr, xy_remap=xy_remap)

        matched_session = reg_results.matched1
        matched_union = reg_results.matched2
        nonmatch_session = reg_results.unmatched1
        nonmatch_union = reg_results.unmatched2
        performance = reg_results.performance
        A2 = reg_results.A2
        xy_remaps_out.append((reg_results.x_remap, reg_results.y_remap))

        logger.info(len(matched_session))

        all_performance.append(performance)

        A_union = deepcopy(A2)
        A_union[:, matched_union] = A[sess][:, matched_session]
        A_union = sparse.csc_array(sparse.hstack((A_union, A[sess][:, nonmatch_session])))
        new_match = np.zeros(A[sess].shape[-1], dtype=int)
        new_match[matched_session] = matched_union
        new_match[nonmatch_session] = range(A2.shape[-1], A_union.shape[-1])
        matchings.append(new_match)

    assignments = np.empty((A_union.shape[-1], n_sessions)) * np.nan
    for sess in range(n_sessions):
        assignments[matchings[sess], sess] = range(len(matchings[sess]))

    return A_union, assignments, matchings, all_performance, xy_remaps_out


def register_ROIs_multisession(
        mouse_id: int, sess_ids: Sequence[int], allow_gaps: bool = False, 
        rec_type: str = 'learning_ppc', tags: Union[None, str, Sequence[Optional[str]]] = None,
        grouptag: Optional[str] = None, force: bool = False,
        use_accepted_only=False, align_flag=True, align_options: Optional[dict] = None,
        xy_remaps: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
        yx_position_guesses: Union[Sequence, np.ndarray, None] = None,
        template_rec_type: Optional[str] = None, template_params: Optional[dict] = None,
        **register_opts) -> dict:
    """
    Uses register_multisession to match spatial components between N sessions that have already been processed.
    Saves to a .pkl file with the datetime and given tag, if any.
    Returns the sess_ids that were processed, followed by the return values of register_multisession.
    Note that the component indices are relative to *accepted* components only of each session.

    alllow_gaps: if True, does not raise an error if there are no recordings for some of the sess_ids (just skips these ids).
    tags: if scalar, loads each result using this tag. Or can be a list the same length as sess_ids to apply the corresponding tag
        when loading each session (in this case, sess_ids may contain duplicates that are disambiguated by the tag.)
    grouptag: label to use when saving results for this set of sessions
    force: if false, attempts to load multisession results rather than re-running
    use_accepted_only: whether to do session to session matching on all cells or only accepted ones
    align_flag: whether to call align_templates_multisession to align based on templates before matching ROIs
        Has no effect if xy_remaps is provided.
    align_options: Options to pass to align_templates_multisession if align_flag is true
    xy_remaps: can be a list of N-1 pre-computed mappings from each session to the next. Alignment will be skipped.
        If provided, allow_gaps is forced to False because a gap with mappings already computed must indicate some error.
    yx_position_guesses: Estimated relative position of each session in (y, x) coordinates - see align_templates_multisession
    template_rec_type: if not None, loads sessinfo from different rec_type (i.e. 'learning_ppc_dlx' for structural) to make templates
        (TODO: find a way to also account for offset b/w functional & structural images if not recorded simultaneously)
    template_params: parameters to get_projection_for_seed to use to get template for each session (default: {'type': 'mean', 'norm_medw': 25})
    register_opts: other inputs to register_ROIs_multiple (e.g., increasing thresh_cost increases lenience for matches)
    """
    if grouptag == '':
        logging.warning('Empty string grouptag is interpreted as no tag (just use None)')
        grouptag = None

    if not force:
        try:
            res = load_latest_multisession(mouse_id, rec_type=rec_type, grouptag=grouptag)
            logging.info('Using existing multisession results')
            if 'processed_sess_ids' in res:
                logging.info(f'Sessions: {res["processed_sess_ids"]}')
            else:
                logging.warning('Session IDs not saved in results')
            return res
        except NoMultisessionResults:
            pass

    if allow_gaps and xy_remaps is not None:
        logging.warning('Setting allow_gaps to False because precomputed mappings were provided')
        allow_gaps = False

    processed_sess_ids: list[int] = []
    processed_tags: list[Optional[str]] = []
    cnmf_uuids: list[str] = []
    As = []
    accepted_inds = []
    n_planes = None
    borders: list[int] = []
    dims = None
    included_rois: list[np.ndarray] = []
    rec_dates: list[date] = []

    if not isinstance(tags, Sequence):
        tags = [tags] * len(sess_ids)

    for sess_id, tag in zip(sess_ids, tags):
        try:
            # load existing analysis object if possible
            info = cma.load_latest(mouse_id=mouse_id, sess_id=sess_id, rec_type=rec_type, tag=tag)
        except (RuntimeError, FileNotFoundError) as e:
            if allow_gaps and e.args[0] == cma.NO_FILES_MSG:
                logging.info(f'No recordings found for session {sess_id:03d}; continuing.')
                continue
            else:
                raise

        if info.cnmf_fit is None or info.cnmf_fit.estimates.A is None:
            raise RuntimeError(f'CNMF not run for session {sess_id}{tag if tag else ""}')

        if dims is None:
            dims = info.cnmf_fit.estimates.dims
        else:
            assert info.cnmf_fit.estimates.dims == dims, 'dims should be equal for all sessions'

        if n_planes is None:
            n_planes = info.metadata['num_planes']
        elif n_planes != info.metadata['num_planes']:
            raise RuntimeError(f'Mismatch in number of planes in session {sess_id}{tag if tag else ""}')
        
        # get border from motion correction results
        if info.mc_result is None:
            raise RuntimeError(f'Motion correction not run for session {sess_id}{tag if tag else ""}')
        
        borders.append(info.mc_result.border_to_0)

        # get contours to match from CNMF results
        A = info.cnmf_fit.estimates.A
        this_accepted_inds = info.cnmf_fit.estimates.idx_components
        assert this_accepted_inds is not None, 'Evaluation not run?'
        accepted_inds.append(this_accepted_inds)
        processed_sess_ids.append(sess_id)
        processed_tags.append(tag)
        uuid = info.get_selected_uuid()
        assert uuid is not None, 'Should have a selected UUID if there are CNMF results'
        cnmf_uuids.append(uuid)
        assert (rec_date := info.scan_day) is not None, 'Should know the date of the session'
        rec_dates.append(rec_date)

        if use_accepted_only:
            if isinstance(A, (sparse.coo_array, sparse.coo_matrix)):
                A = sparse.csc_array(A)  # necessary to allow indexing
            As.append(A[:, this_accepted_inds])
            included_rois.append(this_accepted_inds)
        else:
            As.append(A)
            included_rois.append(np.array(range(A.shape[1])))

    assert n_planes is not None and dims is not None, 'n_planes should be set with >= 1 session'

    # Do alignment, if requested/needed
    if xy_remaps is not None:
        align_flag = False
    elif align_flag:
        if template_rec_type is None:
            template_rec_type = rec_type
        xy_remaps = align_templates_multisession(
            mouse_id, sess_ids=processed_sess_ids, rec_type=template_rec_type, tags=processed_tags, align_options=align_options,
            template_params=template_params, yx_position_guesses=yx_position_guesses)

    spatial_union, assignments, matchings, all_performance, xy_remappings = register_ROIs_multiple(
        A=As, dims=dims, align_flag=False, xy_remaps=xy_remaps, n_planes=n_planes, borders=borders, **register_opts)
    
    coms = com(spatial_union, *dims)

    # also save which are accepted in each session
    if use_accepted_only:
        accepted = ~np.isnan(assignments)  # shortcut b/c we know these are all accepted
    else:
        accepted = np.stack([np.in1d(assn, acc) for assn, acc in zip(assignments.T, accepted_inds)], axis=1)        

    save_data = {
        'mouse_id': mouse_id,
        'included_rois': included_rois,
        'rec_type': rec_type,
        'processed_sess_ids': np.array(processed_sess_ids),
        'processed_tags': [tag if tag else '' for tag in processed_tags],
        'cnmf_uuids': cnmf_uuids,
        'grouptag': grouptag if grouptag else '',
        'spatial_union': spatial_union,
        'assignments': assignments,
        'accepted': accepted,
        'matchings': matchings,
        'all_performance': all_performance,
        'xy_remappings': xy_remappings,
        'cell_subset_name': 'accepted' if use_accepted_only else '',
        'center_of_mass': coms,
        'dates': rec_dates
    }
    save_data = tabularize_multisession_data(**save_data)
    save_multisession(save_data)
    return save_data


def save_multisession(save_data: dict) -> str:
    """
    Save register_ROIS_multisession results to a .pkl file and optionally also a mat-file.
    Returns the pkl filename.
    """
    grouptag = save_data['grouptag']
    mouse_id = save_data['mouse_id']
    rec_type = save_data['rec_type']
    data_dir = get_processed_dir(mouse_id, rec_type=rec_type)
    file_pattern = get_multisession_file_pattern(mouse_id, grouptag=grouptag)
    filename = make_timestamped_filename(file_pattern)
    filepath = os.path.join(data_dir, 'export', filename)
    logging.info(f'Saving multisession results to {filepath}')
    with open(filepath, 'wb') as file:
        pickle.dump(save_data, file)
    
    return filepath


def tabularize_multisession_data(*, mouse_id: int, processed_sess_ids: Sequence[int], processed_tags: Sequence[Optional[str]],
                                 included_rois: list[np.ndarray], accepted: np.ndarray, cnmf_uuids: list[str],
                                 assignments: np.ndarray, matchings: list[np.ndarray], center_of_mass: Optional[np.ndarray] = None,
                                 dates: list[date], **other_data) -> dict:
    """Reformat cell/matching data into pandas tables to make it easier to query"""
    n_cells, n_sessions = assignments.shape
    union_table = pd.DataFrame({
        'union_cell_id': range(n_cells),
        'n_sessions': np.sum(~np.isnan(assignments), axis=1),
        'n_sessions_accepting': np.sum(accepted, axis=1)
    })
    union_table.loc[:, 'frac_sessions'] = union_table.n_sessions / n_sessions
    union_table.loc[:, 'frac_sessions_accepting'] = union_table.n_sessions_accepting / n_sessions
    union_table.loc[:, 'frac_sessions_accepting_of_matched'] = union_table.n_sessions_accepting / union_table.n_sessions
    if center_of_mass is not None:
        union_table.loc[:, 'com_y'] = center_of_mass[:, 0]
        union_table.loc[:, 'com_xz'] = center_of_mass[:, 1]

    sess_names = make_sess_names(processed_sess_ids, processed_tags)

    session_table = pd.DataFrame({
        'mouse_id': mouse_id,
        'sess_id': processed_sess_ids,
        'tag': processed_tags,
        'sess_name': sess_names,
        'date': dates,
        'cnmf_uuid': cnmf_uuids
    })

    matchings_table = pd.concat([
        pd.DataFrame({
            'sess_name': sess_name,
            'union_cell_id': matching,
            'session_cell_id': included,
            'accepted': this_accepted[matching]
        })
        for sess_name, matching, included, this_accepted in zip(sess_names, matchings, included_rois, accepted.T)
    ], ignore_index=True)

    return {'all_cells': union_table, 'sessions': session_table, 'matchings': matchings_table,
            'mouse_id': mouse_id, **other_data}


def populate_accepted_column(matchings: pd.DataFrame, session_info: pd.Series, rec_type: str) -> None:
    """
    Fill in the accepted column of the matchings table by reading session cell metadata. Mutates matchings.
    session_info is one row of the session table; only applies to that session's rows of matchings.
    """
    if 'accepted' not in matchings:
        matchings.insert(3, 'accepted', False)
    
    # load corresponding pickle file
    metadata = cma.load_cell_metadata(
        mouse_id=session_info.mouse_id, sess_id=session_info.sess_id, uuid=session_info.cnmf_uuid,
        tag=session_info.tag, rec_type=rec_type)
    
    # transfer "accepted" info to matchings table
    this_sess = matchings.sess_name == session_info.sess_name
    sess_cell_ids = matchings.loc[this_sess, 'session_cell_id']
    matchings.loc[this_sess, 'accepted'] = metadata['cells'].loc[sess_cell_ids, 'accepted'].values


def get_multisession_file_pattern(mouse_id: Union[int, str], grouptag: Optional[str] = None) -> str:
    tagstr = grouptag + '_' if grouptag else ''
    return f'{mouse_id}_multisession_{tagstr}%dt.pkl'


def load_multisession(multisession_res_path: str) -> dict:
    with open(multisession_res_path, 'rb') as file:
        data = pickle.load(file)
    
    no_uuids_error =  RuntimeError('CNMF UUIDs are missing! These cell IDs may not match the latest exported data.\n'
                                   'Please re-run the registration (or fix manually if you know what you are doing).')
    no_included_rois = 'use_accepted_only' in data and 'included_rois' not in data
    if no_included_rois:
        data['cell_subset_name'] = 'accepted' if data['use_accepted_only'] else ''

    if 'all_cells' in data:
        if 'cnmf_uuid' not in data['sessions']:
            raise no_uuids_error
        
        if no_included_rois:
            if data['use_accepted_only']:
                raise RuntimeError('Matchings saved in table with incorrect session ids - re-run')
            else:
                # ids are correct, just remove 'use_accepted_only'
                del data['use_accepted_only']
        
    else:  # if in old (non-tabular) format, tabularize
        if 'cnmf_uuids' not in data:
            raise no_uuids_error

        if no_included_rois:
            if data['use_accepted_only']:
                # have to load each session to figure out which cell IDs are accepted
                included_rois: list[np.ndarray] = [] 
                for sess_id, tag, uuid in zip(data['processed_sess_ids'], data['processed_tags'], data['cnmf_uuids']):
                    sessinfo = cma.load_latest(data['mouse_id'], sess_id, rec_type=data['rec_type'], tag=tag)
                    sessinfo.select_gridsearch_run(uuid=uuid, force_reload=False, quiet=True)
                    assert sessinfo.cnmf_fit is not None and sessinfo.cnmf_fit.estimates.idx_components is not None, \
                        'No CNMF or evaluation results?'
                    included_rois.append(sessinfo.cnmf_fit.estimates.idx_components)
                data['included_rois'] = included_rois
            else:
                # all cells included so we can infer from length of results
                data['included_rois'] = [np.array(range(len(matchings))) for matchings in data['matchings']]
            del data['use_accepted_only']
        
        if 'center_of_mass' not in data:
            dims = data['xy_remappings'][0][0].shape
            data['center_of_mass'] = com(data['spatial_union'], *dims)

        if 'dates' not in data:
            data['dates'] = []
            for sess_id, tag, uuid in zip(data['processed_sess_ids'], data['processed_tags'], data['cnmf_uuids']):
                sessinfo = cma.load_latest(data['mouse_id'], sess_id, rec_type=data['rec_type'], tag=tag)
                assert (rec_date := sessinfo.scan_day) is not None, 'Should know scan day'
                data['dates'].append(rec_date)

        data = tabularize_multisession_data(**data)
    return data


def load_latest_multisession(mouse_id: Union[int, str], rec_type: str = 'learning_ppc', grouptag: Optional[str] = None) -> dict:
    """Load latest results of register_ROIs_multisession for given mouse/tag"""
    file_pattern = get_multisession_file_pattern(mouse_id, grouptag=grouptag)
    data_dir = get_processed_dir(mouse_id, rec_type=rec_type)
    export_dir = os.path.join(data_dir, 'export')
    latest_file = get_latest_timestamped_file(export_dir, file_pattern)
    if latest_file is None:
        raise NoMultisessionResults()

    return load_multisession(latest_file)    


def save_matched_thumbnails(multisession_res: dict, union_cell_ids: ArrayLike, session_names: Optional[Sequence[str]] = None,
                            **save_roi_thumbnails_opts) -> list[tuple[Optional[str], ...]]:
    """
    Save thumbnail images of given cells from given sessions (box_size x box_size pixels) in png format to files
    in a directory under the export dir of this mouse/rec type. Returns list that contains a tuple of the
    saved image paths for each session name, or None if the cell was not in that session.
    """
    session_table: pd.DataFrame = multisession_res['sessions']
    matchings_table: pd.DataFrame = multisession_res['matchings']
    mouse_id: int = multisession_res['mouse_id']
    rec_type: str = multisession_res['rec_type']

    if session_names is None:
        session_names = list(session_table.loc[:, 'sess_name'])

    save_paths_persession: list[list[Optional[str]]] = []
    cell_id_selector = pd.DataFrame({
        'union_cell_id': union_cell_ids,
        'sess_name': ''  # placeholder
    })

    for session_name in session_names:
        sess_inds = np.flatnonzero(session_table.sess_name == session_name)
        if len(sess_inds) == 0:
            raise ValueError(f'There is no session called {session_name} in these results')
        elif len(sess_inds) > 1:
            raise RuntimeError(f'Multiple sessions called {session_name} found!')
        sess_ind = sess_inds.item()

        # get session cell ids
        cell_id_selector.loc[:, 'sess_name'] = session_name
        match_res = matchings_table.merge(cell_id_selector, on=('union_cell_id', 'sess_name'), how='right')
        matched = match_res.session_cell_id.notna()
        sess_cell_ids = list(match_res.loc[matched, 'session_cell_id'].astype(int))
        
        # load session and select correct CNMF run
        sess_id = session_table.at[sess_ind, 'sess_id']
        tag = session_table.at[sess_ind, 'tag']
        uuid = session_table.at[sess_ind, 'cnmf_uuid']
        sessinfo = cma.load_latest(mouse_id, sess_id, tag=tag, rec_type=rec_type)
        sessinfo.select_gridsearch_run(uuid=uuid, force_reload=False)

        # save thumbnails
        saved_paths = sessinfo.save_roi_thumbnails(sess_cell_ids, **save_roi_thumbnails_opts)

        # fill in paths for indices that were saved
        this_save_paths: list[Optional[str]] = [None] * np.size(union_cell_ids)
        for valid_ind, path in zip(np.flatnonzero(matched), saved_paths):
            this_save_paths[valid_ind] = path
        
        save_paths_persession.append(this_save_paths)
    
    return list(zip(*save_paths_persession))


def get_zmax_templates_and_borders_multisession(
        mouse_id: Union[int, str], sess_ids: Sequence[int], rec_type='learning_ppc',
        tags: Union[None, Sequence[Optional[str]]] = None, projection_params: Optional[Union[str, dict]] = None,
        include_dead_pixel_border=False) -> tuple[list[np.ndarray], list[BorderSpec]]:
    """Load max-z templates and borders for a series of sessions (helper for register_ROIS_multisession_3D)"""
    if tags is None:
        tags = [None] * len(sess_ids)
    
    templates: list[np.ndarray] = []
    borders: list[BorderSpec] = []
    for sess_id, tag in zip(sess_ids, tags):
        sessinfo = cma.load_latest(mouse_id, sess_id, tag=tag, rec_type=rec_type)
        templates.append(sessinfo.get_zmax_projection(projection_params=projection_params))

        if sessinfo.mc_result is None:
            raise RuntimeError('Motion correction not run?')
        
        plane_borders: list[BorderSpec] = sessinfo.mc_result.border_asym
        border = BorderSpec.max(*plane_borders)

        if include_dead_pixel_border:
            # add border on left corresponding to dead pixels
            ndead = sessinfo.odd_row_ndeads
            offset = sessinfo.odd_row_offset
            crop = sessinfo.crop

            ndead_max = 0 if ndead is None else max(ndead)
            shift_max = 0 if offset is None else math.ceil(abs(offset) / 2)
            n_to_clip = max(ndead_max + shift_max - crop.left, 0)
            border = BorderSpec.max(border, BorderSpec(left=n_to_clip))

        borders.append(border)
    return templates, borders


@dataclass(init=False)
class SessionMappingData:
    """Stores info about ROIs in a session and mappings to other sessions"""
    xy_mappings_to_others: dict[str, np.ndarray]  # maps other session names to X/Y remappings
    accepted: np.ndarray  # idx_components
    uuid: str  # UUID of the CNMF run
    matchings: np.ndarray  # union cell IDs of matched components
    session_cell_ids: np.ndarray  # same length as matchings with the session cell IDs (defaults to all, in order)
    scan_date: date  # scan day

    def __init__(self, mouse_id: Union[int, str], sess_id: int, tag: Optional[str],
                 remaps_to_others: dict[str, np.ndarray], rec_type='learning_ppc',
                 session_cell_ids: Optional[np.ndarray] = None):
        """
        Inputs:
            - remaps_to_others should be a mapping from names of *other* sessions to the X/Y remaps
            from this session to others.
            - if dims is passed it will be verified, else it will be read from the saved SessionAnalysis.
        """
        self.sess_name = make_sess_name(sess_id, tag)
        self.xy_mappings_to_others = remaps_to_others

        sessinfo = cma.load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, quiet=True)
        assert (scan_day := sessinfo.scan_day) is not None, 'Should know recording date'
        self.scan_date = scan_day

        if sessinfo.cnmf_fit is None:
            raise RuntimeError(f'CNMF not run for session {sess_id}{tag if tag else ""}')

        est = sessinfo.cnmf_fit.estimates
        if est.A is None:
            raise RuntimeError('No A or dims; CNMF not run?')

        if est.idx_components is None:
            raise RuntimeError('No idx_components; evaluation not run?')

        self.accepted = est.idx_components
        if session_cell_ids is None:
            n_cells = est.A.shape[1]
            self.session_cell_ids = np.arange(n_cells)
        else:
            n_cells = len(session_cell_ids)
            self.session_cell_ids = session_cell_ids

        self.matchings = np.full(n_cells, -1, dtype=int)

        uuid = sessinfo.get_selected_uuid()
        assert uuid is not None, 'Should have a selected UUID if there are CNMF results'
        self.uuid = uuid


@dataclass(init=False)
class SessionMappingDataWithFlatFootprints(SessionMappingData):
    """SessionMappingData plus masks flattened in the Z axis"""
    dims: tuple[int, int]  # Y, X dimensions of each plane
    xy_footprints: sparse.csc_matrix  # footprints flattened across the Z axis
    com: ScaledDataFrame  # x, y, plane coordinates of each ROI, in um
    weights: np.ndarray  # sum of masks for each component

    def __init__(self, mouse_id: Union[int, str], sess_id: int, tag: Optional[str],
                 remaps_to_others: dict[str, np.ndarray], rec_type='learning_ppc',
                 session_cell_ids: Optional[np.ndarray] = None,
                 session_z_offset_um: Optional[float] = None, max_thr=0.,
                 pixel_thr_method: Literal['nrg', 'max'] = 'nrg', pixel_thr: Optional[float] = None):
        """
        Additional inputs:
            - session_z_offset_um: Offset of this session in the Z axis (if none, ignore Z axis)
            - max_thr: see caiman.base.rois.register_ROIs
            - pixel_thr_method, pixel_thr: parameters for counting pixels in each mask to make weights
        """
        super().__init__(mouse_id=mouse_id, sess_id=sess_id, tag=tag, remaps_to_others=remaps_to_others,
                         rec_type=rec_type, session_cell_ids=session_cell_ids)
        
        if pixel_thr is None:
            pixel_thr = 0.9 if pixel_thr_method == 'nrg' else 0.2

        sessinfo = cma.load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, quiet=True)
        if sessinfo.cnmf_fit is None:
            raise RuntimeError('CNMF not run?')
        est = sessinfo.cnmf_fit.estimates
        if est.dims is None:
            raise RuntimeError('No dims; CNMF not run?')
        
        self.dims = (int(est.dims[0]), int(est.dims[1]) // sessinfo.metadata['num_planes'])
        self.com = sessinfo.get_coms_3d(unit='um').iloc[self.session_cell_ids, :]
        if session_z_offset_um is None:
            # ignore Z
            self.com.loc[:, 'plane'] = 0
        else:
            # account for session-specific Z offset
            self.com.loc[:, 'plane'] += session_z_offset_um

        xy_footprints = sessinfo.get_xy_footprints(normalize=True)[:, self.session_cell_ids]
        self.weights = footprints.count_pixels(xy_footprints, method=pixel_thr_method, thr=pixel_thr)
        threshold_masks(xy_footprints, max_thr)
        self.xy_footprints = xy_footprints


def register_ROIs_multisession_3D(
        mouse_id: Union[int, str], sess_ids: Sequence[int], rec_type='learning_ppc', tags: Union[None, Sequence[Optional[str]]] = None,
        grouptag: Optional[str] = None, max_thr=0., thresh_cost=0.7, max_dist_um=20., n_matched_weight=0.5,
        pixel_thr_method: Literal['nrg', 'max'] = 'nrg', pixel_thr: Optional[float] = None,
        use_saved_xy_offsets=False, saved_offset_filename_fmt: Optional[str] = '{}_daily_offsets.csv',
        use_saved_mappings: Optional[bool] = None, save_mappings_with_grouptag: Optional[bool] = None) -> dict:
    """
    Register ROIs for individual planes while keeping track of the estimated Z position
    for each ROI, based on size of ROI in each session and estimated Z of each plane
    in each session. Z offset between each ROI and planes of the current session is taken into
    account when scoring ROI matches.

    Args:
        mouse_id...max_dist_um:
            See register_ROIs_multisession, register_ROIs (except max_dist_um is in um instead of pixels)

        n_matched_weight (float):
            How much to weight how many previous sessions each ROI has matched with when making matchings.
            Any value > 0 will bias the algorithm towards matching with ROIs with more previous matched sessions.
            each column i of the distance matrix passed to the Hungarian algorithm
            (corresponding to union ROIs) is raised to the power of (1 + weight * (n_matched[i]-1)).

        pixel_thr_method (Literal['nrg', 'max']):
            Method of thresholding pixel values in each ROI to obtain a weight for combining with other
            ROIs based on number of pixels. Either cumulative energy or fraction of max value.
        
        pixel_thr (Optional[float]):
            The value of the energy or max threshold, see above. Defaults to 0.9 for energy or 0.2 for max.

        use_saved_xy_offsets (bool):
            Load X/Y um offsets of each session from the first from a file and convert to pixels.
            Otherwise, estimates rigid offsets using guess_yx_positions_multiple.
            Either way, this is only used as the initial seed for nonrigid XY alignment.
        
        saved_offset_filename_fmt (Optional[str]):
            Filename of saved um offsets, used for Z offsets and X/Y if use_saved_xy_offsets is true.
            Can be a format string that takes mouse_id as a parameter.
            If None (which is not the default), does not load any offsets and ignores Z position.
        
        use_saved_mappings (Optional[bool]):
            Whether to load nonrigid mappings between pairs of sessions from a file:
            - If False, skips trying to load and just computes the mappings
            - If True, only loads and errors if they are not available
            - If None (default), tries to load, and computes whichever are missing if they are not available.

        save_mappings_with_grouptag (bool):
            If new mappings are computed, whether to save them in a file with the grouptag or the general file for this mouse.
            The mappings do not depend on the matching-related parameters, so most of the time it doesn't make sense to
            save them separately. However, there is no capacity to merge sets of mappings, so to avoid overwriting, by
            default they are saved with the grouptag if the main file contains sessions that are not used here.
    """
    if len(sess_ids) == 0:
        raise ValueError('Must include at least one session in registration')

    if tags is None:
        tags = [None] * len(sess_ids)

    ## Step 1: load estimated rigid offsets
    if saved_offset_filename_fmt is None:
        logging.info('Matching without offsets; ignoring Z axis')
        daily_offsets_um = None
        offset_file = None
    else:
        daily_offsets_um = load_offsets_for_sessions(
            mouse_id, sess_ids, rec_type=rec_type, tags=tags, filename_fmt=saved_offset_filename_fmt)
        offset_file = saved_offset_filename_fmt.format(mouse_id)

    ## Step 2: get nonrigid X/Y mappings between sessions
    rigid_offsets = daily_offsets_um if use_saved_xy_offsets else None
    xy_remaps, xy_remap_path = load_or_compute_remaps_for_sessions(
        mouse_id, sess_ids, rec_type=rec_type, tags=tags, grouptag=grouptag, use_saved_mappings=use_saved_mappings,
        save_mappings_with_grouptag=save_mappings_with_grouptag,
        rigid_offsets=rigid_offsets)
        
    ## Step 3: load each session and collect masks, COMs, mappings, etc.
    logging.info('Loading ROI info from each session')
    session_data: list[SessionMappingDataWithFlatFootprints] = []
    sess_names = make_sess_names(sess_ids, tags)
    for i, (sess_id, tag, remaps_to_others) in enumerate(zip(sess_ids, tags, xy_remaps)):
        (other_names := sess_names.copy()).pop(i)
        remap_dict = {name: remap for name, remap in zip(other_names, remaps_to_others)}
        session_z_offset_um = None if daily_offsets_um is None else daily_offsets_um.iloc[i].at['z']

        this_session_data = SessionMappingDataWithFlatFootprints(
            mouse_id=mouse_id, sess_id=sess_id, tag=tag, remaps_to_others=remap_dict, rec_type=rec_type,
            session_z_offset_um=session_z_offset_um, max_thr=max_thr, pixel_thr_method=pixel_thr_method, pixel_thr=pixel_thr
        )
        session_data.append(this_session_data)
    
    if not all(sess.dims == session_data[0].dims for sess in session_data[1:]):
        raise RuntimeError('X/Y dimensions of sessions being registered do not all match')

    ## Step 4: session by session, match masks with union of matched cells found so far
    ## based on XYZ distance in um and XY mask overlap
    n_cells_union = 0
    comp_weights_union = np.zeros((0,))  # total weights to use when computing COMs
    n_matched = np.zeros((0,), dtype=int)
    for i, session in enumerate(session_data):
        logging.info(f'Matching to session {sess_ids[i]}')
        # build A_union and com_union to match with
        # only consider cells that have not yet been matched - might do multiple passes in the future
        cells_to_map = np.setdiff1d(np.arange(n_cells_union), session.matchings)
        A_union = sparse.lil_array((session.xy_footprints.shape[0], n_cells_union))
        com_union = np.zeros((n_cells_union, 3))

        for other_session in session_data:
            if other_session is session:
                continue
            
            # map ROIs
            map_mask = np.in1d(other_session.matchings, cells_to_map)
            mapped_cell_ids = other_session.matchings[map_mask]
            xy_mapping = other_session.xy_mappings_to_others[session.sess_name]
            A_mapped = footprints.map_footprints(other_session.xy_footprints[:, map_mask], tuple(xy_mapping))
            A_union[:, mapped_cell_ids] += A_mapped.multiply(other_session.weights[map_mask])
            coms_mapped = remap_points_from_df(other_session.com.loc[map_mask, :], *xy_mapping)
            coms_weighted = coms_mapped.to_numpy() * other_session.weights[map_mask, np.newaxis]
            com_union[mapped_cell_ids, :] += coms_weighted
        
        # Only consider cells that were actually mapped
        com_union = com_union[cells_to_map, :] / comp_weights_union[cells_to_map, np.newaxis]
        A_union = sparse.csc_array(A_union[:, cells_to_map])
        # normalize to be unit vectors
        A_union = A_union / A_union.power(2).sum()

        # now find matchings to this session's ROIs using register_ROIs
        D_pow = 1 + n_matched_weight * (n_matched[cells_to_map] - 1)  # weigh components by n_matched

        reg_results = register_ROIs(
            A1=session.xy_footprints, A2=A_union, dims=session.dims, align_flag=False,
            com1=session.com.to_um().to_numpy(), com2=com_union,
            max_thr=max_thr, thresh_cost=thresh_cost, max_dist=max_dist_um, D_pow=D_pow
        )
        # update matchings and other info
        matched_session_inds = reg_results.matched1
        unmatched_session_inds = reg_results.unmatched1
        matched_union_inds = cells_to_map[reg_results.matched2]
        session.matchings[matched_session_inds] = matched_union_inds
        session.matchings[unmatched_session_inds] = range(n_cells_union, n_cells_union + len(unmatched_session_inds))
        n_cells_union += len(unmatched_session_inds)
        n_matched[matched_union_inds] += 1
        n_matched = np.concatenate((n_matched, np.ones(len(unmatched_session_inds), dtype=int)))
        comp_weights_union[matched_union_inds] += session.weights[matched_session_inds]
        comp_weights_union = np.concatenate((comp_weights_union, session.weights[unmatched_session_inds]))
    
    ## Step 5: compile data and save
    matchings: list[np.ndarray] = []
    included_rois: list[np.ndarray] = []
    assignments = np.full((n_cells_union, len(session_data)), np.nan)
    accepted = np.zeros((n_cells_union, len(session_data)), dtype=bool)

    for sess, assgn_col, accept_col in zip(session_data, assignments.T, accepted.T):
        matchings.append(sess.matchings)
        this_included_rois = sess.session_cell_ids
        included_rois.append(this_included_rois)
        assgn_col[sess.matchings] = this_included_rois
        accept_col[sess.matchings] = np.in1d(this_included_rois, sess.accepted)
    
    save_data = {
        'mouse_id': mouse_id,
        'included_rois': included_rois,
        'rec_type': rec_type,
        'processed_sess_ids': np.array(sess_ids),
        'processed_tags': [tag if tag else '' for tag in tags],
        'cnmf_uuids': [sess.uuid for sess in session_data],
        'grouptag': grouptag if grouptag else '',
        'assignments': assignments,
        'accepted': accepted,
        'matchings': matchings,
        'cell_subset_name': '',
        'xy_remap_file': os.path.split(xy_remap_path)[1],
        'offset_file': offset_file,
        'dates': [data.scan_date for data in session_data]
    }
    save_data = tabularize_multisession_data(**save_data)
    save_multisession(save_data)
    return save_data


def rebuild_session_mapping_data_and_footprints(
        align_results: dict, offset_file: Optional[str] = None) -> tuple[list[SessionMappingData], list[footprints.FootprintsPerPlane]]:
    """
    Gather session-mapping data and per-plane footprints from a completed multisession alignment run.
        - offset_file: can be provided to override Z offsets or fill in if the file was not saved with results.
    """
    # Unpack align_results into typed variables
    mouse_id: int = align_results['mouse_id']
    rec_type: str = align_results['rec_type']
    sessions: pd.DataFrame = align_results['sessions']
    matchings: pd.DataFrame = align_results['matchings']
    xy_remap_file: str = align_results['xy_remap_file']

    sess_ids: list[int] = list(sessions.loc[:, 'sess_id'])
    tags: list[str] = list(sessions.loc[:, 'tag'])
    sess_names: list[str] = list(sessions.loc[:, 'sess_name'])

    # we read offset_file from align_results unless it was provided
    if offset_file is None and not (offset_file := align_results.get('offset_file')):
        raise RuntimeError('Z offsets required to make per-plane session mapping data')
    
    try:
        offsets_um = load_offsets_for_sessions(
            mouse_id, sess_ids=sess_ids, rec_type=rec_type, tags=tags, filename_fmt=offset_file)
    except FileNotFoundError as e:
        raise RuntimeError('Rigid offsets file not found') from e
    except KeyError as e:
        raise RuntimeError('Not all session dates found in rigid offsets file') from e

    z_offsets = offsets_um.loc[:, 'z'].to_numpy()
    xy_remap_path = os.path.join(get_processed_dir(mouse_id, rec_type=rec_type), 'alignment', xy_remap_file)
    b_found, all_remaps = load_remaps_allpairs(xy_remap_path, sess_names)
    if not all(b_found):
        raise RuntimeError('Not all nonrigid mappings for these sessions found in ' + xy_remap_file)

    session_data: list[SessionMappingData] = []
    footprints_per_plane: list[footprints.FootprintsPerPlane] = []

    for i, (sess_id, tag, remaps_to_others, z_offset) in enumerate(zip(sess_ids, tags, all_remaps, z_offsets)):
        sess_name = (other_names := sess_names.copy()).pop(i)
        remap_dict = {name: remap for name, remap in zip(other_names, remaps_to_others)}
        session_cell_ids = matchings.session_cell_id[matchings.sess_name == sess_name].to_numpy()
        this_matchings = matchings.union_cell_id[matchings.sess_name == sess_name].to_numpy()

        this_session_data = SessionMappingData(
            mouse_id=mouse_id, sess_id=sess_id, tag=tag, remaps_to_others=remap_dict,
            rec_type=rec_type, session_cell_ids=session_cell_ids
        )
        this_session_data.matchings = this_matchings
        session_data.append(this_session_data)

        footprints_per_plane.append(footprints.FootprintsPerPlane(
            mouse_id=mouse_id, sess_id=sess_id, tag=tag, session_z_offset_um=z_offset,
            rec_type=rec_type, session_cell_ids=session_cell_ids
        ))
    
    # validate that dimensions match
    if len(footprints_per_plane) > 1:
        dims = footprints_per_plane[0].dims
        if not all(fp.dims == dims for fp in footprints_per_plane[1:]):
            raise RuntimeError('Not all dimensions match across requested sessions')

    return session_data, footprints_per_plane