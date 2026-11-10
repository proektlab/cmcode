"""Miscellaneous imaging data utilities"""
import glob
import os
import re
import logging
from typing import Optional, Sequence, Union, cast

from caiman.utils import sbx_utils
import numpy as np
from numpy.typing import DTypeLike
from tqdm import tqdm

from cmcode.util import paths
from cmcode.util.types import Integral

def subinds_per_file(sbx_files: list[str], frames: Union[int, slice]
                     ) -> tuple[dict[int, slice], int, tuple[int, int, int]]:
    """
    For loading files for average_raw_frames.
    Determine the slice of frames to load from each file to satisfy the overall frames slice
    Second output is the total number of frames that will be loaded
    Third output is the spatial dimensions - (Y, X, [Z]) - which should be the same for all files.
    """
    if isinstance(frames, int):
        frames = slice(frames)

    # first get the indices from the whole session to pull
    all_nframes: list[int] = []
    dims: Optional[tuple[int, int, int]] = None
    for file in sbx_files:
        _, x, y, z, nframes = sbx_utils.sbx_shape(file)
        if dims is None:
            dims = (y, x, z)
        elif dims != (y, x, z):
            raise RuntimeError('Files do not have the same spatial dimensions')
        all_nframes.append(nframes)
    if dims is None:
        raise RuntimeError('List of files was empty')

    cum_frames = np.insert(np.cumsum(all_nframes), 0, 0)
    frames_arr = np.arange(cum_frames[-1], dtype=int)[frames]

    # convert into slice for each individual recording
    frames_map: dict[int, slice] = {}
    step = abs(frames.step) if frames.step is not None else 1  # order doesn't matter
    for i, (start, end) in enumerate(zip(cum_frames[:-1], cum_frames[1:])):
        this_frames = frames_arr[(frames_arr >= start) & (frames_arr < end)]
        if len(this_frames) > 0:
            sub_start = min(this_frames) - start
            sub_end = max(this_frames) - start + 1
            frames_map[i] = slice(sub_start, sub_end, step)
    
    return frames_map, len(frames_arr), dims


def average_raw_frames(sbx_files: list[str], frames: Union[int, slice], *, channel: Optional[int] = 0,
                       subinds_spatial: Sequence[sbx_utils.DimSubindices] = (), crop_dead=True,
                       plane: Optional[int] = None, dtype: Optional[DTypeLike] = np.float64, quiet=False, dview=None,
                       odd_row_offset=0) -> np.ndarray:
    """load frames, slicing across all given files, and take mean projection"""
    subinds_map, total_frames, dims = subinds_per_file(sbx_files, frames)
    mean_data = None
    file_iterator = subinds_map.items()
    if not quiet:
        file_iterator = tqdm(file_iterator, desc='Loading and averaging selected frames...', unit='file')

    if crop_dead:  # crop out dead columns
        ndead = sbx_utils.get_odd_row_ndead(sbx_files[0]) - odd_row_offset // 2
        if len(subinds_spatial) == 0:
            subinds_spatial = (slice(None), slice(ndead, None))
        elif len(subinds_spatial) == 1:
            subinds_spatial = (subinds_spatial[0], slice(ndead, None))
        else:
            # remove the dead columns from the existing column subindices using setdiff
            curr_subinds_x = subinds_spatial[1]
            if isinstance(curr_subinds_x, slice):
                curr_subinds_x = range(dims[1])[curr_subinds_x]
            new_subinds_x = list(np.setdiff1d(np.asarray(curr_subinds_x), range(ndead)))
            subinds_spatial = list(subinds_spatial)
            subinds_spatial[1] = new_subinds_x

    for file_ind, subinds_t in file_iterator:
        subindices = (subinds_t,) + tuple(subinds_spatial)
        data = sbx_utils.sbxread(sbx_files[file_ind], subindices=subindices, channel=channel, plane=plane, odd_row_ndead=0,
                                 odd_row_offset=odd_row_offset, interp=False, dview=dview, quiet=True, dtype=dtype) # frames x Y x X [x planes]
        mean_data_file = np.sum(data, axis=0) / total_frames
        if mean_data is not None:
            mean_data += mean_data_file
        else:
            mean_data = mean_data_file
    
    assert mean_data is not None, "No files passed to average"   
    return mean_data


def get_trial_numbers_from_files(sbx_files: list[str]) -> tuple[np.ndarray, list[bool]]:
    """
    Extract trial numbers from SBX filenames (with or without file extension)
    1st output is the extracted numbers (int array)
    2nd output is the boolean mask of which files had a trial number
    """
    filenames = [os.path.split(filepath)[1] for filepath in sbx_files]
    trial_number_maybe_matches = [re.search(r'^[^_]+_\d+_(\d+)', fn) for fn in filenames]
    trial_number_matches = [m for m in trial_number_maybe_matches if m is not None]
    trial_numbers = np.array([int(match.group(1)) for match in trial_number_matches])
    return trial_numbers, [m is not None for m in trial_number_maybe_matches]


def find_sess_sbx_files(mouse_id: Union[int, str], sess_id: Union[int, Sequence[int]],
                        trials_to_include: Union[Optional[Sequence[int]], Sequence[Optional[Sequence[int]]]] = None,
                        trials_to_exclude: Union[Optional[Sequence[int]], Sequence[Optional[Sequence[int]]]] = None,
                        rec_type: str = 'learning_ppc', remove_ext: bool = False) -> tuple[list[str], np.ndarray]:
    """
    Gets a list of SBX files for the given mouse, session, etc. Removes the ".sbx" if remove_ext is true.
    If a list of sess_ids is given, returns a concatenated list of files from each session.
    Also returns the (0-based) trial indices from the full set of trials from all requested sessions.
    """
    if not isinstance(sess_id, Sequence) and (isinstance(sess_id, int) or np.ndim(sess_id) == 0):  # include numpy scalars
        sess_id = [sess_id]
        if not (trials_to_include is None or all(isinstance(tr, Integral) for tr in trials_to_include)):
            raise TypeError('trials_to_include cannot be a list of lists or None if a scalar session is passed')
        
        trials_to_include_each = cast(Sequence[Optional[Sequence[int]]], [trials_to_include])

        if not (trials_to_exclude is None or all(isinstance(tr, Integral) for tr in trials_to_exclude)):
            raise TypeError('trials_to_exclude cannot be a list of lists or None if a scalar session is passed')
        
        trials_to_exclude_each = cast(Sequence[Optional[Sequence[int]]], [trials_to_exclude])
    else:
        if trials_to_include is None:
            trials_to_include_each = [None] * len(sess_id)
        else:
            if any(isinstance(tr, Integral) for tr in trials_to_include):
                raise TypeError('trials_to_include cannot be a single list of ints if multiple sessions are passed')
            
            trials_to_include_each = cast(Sequence[Optional[Sequence[int]]], trials_to_include)

        if trials_to_exclude is None:
            trials_to_exclude_each = [None] * len(sess_id)
        else:
            if any(isinstance(tr, Integral) for tr in trials_to_exclude):
                raise TypeError('trials_to_exclude cannot be a single list of ints if multiple sessions are passed')
            
            trials_to_exclude_each = cast(Sequence[Optional[Sequence[int]]], trials_to_exclude)


    image_dir = paths.get_raw_dir(mouse_id, rec_type)

    trial_number_offset = 0
    numbers_per_sess: list[np.ndarray] = []
    sbx_files_all: list[str] = []

    for this_sess_id, tr_in, tr_ex in zip(sess_id, trials_to_include_each, trials_to_exclude_each):

        sbx_files = sorted(glob.glob(
            os.path.join(image_dir, f'{mouse_id}_{this_sess_id:03d}_*.sbx')))

        # remove any files that don't have a trial number for some reason
        trial_numbers, b_valid = get_trial_numbers_from_files(sbx_files)
        this_sess_n_trials = trial_numbers.size
        sbx_files = [f for f, bv in zip(sbx_files, b_valid) if bv]
        if not all(b_valid):
            logging.warning(f'Ignoring {sum(~np.array(b_valid))} .sbx files with no trial number')

        # apply trials_to_include
        if tr_in is not None:
            b_include = np.isin(trial_numbers, tr_in)
        else:
            b_include = np.ones(trial_numbers.shape, dtype=bool)

        if tr_ex is not None:
            b_include = b_include & ~np.isin(trial_numbers, tr_ex)

        trial_numbers = trial_numbers[b_include]

        if tr_in is not None and any(~np.isin(tr_in, trial_numbers)):
            logging.warning('Not all requested trials were found')

        sbx_files = [f for f, bi in zip(sbx_files, b_include) if bi]

        # strip off extension if requested
        if remove_ext:
            sbx_files = [os.path.splitext(f)[0] for f in sbx_files]

        numbers_per_sess.append(trial_numbers + trial_number_offset)
        trial_number_offset += this_sess_n_trials
        sbx_files_all.extend(sbx_files)

    trial_numbers_cat = np.concatenate(numbers_per_sess)
    return sbx_files_all, trial_numbers_cat