"""Miscellaneous imaging data utilities"""
import glob
import os
import re
import logging
from typing import Optional, Sequence, Union

from caiman.utils import sbx_utils
import numpy as np
from tqdm import tqdm

from cmcode.util import paths

def subinds_per_file(sbx_files: list[str], frames: Union[int, slice]) -> tuple[dict[int, slice], int]:
    """
    For loading files for average_raw_frames.
    Determine the slice of frames to load from each file to satisfy the overall frames slice
    Second output is the total number of frames that will be loaded
    """
    if isinstance(frames, int):
        frames = slice(frames)

    # first get the indices from the whole session to pull
    all_nframes = [sbx_utils.sbx_shape(file)[4] for file in sbx_files]
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
    
    return frames_map, len(frames_arr)


def average_raw_frames(sbx_files: list[str], frames: Union[int, slice], channel: Optional[int] = 0,
                       subinds_spatial: Sequence[sbx_utils.DimSubindices] = (), crop_dead=True,
                       plane: Optional[int] = None, to32: Optional[bool] = None, quiet=False, dview=None,
                       odd_row_offset=0) -> np.ndarray:
    """load frames, slicing across all given files, and take mean projection"""
    subinds_map, total_frames = subinds_per_file(sbx_files, frames)
    mean_data = None
    file_iterator = subinds_map.items()
    if not quiet:
        file_iterator = tqdm(file_iterator, desc='Loading and averaging selected frames...', unit='file')
    for file_ind, subinds_t in file_iterator:
        subindices = (subinds_t,) + tuple(subinds_spatial)
        data = sbx_utils.sbxread(sbx_files[file_ind], subindices=subindices, channel=channel, plane=plane, odd_row_ndead=0,
                                 odd_row_offset=odd_row_offset, interp=False, dview=dview, quiet=True, to32=to32) # frames x Y x X [x planes]
        mean_data_file = np.sum(data, axis=0) / total_frames
        if mean_data is not None:
            mean_data += mean_data_file
        else:
            mean_data = mean_data_file
    assert mean_data is not None, "No files passed to average"

    # crop out dead columns
    if crop_dead:
        ndead = sbx_utils.get_odd_row_ndead(sbx_files[0]) - odd_row_offset // 2
        mean_data = mean_data[:, ndead:]
    
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


def find_sess_sbx_files(mouse_id: Union[int, str], sess_id: int, trials_to_include: Optional[Sequence[int]] = None,
                        trials_to_exclude: Optional[Sequence[int]] = None,
                        rec_type: str = 'learning_ppc', remove_ext: bool = False) -> list[str]:
    """Gets a list of SBX files for the given mouse, session, etc. Removes the ".sbx" if remove_ext is true."""
    image_dir = paths.get_raw_dir(mouse_id, rec_type)
    sbx_files = sorted(glob.glob(
        os.path.join(image_dir, f'{mouse_id}_{sess_id:03d}_*.sbx')))

    # remove any files that don't have a trial number for some reason
    trial_numbers, b_valid = get_trial_numbers_from_files(sbx_files)
    sbx_files = [f for f, bv in zip(sbx_files, b_valid) if bv]
    if not all(b_valid):
        logging.warning(f'Ignoring {sum(~np.array(b_valid))} .sbx files with no trial number')

    # apply trials_to_include
    if trials_to_include is not None:
        b_include = np.in1d(trial_numbers, trials_to_include)
    else:
        b_include = np.ones(trial_numbers.shape, dtype=bool)

    if trials_to_exclude is not None:
        b_include = b_include & ~np.in1d(trial_numbers, trials_to_exclude)

    trial_numbers = trial_numbers[b_include]
    if trials_to_include is not None and any(~np.in1d(trials_to_include, trial_numbers)):
        logging.warning('Not all requested trials were found')
    sbx_files = [f for f, bi in zip(sbx_files, b_include) if bi]

    # strip off extension if requested
    if remove_ext:
        sbx_files = [os.path.splitext(f)[0] for f in sbx_files]

    return sbx_files