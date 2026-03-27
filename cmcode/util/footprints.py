"""Utilities for manipulating sets of cell footprints (A)"""
from dataclasses import dataclass
from functools import cache
import logging
import math
from typing import Any, Literal, Union, Sequence, Callable, Optional, Mapping, overload
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import optype.numpy as onp
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter
from scipy import sparse

from caiman.source_extraction.cnmf.merging import get_ROIs_to_merge

from cmcode import caiman_analysis as cma, cmcustom
from cmcode.cmcustom import my_get_contours
from cmcode.util.image import make_merge, BorderSpec
from cmcode.util.naming import make_sess_name
from cmcode.util.types import MaybeSparse, ST


# type for nonrigid remapping
RemapMats = tuple[onp.Array2D[np.floating], onp.Array2D[np.floating]]
RemapOrNull = Union[RemapMats, tuple[None, None]]


def normalize_footprints(A: MaybeSparse[np.floating]) -> sparse.csc_matrix[np.floating]:
    """Make a copy with each column normalized"""
    A = sparse.csc_matrix(A, copy=True)
    norms = np.asarray(np.sqrt(A.power(2).sum(axis=0))).ravel()
    for col, norm in enumerate(norms):
        if norm != 0:
            A.data[A.indptr[col]:A.indptr[col+1]] /= norm
    return A


def binarize_footprints(A: MaybeSparse, method: Literal['nrg', 'max'] = 'nrg', thr: float = 0.9,
                        nonempty_filter: Optional[onp.Array1D[Union[np.bool_, np.integer]]] = None
                        ) -> sparse.csc_matrix[np.bool_]:
    """
    Binarize pixels in each column of A using thresholding method
    Pass nonempty_filter as a list of indices or binary mask to indicate which columns should be processed.
    """
    A = sparse.csc_matrix(A)
    if A.dtype == np.bool_:
        # don't re-binarize
        return A

    if nonempty_filter is None:
        nonempty_filter = np.arange(A.shape[1])

    if onp.is_array_1d(nonempty_filter, np.bool_):
        if nonempty_filter.size != A.shape[1]:
            raise ValueError('Boolean nonempty_filter must be a mask with length equal to number of components')
        nonempty_filter = np.flatnonzero(nonempty_filter)
    elif not onp.is_array_1d(nonempty_filter, np.integer):
        raise TypeError('nonempty_filter must be a 1D array of ints or bools')

    if method == 'max':
        max_vals = A.max(axis=0)
        return A > max_vals * thr
    else:
        if method != 'nrg':
            raise ValueError(f'Unrecogized thresholding method {thr}')
        A_binarized = sparse.lil_matrix(A.T.shape, dtype=np.bool_)  # construct transposed
        for c in nonempty_filter:
            c = int(c)
            patch_data = A.data[A.indptr[c]:A.indptr[c+1]]
            if len(patch_data) == 0:
                continue

            indx_sorted = np.argsort(patch_data)[::-1]
            cumEn = np.cumsum(patch_data[indx_sorted]**2)
            above_thresh = np.zeros(len(patch_data), dtype=bool)  # in row order
            above_thresh[indx_sorted] = cumEn < thr * cumEn[-1]
            true_inds = A.indices[A.indptr[c]:A.indptr[c+1]][above_thresh]
            A_binarized[c, true_inds] = True
        return A_binarized.tocsr().T


def count_pixels(A: sparse.csc_matrix, method: Literal['nrg', 'max'] = 'nrg', thr: float = 0.9) -> onp.Array1D[np.integer]:
    """Just count pixels for each component in A after applying energy or max threshold"""
    if method == 'max' or A.dtype == bool:
        A_binarized = binarize_footprints(A, method, thr)
        return np.asarray(A_binarized.sum(axis=0))[0]
    else:
        if method != 'nrg':
            raise ValueError(f'Unrecogized thresholding method {thr}')
        n_pix = np.empty(A.shape[1], dtype=int)
        for c in range(A.shape[1]):
            patch_data = A.data[A.indptr[c]:A.indptr[c+1]]
            indx_sorted = np.argsort(patch_data)[::-1]
            cumEn = np.cumsum(patch_data[indx_sorted]**2)
            n_pix[c] = np.sum(cumEn < thr * cumEn[-1])
        return n_pix


def binarize_and_collapse_to_xy(A: MaybeSparse, n_planes: int, **binarize_kwargs) -> sparse.csc_matrix[np.bool_]:
    """
    Convert X/Y/Z ROIs to X/Y ROIs by binarizing each plane separately and taking the intersection.
    """
    xy_size = A.shape[0] // n_planes
    xy_mask = sparse.csc_matrix((xy_size, A.shape[1]), dtype=bool)

    for k_plane in range(n_planes):
        plane_footprints = A[k_plane*xy_size:(k_plane+1)*xy_size, :]
        plane_mask = binarize_footprints(plane_footprints, **binarize_kwargs)
        xy_mask = xy_mask + plane_mask

    return xy_mask
    

@overload
def collapse_footprints_to_xy(A: MaybeSparse[ST], n_planes: int, binarize: Literal[False], **binarize_kwargs) -> sparse.csc_matrix[ST]:
    ...

@overload
def collapse_footprints_to_xy(A: MaybeSparse, n_planes: int, binarize: Literal[True], **binarize_kwargs) -> sparse.csc_matrix[np.bool_]:
    ...

def collapse_footprints_to_xy(A: MaybeSparse, n_planes: int, binarize=False, **binarize_kwargs) -> sparse.csc_matrix:
    """
    Convert X/Y/Z ROIs to X/Y ROIs by summing over z.
    If binarize is true, binarizes each plane separately, to avoid weird effects of 
    components from different planes being jointly normalized.
    """
    if binarize:
        return binarize_and_collapse_to_xy(A, n_planes, **binarize_kwargs)

    # A = sparse.csc_array(A)
    xy_size = A.shape[0] // n_planes
    xy_footprints = sparse.csc_matrix((xy_size, A.shape[1]), dtype=A.dtype)

    for k_plane in range(n_planes):
        plane_footprints = A[k_plane*xy_size:(k_plane+1)*xy_size, :]
        xy_footprints += plane_footprints

    return sparse.csc_matrix(xy_footprints)


def get_bboxes(
        A: MaybeSparse, dims: tuple[int, int], nonempty_filter: Optional[onp.Array1D[Union[np.bool_, np.integer]]] = None,
        expand_radius=0) -> list[BorderSpec]:
    """For 2D footprints, find the bounding boxes of nonempty pixels"""
    if nonempty_filter is None:
        b_nonempty = np.ones(A.shape[1], dtype=bool)
    elif not onp.is_array_1d(nonempty_filter, np.bool_):
        b_nonempty = np.zeros(A.shape[1], dtype=bool)
        b_nonempty[nonempty_filter] = True
    else:
        b_nonempty = nonempty_filter

    if expand_radius < 0:
        raise ValueError('Expand radius must be nonnegative')

    bboxes: list[BorderSpec] = []
    for c, nonempty in zip(range(A.shape[1]), b_nonempty):
        nrows, ncols = dims
        if nonempty:
            if isinstance(A, np.ndarray):
                fp_2d = A[:, c].reshape(dims, order='F')
                rows, cols = np.nonzero(fp_2d)
            else:
                spatial_inds = A.indices[A.indptr[c]:A.indptr[c+1]]
                rows = spatial_inds % nrows
                cols = spatial_inds // nrows

            if len(rows) == 0:
                bboxes.append(BorderSpec.maximal(dims))
            else:
                border = BorderSpec(top=min(rows).item(), bottom=nrows - 1 - max(rows).item(),
                                    left=min(cols).item(), right=ncols - 1 - max(cols).item())
                if expand_radius != 0:
                    border = border.decreased(expand_radius)
                bboxes.append(border)
        else:
            bboxes.append(BorderSpec.maximal(dims))
    return bboxes


def smooth_footprints(
        A: MaybeSparse[np.floating], dims: tuple[int, int], sigma: float, truncate_sigmas=4.,
        nonempty_filter: Optional[onp.Array1D[Union[np.bool_, np.integer]]] = None,
        bboxes: Optional[list[BorderSpec]] = None) -> sparse.csc_matrix[np.floating]:
    """
    Smooth footprints with a Gaussian kernel.
        dims: spatial dimensions of each footprint
        sigma: SD of kernel
        truncate_sigmas: how many SDs to truncate the filter at (beyond will be filled with zeros)
        nonempty_filter: list of indices or binary mask to indicate which columns should be processed.
        bboxes: list of BorderSpecs reflecting bounding boxes of nonempty pixels, for efficient filtering.
            If not None, will be UPDATED with new bounding boxes after smoothing.
    """
    if bboxes is None:
        bboxes = get_bboxes(A, dims=dims, nonempty_filter=nonempty_filter)

    if nonempty_filter is None:
        nonempty_filter = np.arange(A.shape[1])
    elif onp.is_array_1d(nonempty_filter, np.bool_):
        if nonempty_filter.size != A.shape[1]:
            raise ValueError('Boolean nonempty_filter must be a mask with length equal to number of components')
        nonempty_filter = np.flatnonzero(nonempty_filter)
    elif not onp.is_array_1d(nonempty_filter, np.integer):
        raise TypeError('nonempty_filter should be a 1D array of ints or bools')

    # process one component at a time
    radius = round(truncate_sigmas * sigma)
    A_smoothed = sparse.lil_matrix(A.T.shape, dtype=float)
    for c in nonempty_filter:
        c = int(c)
        # efficiently smooth only the nonzero entries
        # update bboxes first to include zeros within the filter's radius
        bboxes[c] = bboxes[c].decreased(radius)
        bbox = bboxes[c]
        bbox_slices = bbox.slices(dims)
        bbox_mask = bbox.flatmask(dims, order='F')

        comp = A[:, c] if isinstance(A, np.ndarray) else A[:, c].toarray()
        fp = comp.reshape(dims, order='F')[bbox_slices].astype(float)
        A_smoothed[c, bbox_mask] = gaussian_filter(fp, sigma, mode='nearest', radius=radius).ravel(order='F')

    return A_smoothed.tocsr().T
    

def map_footprints(A: MaybeSparse, xy_remap: RemapOrNull) -> sparse.csc_matrix[np.float32]:
    """Use x/y remap derived from align_templates to map a set of footprints"""
    x_remap, y_remap = xy_remap
    if x_remap is None or y_remap is None:
        if x_remap is not None or y_remap is not None:
            raise ValueError('x_remap and y_remap should be either both or neither None')
        return sparse.csc_matrix(A, copy=True)

    dims = x_remap.shape
    A2 = sparse.csc_matrix(A)
    rois2_2d = (A2[:, [i]].toarray().reshape(dims, order='F') for i in range(A2.shape[1])) 
    coo_data = np.array([], dtype=np.float32)
    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    # use iterator to avoid pulling whole array into memory at once
    for i, roi2_2d in enumerate(rois2_2d):
        remapped_roi_2d = np.empty(x_remap.shape, dtype=np.float32)
        cv2.remap(roi2_2d.astype(np.float32), x_remap, y_remap, cv2.INTER_NEAREST, dst=remapped_roi_2d)
        remapped_roi = np.ravel(remapped_roi_2d, order='F')
        roi_nonzero = np.flatnonzero(remapped_roi)
        coo_data = np.concatenate((coo_data, remapped_roi[roi_nonzero]))
        rows = np.concatenate((rows, roi_nonzero))
        cols = np.concatenate((cols, np.repeat(i, len(roi_nonzero))))

    return sparse.csc_matrix(sparse.coo_matrix((coo_data, (rows, cols)), shape=A2.shape))


def make_spatial_seed_from_projection(proj: onp.Array2D[np.floating], seed_params_extra: dict[str, Any]) -> sparse.csc_array[np.bool_]:
    """Extract binary spatial seed from projection image and parameters"""
    borders: list[BorderSpec] = seed_params_extra.pop('borders')
    concat_planes = len(borders)
    
    # get masks from each plane separately
    if concat_planes > 1:
        planes = np.split(proj, concat_planes, axis=1)
    else:
        planes = [proj]

    Ain_planes: list[sparse.csc_array[np.bool_]] = []
    for plane, border in zip(planes, borders):
        center_slices = border.slices(plane.shape)
        plane_center = plane[center_slices]
        Ain_center, _ = cmcustom.my_extract_binary_masks_from_structural_channel(plane_center, **seed_params_extra)
        # fix rows to take border into account
        ind_array = np.arange(plane.size, dtype=np.int32).reshape(plane.shape, order='F')
        inds_used = ind_array[center_slices].ravel(order='F')

        Ain_plane = sparse.csc_array( # offset indices to take border into account
            (Ain_center.data, inds_used[Ain_center.indices], Ain_center.indptr),
            shape=(plane.size, Ain_center.shape[1])
        )
        Ain_planes.append(Ain_plane)

    return sparse.block_diag(Ain_planes, format='csc')  # type: ignore


def augment_data_for_interpolation(footprints: np.ndarray, zs: Union[Sequence[float], np.ndarray], n_border_points,
                                   z_border: float = 20, min_total_points=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Add linear ramp to 0 above and below given footprint and z data. n_border_points will be added going up
    and down until z_border from the max/min z points. If min_total_points is nonzero, will also ensure there are
    at least this number of points in the resulting arrays (increasing the number of border points if necessary).
    Returns (footprints_aug, zs_aug).
    """
    n_unique = len(set(zs))
    n_aug_needed = max(2*n_border_points, min_total_points - n_unique)  # make_smoothing_spline requires at least 5 points
    n_aug_below = n_aug_needed // 2
    n_aug_above = n_aug_needed - n_aug_below
    min_z_ind, max_z_ind = int(np.argmin(zs)), int(np.argmax(zs))
    ramp_z_below = np.linspace(zs[min_z_ind] - z_border, zs[min_z_ind], n_aug_below, endpoint=False)
    ramp_z_above = np.linspace(zs[max_z_ind] + z_border, zs[max_z_ind], n_aug_above, endpoint=False)
    ramp_footprints_below = np.linspace(0, footprints[min_z_ind], n_aug_below, endpoint=False)
    ramp_footprints_above = np.linspace(0, footprints[max_z_ind], n_aug_above, endpoint=False)

    zs_aug = np.concatenate([zs, ramp_z_below, ramp_z_above])
    footprints_aug = np.concatenate([footprints, ramp_footprints_below, ramp_footprints_above], axis=0)
    return footprints_aug, zs_aug


def make_footprint_interpolator(footprints: onp.ToArrayStrict2D, zs: Union[Sequence[float], np.ndarray],
                                z_border: float = 20, n_border_points=0, **mss_kwargs) -> Callable[[float], np.ndarray]:
    """
    Make function to interpolate between multiple footprints at different z locations, to a new z location.
    Each row of footprints should be a single footprint across space, of length prod(dims) in F-order.
    N-D arrays are also supported, with z along the first axis.
    The length of zs should equal footprints.shape[0]. Typically this is in um although it only has to be in the 
        same unit as z_border.
    z_border is the maximum distance past the edge where the footprint might be nonzero;
        z values less than min(zs) - z_border or greater than max(zs) + z_border will always yiels all 0s.
    mss_kwargs are passed on to make_smoothing_spline. For example, "lam" controls the smoothness (larger = smoother);
        by default this is estimated automatically.
    The return value of the interpolator will be a vector, again flattened in F-order.
    """
    footprints = np.asarray(footprints)
    if footprints.shape[0] != len(zs):
        raise ValueError('Exactly one z value must be provided for each footprint')
    
    if z_border < 0:
        raise ValueError('z_border must be non-negative')
    
    pixel_shape = footprints.shape[1:]
    n_pixels = np.prod(pixel_shape)

    if len(zs) == 0:
        # just zeros
        return lambda _: np.zeros(pixel_shape)

    # find which pixels to include in interpolator
    inds_to_interp = np.flatnonzero(np.any(footprints != 0, axis=0))
    data_for_interp = footprints.reshape((len(zs), -1))[:, inds_to_interp]
    min_data_z, max_data_z = np.min(zs), np.max(zs)
    min_z = min_data_z - z_border
    max_z = max_data_z + z_border

    # augment data so that cubic interpolation works
    data_aug, zs_aug = augment_data_for_interpolation(   # make_smoothing_spline requires at least 5 points
        data_for_interp, zs, z_border=z_border, n_border_points=n_border_points, min_total_points=5)

    # sort and average any repeats
    z_sorted, inv_inds, counts = np.unique(zs_aug, return_inverse=True, return_counts=True)
    data_sorted = np.zeros((len(z_sorted), data_aug.shape[1]))
    for inv_ind, data in zip(inv_inds, data_aug):  # index into z_sorted and data_sorted for this footprint
        data_sorted[inv_ind] += data / counts[inv_ind]

    # make functions for interpolation
    interpolants = [make_smoothing_spline(z_sorted, d, **mss_kwargs) for d in data_sorted.T]

    def interpolator(z: float) -> np.ndarray:
        interp_vals = np.zeros(n_pixels)
        if z > min_data_z and z < max_data_z:
            for ind, interpolant in zip(inds_to_interp, interpolants):
                interp_vals[ind] = interpolant(z)
        elif z > min_z and z <= min_data_z:
            for ind, interpolant in zip(inds_to_interp, interpolants):
                minval = interpolant(min_data_z)
                interp_vals[ind] = minval * (z - min_z) / z_border
        elif z < max_z and z >= max_data_z:
            for ind, interpolant in zip(inds_to_interp, interpolants):
                maxval = interpolant(max_data_z)
                interp_vals[ind] = maxval * (max_z - z) / z_border
        return interp_vals.reshape(pixel_shape)
    
    return interpolator


@dataclass(init=False)
class FootprintsPerPlane:
    """Footprints with ROIs on multiple planes separate, with methods to binarize, smooth, etc."""
    dims: tuple[int, int]  # Y, X dimensions of each plane
    data: list[sparse.csc_matrix]  # each entry contains the ROIs from 1 plane
    z_positions: np.ndarray  # Z position of each plane in xy_masks
    nonempty: np.ndarray # n_planes x n_rois boolean matrix of whether each ROI is present in each plane
    bboxes: list[list[BorderSpec]]  # bboxes of nonempty pixels for each cell in each plane
    footprint_type: Literal['raw', 'binary', 'likelihood']  # what kind of data are in data
    source_session: str  # session name of session these are from
    space_of_session: Optional[str]  # session name of session the footprints are in the space of (same as source_session unless they were remapped)

    def __init__(self, mouse_id: Union[int, str], sess_id: int, tag: Optional[str],
                 session_z_offset_um: float, rec_type='learning_ppc', session_cell_ids: Optional[np.ndarray] = None):

        sessinfo = cma.load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, quiet=True)
        if sessinfo.cnmf_fit is None:
            raise RuntimeError('CNMF not run?')
        est = sessinfo.cnmf_fit.estimates
        if est.dims is None or est.A is None:
            raise RuntimeError('No dims or A; CNMF not run?')

        self.dims = (int(est.dims[0]), int(est.dims[1]) // sessinfo.metadata['num_planes'])

        if session_cell_ids is None:
            n_cells = est.A.shape[1]
            session_cell_ids = np.arange(n_cells)

        self.data = [A[:, session_cell_ids] for A in sessinfo.get_footprints_per_plane()]
        self.recalc_nonempty()
        self.recalc_bboxes()
        self.z_positions = sessinfo.get_relative_depths() + session_z_offset_um
        
        self.footprint_type = 'raw'
        self.source_session = make_sess_name(sess_id, tag)
        self.space_of_session = self.source_session

    def recalc_nonempty(self):
        self.nonempty = np.stack([masks.getnnz(axis=0) > 0 for masks in self.data])

    def recalc_bboxes(self):
        self.bboxes = []
        for A, nonempty in zip(self.data, self.nonempty):
            self.bboxes.append(get_bboxes(A, dims=self.dims, nonempty_filter=nonempty))

    # transformations:

    def binarize(self, **binarize_kwargs):
        """Binarize footprints from each plane. Takes same params as footprints.binarize_footprints."""
        if self.footprint_type == 'binary':
            return
        
        if self.footprint_type != 'raw':
            logging.warning('Binarizing footprints that are not of type "raw"')
        
        for i, (A, nonempty) in enumerate(zip(self.data, self.nonempty)):
            self.data[i] = binarize_footprints(A, nonempty_filter=nonempty, **binarize_kwargs)
        
        self.recalc_nonempty()
        self.footprint_type = 'binary'
    
    def smooth(self, **smooth_kwargs):
        """Smooth footprints from each plane. Takes same params as footprints.smooth_footprints."""
        if self.footprint_type != 'binary':
            logging.warning('Smoothing footprints that are not of type "binary"')

        for i, (A, nonempty, bboxes) in enumerate(zip(self.data, self.nonempty, self.bboxes)):
            self.data[i] = smooth_footprints(
                A, self.dims, nonempty_filter=nonempty, bboxes=bboxes, **smooth_kwargs)

        if self.footprint_type == 'binary':
            self.footprint_type = 'likelihood'


    def remap(self, xy_remap: Union[onp.Array3D[np.floating], Mapping[str, onp.Array3D[np.floating]]],
              to_sess: Optional[str] = None):
        """
        Remap footprints to be in the space of another session
        xy_remap can either be a single tuple (X, Y) of remapping matrices or a dict mapping from
        session names to mappings to those sessions. If it is a dict, to_sess must be non-None to 
        select the correct mapping. self.space_of_session will be updated to to_sess regardless.
        """
        if isinstance(xy_remap, Mapping):
            if to_sess is None:
                raise TypeError('to_sess must be provided if xy_remap is a dict')
            xy_remap = xy_remap[to_sess]

        if len(xy_remap) != 2:
            raise TypeError('xy_remap must have a size of 2 along the first dimension (X and Y)')

        this_remap = (xy_remap[0], xy_remap[1])
        self.data = [map_footprints(fp, this_remap) for fp in self.data]
        self.space_of_session = to_sess

        # recompute nonemepty and bboxes relative to new space
        self.recalc_nonempty()
        self.recalc_bboxes()


def validate_all_mapped_to_same_session(footprints: Sequence[FootprintsPerPlane]):
    """
    Error if not all space_of_sessions are the same and warn if some are unknown.
    """
    warned = False
    to_sess: Optional[str] = None
    for fp in footprints:
        if fp.space_of_session is None:
            if not warned:
                 warn("One or more sessions' footprints cannot be verified to be mapped to the same space as others.")
            warned = True
        elif to_sess is None:
            to_sess = fp.space_of_session
        elif fp.space_of_session != to_sess:
            raise RuntimeError("One or more sessions' footprints are not in the same space. "
                               "Use remap to map footprints from the space of one session to another.")


def maxproj_per_cell(footprints: Sequence[FootprintsPerPlane], matchings: Sequence[np.ndarray],
                     borders: Sequence[BorderSpec], cached=False) -> Callable[[int], np.ndarray]:
    """
    Make a function that maps cell IDs to max projections, optionally with a cache.
        - footprints: list of FootprintsPerPlane objects (1 per session) that are all mapped to the same session
        - matchings: for each session, the union cell ids that each cell in footprints corresponds to
        - borders: list of BorderSpec objects specifying what region of the image to fit each interpolator to (1 per cell)
        - cached: whether to cache the mapping from cell to interpolator.
    """
    validate_all_mapped_to_same_session(footprints)

    def make_max_projection(cell_id: int) -> np.ndarray:
        """Cache max projection near cell of interest"""
        dims = footprints[0].dims
        yslice, xslice = borders[cell_id].slices(dims)
        plot_shape = borders[cell_id].center_shape(dims)
        maxproj = np.zeros(plot_shape)

        for fp, this_matchings in zip(footprints, matchings):
            sess_cell_inds = np.flatnonzero(this_matchings == cell_id)
            if len(sess_cell_inds) == 0:
                continue
            sess_cell_ind = sess_cell_inds[0]

            for kplane in np.flatnonzero(fp.nonempty[:, sess_cell_ind]):
                footprint_2d = fp.data[kplane][:, sess_cell_ind].toarray().reshape(dims, order='F')
                footprint_window = footprint_2d[yslice, xslice]
                maxproj = np.maximum(maxproj, footprint_window)
        return maxproj

    if cached:
        make_max_projection = cache(make_max_projection)

    return make_max_projection


def footprint_interpolator_per_cell(
        footprints: Sequence[FootprintsPerPlane], matchings: Sequence[np.ndarray],
        borders: Sequence[BorderSpec], cached=False
        ) -> Callable[[int], Callable[[float], np.ndarray]]:
    """
    Make a function that maps cell IDs to Z-axis footprint interpolators, optionally with a cache.
        - footprints: list of FootprintsPerPlane objects (1 per session) that are all mapped to the same session
        - matchings: for each session, the union cell ids that each cell in footprints corresponds to
        - borders: list of BorderSpec objects (1 per cell) specifying what region of the image to fit each interpolator to
        - cached: whether to cache the mapping from cell to interpolator.
    """
    validate_all_mapped_to_same_session(footprints)

    def likelihood_map_interpolator(cell_id: int) -> Callable[[float], np.ndarray]:
        """Cache mappings from z position to likelihood maps"""
        dims = footprints[0].dims
        plot_slices = borders[cell_id].slices(dims)
        zs: list[float] = []
        all_footprints: list[np.ndarray] = []

        for fp, this_matchings in zip(footprints, matchings):
            sess_cell_inds = np.flatnonzero(this_matchings == cell_id)
            if len(sess_cell_inds) == 0:
                continue
            sess_cell_ind = sess_cell_inds[0]

            for kplane in np.flatnonzero(fp.nonempty[:, sess_cell_ind]):
                zs.append(float(fp.z_positions[kplane]))
                footprint_2d = fp.data[kplane][:, sess_cell_ind].toarray().reshape(dims, order='F')
                footprint_window = footprint_2d[plot_slices]
                all_footprints.append(footprint_window)

        return make_footprint_interpolator(all_footprints, zs)

    if cached:
        likelihood_map_interpolator = cache(likelihood_map_interpolator)

    return likelihood_map_interpolator


# The below are obsolete (replaced by EstimatesExt.merge_components_crossplane),
# but may still be useful for troubleshooting/visualizing merged components sometime

def get_ROIs_to_merge_crossplane(A: MaybeSparse, C: np.ndarray, n_planes: int, planes_to_merge: tuple[int, ...], thr=0.7
                                 ) -> tuple[list[np.ndarray], list[np.ndarray], list[sparse.csc_matrix]]:
    """
    Get components that can be merged between N planes of a single recording (does not work between recordings)
    Outputs: list of arrays of merged component indices; list of arrays of corresponding plane numbers;
             component X/Y masks from each plane
    """
    # Select components in the requested planes from A and flatten to 1 plane
    if not isinstance(A, sparse.csc_matrix):
        A = sparse.csc_matrix(A)

    plane_pixels = A.shape[0] // n_planes
    plane_per_comp = A.indices[A.indptr[:-1]] // plane_pixels
    subset_inds = np.flatnonzero(np.isin(plane_per_comp, planes_to_merge))
    plane_per_comp_subset = plane_per_comp[subset_inds]
    A_subset = A[:, subset_inds]
    
    A_flat = sparse.csc_matrix((A_subset.data, A_subset.indices % plane_pixels, A_subset.indptr), shape=(plane_pixels, A_subset.shape[1]))

    # Call merge_components to identify cells to merge from each plane
    merged_ROIs = get_ROIs_to_merge(A_flat, C[subset_inds], thr=thr)[0]

    # From the merged components, order by which plane they came from
    roi_planes = [plane_per_comp_subset[rois] for rois in merged_ROIs]
    to_remove: list[int] = []
    for i, planes in enumerate(roi_planes):
        if all(planes == planes[0]):
            to_remove.append(i)
            logging.warning(f'ROIs {merged_ROIs[i]} are all from the same plane; skipping')

    merged_ROIs = [rois for i, rois in enumerate(merged_ROIs) if i not in to_remove]
    roi_planes = [planes for i, planes in enumerate(roi_planes) if i not in to_remove]

    merged_ROIs_absolute = [subset_inds[rois] for rois in merged_ROIs]

    # build A matrices of merged components from each plane
    As_merged: list[sparse.csc_matrix] = []
    for k_plane in planes_to_merge:
        A_merged = sparse.lil_matrix((A_flat.shape[0], len(merged_ROIs)), dtype=A_flat.dtype)

        # this is not the best way to do it but just try it for now
        for k_comp, (rois, planes) in enumerate(zip(merged_ROIs, roi_planes)):
            for roi, plane in zip(rois, planes):
                if plane == k_plane:
                    A_merged[:, k_comp] += A_flat[:, roi]

        As_merged.append(sparse.csc_matrix(A_merged))

    return merged_ROIs_absolute, roi_planes, As_merged


def plot_ROIs_to_merge_crossplane(sessinfo: 'cma.SessionAnalysis', planes_to_merge: tuple[int, int],
                                  merged_ROIs: list[np.ndarray], roi_planes: list[np.ndarray],
                                  A0_merged: sparse.csc_matrix, A1_merged: sparse.csc_matrix, n_cols=1):
    """Plot results from get_ROIs_to_merge_crossplane"""
    projs_all = sessinfo.get_plane_projections({'type': 'mean', 'norm_medw': 25}, exclude_border=False)
    projs = [projs_all[i] for i in planes_to_merge]

    merge_projection = make_merge(projs[0], projs[1], 'g', 'm', clip_percentile=0.3)

    # load df/f and original A
    assert sessinfo.cnmf_fit is not None
    dff = sessinfo.cnmf_fit.estimates.F_dff_denoised
    if dff is None:
        sessinfo.make_df_over_f(denoised=True)
        dff = sessinfo.cnmf_fit.estimates.F_dff_denoised
    assert dff is not None
    assert (A_orig := sessinfo.cnmf_fit.estimates.A) is not None
    A_orig = sparse.csc_array(A_orig)

    # subset based on which components are in both planes
    in_planes_inds = np.flatnonzero([all(np.isin(planes_to_merge, planes)) for planes in roi_planes])
    A0_merged = A0_merged[:, in_planes_inds]
    A1_merged = A1_merged[:, in_planes_inds]

    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(15, 8))
        n_rows = math.ceil(A0_merged.shape[1] / n_cols)
        axs = fig.subplot_mosaic([['image', [[f'dff{kc*n_rows + kr}' for kc in range(n_cols)] for kr in range(n_rows)]]])

        axs['image'].imshow(merge_projection)
        axs['image'].set_axis_off()

        for k_plane, (plane, A, color) in enumerate(zip(planes_to_merge, (A0_merged, A1_merged), ('g', 'm'))):
            # get merged ROI contour
            coo = my_get_contours(A, projs[0].shape, thr=0.2, thr_method='max')
            for i, (c, ind) in enumerate(zip(coo, in_planes_inds)):
                this_rois = merged_ROIs[ind]
                this_roi_planes = roi_planes[ind]
                axs['image'].plot(*c['coordinates'].T, c=color)
                if k_plane == 0:
                    axs['image'].text(c['CoM'][1], c['CoM'][0], str(ind), color='k', clip_on=True)
                
                # Plot df/f on the right side
                # do weighted average of dffs included
                rois_thisplane = this_rois[this_roi_planes == plane]
                roi_weights = A_orig[:, rois_thisplane].sum(axis=0)
                roi_weights = roi_weights / roi_weights.sum()
                dff_total = dff[rois_thisplane[0]] * roi_weights[0]
                for roi, weight in zip(rois_thisplane[1:], roi_weights[1:]):
                    dff_total += dff[roi] * weight
                    
                # plot with second plane being negative
                axs[f'dff{i}'].plot(dff_total * (-1) ** k_plane, color=color)
                axs[f'dff{i}'].set_title(f'component {ind}')
    fig.tight_layout()
    return fig
