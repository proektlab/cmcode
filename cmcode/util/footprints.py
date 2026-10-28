"""Utilities for manipulating sets of cell footprints (A)"""
from collections.abc import Sequence, Mapping, Callable, Generator
from dataclasses import dataclass, replace, asdict
from functools import cache
import logging
import math
from typing import Any, Literal, Union, Optional, overload, cast
from typing_extensions import Self
from warnings import warn

import cellpose.models
import cv2
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import optype.numpy as onp
import pandas as pd
from scipy.interpolate import make_smoothing_spline, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy import sparse
import torch

from caiman.base import rois
from caiman.source_extraction.cnmf.merging import get_ROIs_to_merge

from cmcode import caiman_analysis as cma, cmcustom, caiman_params as cmp
from cmcode.cmcustom import my_get_contours
from cmcode.util import scaled
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


def get_threshold_mask_per_comp(
    A: sparse.csc_matrix, method: Literal['nrg', 'max'], thr: float, among_nonzero=True
    ) -> Generator[onp.Array1D[np.bool_]]:
    """
    For each column of A, yield a boolean mask for which pixels pass the threshold
    If among_nonzero is true, the indices are among the "filled" indices of A (A.indices) for each component.
    """
    use_max = method == 'max'
    if not use_max and method != 'nrg':
        raise ValueError(f'Unrecognized threshold method {method}')

    for c in range(A.shape[1]):
        comp = A.data[A.indptr[c]:A.indptr[c+1]]

        if comp.size == 0:
            if among_nonzero:  # handle no-data case separately, since it messes with the nrg method
                yield np.array([], dtype=bool)
            else:
                yield np.zeros(A.shape[0], dtype=bool)
        elif use_max:
            above_thresh = comp > comp.max() * thr
            if among_nonzero:
                yield above_thresh
            else:
                mask = np.zeros(A.shape[0], dtype=bool)
                mask[A.indices[A.indptr[c]:A.indptr[c+1]]] = above_thresh
                yield mask
        else:
            idx_sorted = np.argsort(comp)[::-1]
            cumEn = np.cumsum(comp[idx_sorted]**2)
            sorted_above_thresh = cumEn < thr * cumEn[-1]

            if among_nonzero:
                above_thresh = np.zeros(comp.size, dtype=bool)
                above_thresh[idx_sorted] = sorted_above_thresh
                yield above_thresh
            else:
                mask = np.zeros(A.shape[0], dtype=bool)
                mask[A.indices[A.indptr[c]:A.indptr[c+1]][idx_sorted]] = sorted_above_thresh
                yield mask


def count_pixels(A: sparse.csc_matrix, method: Literal['nrg', 'max'] = 'nrg', thr: float = 0.9) -> onp.Array1D[np.integer]:
    """Just count pixels for each component in A after applying energy or max threshold"""
    return np.array([mask.sum() for mask in get_threshold_mask_per_comp(A, method=method, thr=thr)])


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

    A_binarized = sparse.lil_matrix(A.T.shape, dtype=np.bool_)  # construct transposed
    # compute mask for each nonempty component
    for c, comp_mask in zip(
        nonempty_filter,
        get_threshold_mask_per_comp(A[:, nonempty_filter], method=method, thr=thr, among_nonzero=False)):
        A_binarized[int(c), :] = comp_mask

    return A_binarized.tocsr().T


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
    

def threshold_sparse_footprints_inplace(A: sparse.csc_matrix, method: Literal['nrg', 'max'], thr: float):
    """Modify masks to apply given threshold"""
    remove_rows = np.array([], dtype=np.int32)
    remove_cols = np.array([], dtype=np.int32)

    for c, keep_mask in enumerate(get_threshold_mask_per_comp(A, method=method, thr=thr, among_nonzero=True)):
        below_thr_rows = A.indices[A.indptr[c]:A.indptr[c+1]][~keep_mask]
        remove_rows = np.concatenate((remove_rows, below_thr_rows))
        remove_cols = np.concatenate((remove_cols, np.full_like(below_thr_rows, c)))

    remove_data = np.repeat(True, len(remove_rows))
    remove_mask = sparse.coo_matrix((remove_data, (remove_rows, remove_cols)), shape=A.shape)
    A[remove_mask] = 0


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


def get_coms_3d(
    A: MaybeSparse, plane_shape: tuple[int, int], um_per_pixel_x: float, um_per_pixel_y: float,
    depths: onp.Array1D[np.floating], unit: Literal['um', 'pixels'] = 'pixels') -> scaled.ScaledDataFrame:
    """
    Calculates the center of mass of each component
    (distance from top left corner and using the given depths for each plane)
    This weights the 2D COMs according to the number of pixels
    The pixel size is taken into account, particularly spacing between z-planes which may be uneven.
    Return value is a 3D position object with vectors for x y and z positions in um.
    """
    pix_y, pix_x = plane_shape
    um_vals_y = np.arange(pix_y) * um_per_pixel_y
    um_vals_x = np.arange(pix_x) * um_per_pixel_x
    spacings: dict[str, Optional[float]] = {'y': um_per_pixel_y, 'x': um_per_pixel_x}

    depths = np.array(depths)
    # if the planes are uniform, we save the spacing; otherwise just set it to None
    spacings_z = np.diff(depths)
    if spacings_z.size > 0 and np.all(spacings_z == spacings_z[0]):
        spacings['plane'] = float(spacings_z[0])
    else:
        spacings['plane'] = None
    
    # now actually calculate the COMs, interpreting dims as being in 3D
    coms_3d_um = cmcustom.my_com(A, um_vals_y, um_vals_x, depths)

    # if we want the result in pixels but the z-spacing is nonuniform, we have to interpolate
    if spacings['plane'] is None and unit == 'pixels':
        spacings.pop('plane')
        yx_df = scaled.make_um_df(coms_3d_um[:, :-1], pixel_size=spacings)
        depth_sortinds = np.argsort(depths)  # np.interp requires sorted xp
        plane_pix = np.interp(coms_3d_um[:, -1], depths[depth_sortinds], depth_sortinds)
        plane_df = scaled.make_pixel_df({'plane': plane_pix})
        coms_df = cast(scaled.ScaledDataFrame, pd.concat([yx_df, plane_df], axis=1))
    else:
        coms_df = scaled.make_um_df(coms_3d_um, pixel_size=spacings)

    return coms_df.to_unit(unit)


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


def make_spatial_seed_caiman(proj: onp.Array2D[np.floating], seed_params_extra: dict[str, Any]) -> sparse.csc_array[np.bool_]:
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


def make_spatial_seed_from_masks(masks: onp.Array2D[np.integer]) -> sparse.csc_array[np.bool_]:
    """Convert an image of labels (0 = background, >= 1 = specific ROI) to a binary spatial seed"""
    masks_flat = masks.ravel(order='F')
    filled_pixels = np.flatnonzero(masks_flat)
    comp_nums = masks_flat[filled_pixels] - 1  # now 0 to n_comps-1

    # make coordinate array 
    coo_data = np.ones(len(filled_pixels), dtype=bool)
    coords = (filled_pixels, comp_nums)
    seed = sparse.coo_array((coo_data, coords), (len(masks_flat), int(np.max(masks_flat))))
    return seed.tocsc()


def make_spatial_seed(
    proj: onp.Array2D[np.floating], seed_params: cmp.SeedParams, 
    seed_params_extra: Optional[dict[str, Any]] = None, default_gSig=5) -> sparse.csc_array[np.bool_]:
    """Make seed given projection and seed params"""
    if seed_params.use_cellpose:
        cellparams = seed_params.cellpose_params

        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            logging.warning('GPU not available for torch (when running cellpose)')

        model = cellpose.models.CellposeModel(gpu=use_gpu, **cellparams.get_constructor_params())
        masks, _, _ = model.eval(proj, **cellparams.get_eval_params())
        seed = make_spatial_seed_from_masks(masks)

    else:  # use my_extract_binary_masks_from_structural_channel
        if seed_params_extra is None:
            raise RuntimeError('Cannot compute seed using caiman without seed_params_extra')

        for extract_param in ('gSig', 'blur_type', 'blur_gSig_multiple', 'min_area_size', 'min_hole_size', 'expand_method', 'selem'):
            seed_params_extra[extract_param] = getattr(seed_params, extract_param)

        # fill in default value of gSig
        if seed_params_extra['gSig'] is None:
            seed_params_extra['gSig'] = default_gSig

        seed = make_spatial_seed_caiman(proj, seed_params_extra)
    return seed


def plot_masks(masks: onp.Array2D[np.integer], background: Optional[onp.Array2D[np.floating]] = None):
    """Plot an image of ROI labels as a contour map on top of a background image"""
    fig, ax = plt.subplots()
    if background is not None:
        ax.imshow(background)
    ax.contourf(masks, levels=np.arange(0.5, np.max(masks)+0.5), colors=[f'C{i}' for i in range(10)], alpha=0.4)
    return fig


def extract_roi_thumbnails(
    A: MaybeSparse[np.floating], roi_ids: Sequence[int], proj: onp.Array2D[np.floating], box_size=25,
    edge_fill_value=0.) -> onp.Array3D[np.floating]:
    """Make a stack of thumbnail images from the given projection around the given ROIs"""
    dims = proj.shape
    coms: onp.Array2D[np.floating] = rois.com(A[:, roi_ids], *dims)
    thumbnails = np.full((len(roi_ids), box_size, box_size), edge_fill_value, dtype=proj.dtype)

    for com, dest in zip(coms, thumbnails):
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
        data = proj[top+nclip_top:bottom-nclip_bottom, left+nclip_left:right-nclip_right]
        dest[nclip_top:box_size-nclip_bottom, nclip_left:box_size-nclip_right] = data

    return thumbnails

def extract_roi_thumbnail(
    A: MaybeSparse[np.floating], roi_id: int, proj: onp.Array2D[np.floating], box_size=25) -> onp.Array2D[np.floating]:
    return extract_roi_thumbnails(A, [roi_id], proj, box_size=box_size)[0]


## ----------------- footprint interpolation ------------------ ##

@dataclass
class FootprintsPerPlane:
    """
    Footprints with ROIs on multiple planes separate, with methods to binarize, smooth, etc.
    Methods mutate the object's references, but should not mutate referenced objects themselves, so a shallow copy
        should be sufficient to avoid changing the original.
    """
    dims: tuple[int, int]  # Y, X dimensions of each plane
    data: sparse.csc_matrix  # pixels (flattened in F-order) x cells
    z_positions: onp.Array1D[np.floating]  # Z position of each plane (um)
    n_pixels_per_plane: onp.Array2D[np.integer]  # n_planes x n_rois matrix of # of pixels for each ROI in each plane
    bboxes: list[list[BorderSpec]]  # bboxes of nonempty pixels for each cell in each plane
    footprint_type: Literal['raw', 'binary', 'likelihood']  # what kind of data are in data
    source_session: str  # session name of session these are from
    space_of_session: Optional[str]  # session name of session the footprints are in the space of (same as source_session unless they were remapped)

    @property
    def n_planes(self) -> int:
        return self.z_positions.size

    @property
    def plane_pix(self) -> int:
        return self.dims[0] * self.dims[1]

    @property
    def nonempty(self) -> onp.Array2D[np.bool_]:
        """n_planes x n_rois boolean matrix of whether each ROI is present in each plane"""
        return self.n_pixels_per_plane > 0

    def plane_data(self, k_plane: int) -> sparse.csc_matrix:
        if k_plane < 0 or k_plane >= self.n_planes:
            raise ValueError(f'Invalid plane index; must be in [0, {self.n_planes})')
        
        return self.data[k_plane*self.plane_pix:(k_plane+1)*self.plane_pix, :]

    @property
    def planes(self) -> Generator[sparse.csc_matrix]:
        for k_plane in range(self.n_planes):
            yield self.data[k_plane*self.plane_pix:(k_plane+1)*self.plane_pix, :]


    @classmethod
    def load_from_session(
        cls, mouse_id: Union[int, str], sess_id: Union[int, str], tag: Optional[str], session_z_offset_um: float,
        rec_type='learning_ppc', session_cell_ids: Optional[np.ndarray] = None, uuid: Optional[str] = None) -> Self:
        """Typical factory method for loading footprints from SessionAnalysis"""

        sessinfo = cma.load_latest(mouse_id, sess_id, rec_type=rec_type, tag=tag, quiet=True)
        if uuid is not None:
            sessinfo.select_cnmf_run(uuid, save=False)

        if sessinfo.cnmf_fit is None:
            raise RuntimeError('CNMF not run?')
        est = sessinfo.cnmf_fit.estimates
        if est.dims is None or est.A is None:
            raise RuntimeError('No dims or A; CNMF not run?')

        n_cells = est.A.shape[1]
        if session_cell_ids is None:
            session_cell_ids = np.arange(n_cells)

        footprints = normalize_footprints(sparse.csc_matrix(est.A[:, session_cell_ids]))
        n_cells = footprints.shape[1]

        z_positions = sessinfo.get_relative_depths() + session_z_offset_um
        n_planes = z_positions.size

        if isinstance(sess_id, int):
            source_session = make_sess_name(sess_id, tag)
        else:
            source_session = sess_id

        self = cls(
            dims=(int(est.dims[0]), int(est.dims[1]) // sessinfo.metadata['num_planes']),
            data=footprints,
            z_positions=z_positions,
            n_pixels_per_plane=np.zeros((n_planes, n_cells), dtype=int),
            bboxes=[],
            footprint_type='raw',
            source_session=source_session,
            space_of_session=source_session
        )

        # initialize nonempty indicators and bounding boxes
        self.recalc_nonempty()
        self.recalc_bboxes()
        return self


    def select_and_copy(self, cell_index: Union[onp.ToInt1D, onp.ToBool1D, slice]):
        """
        Copy a subset of cells, specified by a boolean mask, indices, or a slice
        Note this is always an index directly into the set of cells in self, with no
        reference to session_cell_ids that may have been used when creating the object.
        """
        n_cells = self.data.shape[1]
        all_inds = np.arange(n_cells, dtype=int)
        if isinstance(cell_index, slice):
            cell_inds = all_inds[cell_index]
        else:
            cell_inds = all_inds[np.asarray(cell_index)]

        return type(self)(
            dims=self.dims,
            data=self.data[:, cell_inds],
            z_positions=self.z_positions,
            n_pixels_per_plane=self.n_pixels_per_plane[:, cell_inds],
            bboxes=[[plane_bboxes[i] for i in cell_inds] for plane_bboxes in self.bboxes],
            footprint_type=self.footprint_type,
            source_session=self.source_session,
            space_of_session=self.space_of_session
        )


    def recalc_nonempty(self):
        # count pixels in bins of rows and for each component using indptr
        plane_bins = np.arange(0, self.plane_pix * self.n_planes + 1, self.plane_pix)  # include endpoint
        comp_bins = self.data.indptr
        n_per_plane, _ = np.histogramdd((self.data.indices, np.arange(self.data.nnz)), bins=(plane_bins, comp_bins))
        self.n_pixels_per_plane = n_per_plane.astype(int)

    def recalc_bboxes(self):
        self.bboxes = []
        for plane_fp, nonempty in zip(self.planes, self.nonempty):
            self.bboxes.append(get_bboxes(plane_fp, dims=self.dims, nonempty_filter=nonempty))

    # transformations:

    def binarize(self, planewise=True, **binarize_kwargs):
        """
        Binarize footprints from each plane. Takes same params as footprints.binarize_footprints.
        By default, applies binarization to each plane separately; use planewise=False to apply to 3D footprints
        """
        if self.footprint_type == 'binary':
            return
        
        if self.footprint_type != 'raw':
            logging.warning('Binarizing footprints that are not of type "raw"')
        
        if planewise:
            self.data = sparse.vstack([
                binarize_footprints(A, nonempty_filter=nonempty, **binarize_kwargs)
                for A, nonempty in zip(self.planes, self.nonempty)
            ], format="csc")
        else:
            self.data = binarize_footprints(self.data, nonempty_filter=np.any(self.nonempty, axis=0), **binarize_kwargs)
        
        self.recalc_nonempty()
        self.footprint_type = 'binary'
    
    def smooth(self, sigma: float, **smooth_kwargs):
        """Smooth footprints from each plane. Takes same params as footprints.smooth_footprints."""
        if self.footprint_type != 'binary':
            logging.warning('Smoothing footprints that are not of type "binary"')

        self.data = sparse.vstack([
            smooth_footprints(A, self.dims, sigma, nonempty_filter=nonempty, bboxes=bboxes, **smooth_kwargs)
            for A, nonempty, bboxes in zip(self.planes, self.nonempty, self.bboxes)
        ], format="csc")

        if self.footprint_type == 'binary':
            self.footprint_type = 'likelihood'


    def remap(self, xy_remap: Union[onp.Array3D[np.floating], Mapping[str, onp.Array3D[np.floating]]],
              to_sess: Optional[str] = None):
        """
        Remap footprints to be in the space of another session
        xy_remap can either be a single tuple (X, Y) of remapping matrices or a dict mapping from
        session names to mappings to those sessions. If it is a dict, to_sess must be non-None to 
        select the correct mapping. self.space_of_session will be updated to to_sess regardless.

        If x-size of the mapping matches the plane size, it will be interpreted as a flat mapping to apply
        to each plane; otherwise it will be treated as a mapping that applies over the concatenated planes,
        if possible.
        """
        if isinstance(xy_remap, Mapping):
            if to_sess is None:
                raise TypeError('to_sess must be provided if xy_remap is a dict')
            xy_remap = xy_remap[to_sess]

        remap_dims, remap_len_y, remap_len_x = xy_remap.shape

        if remap_dims != 2:
            raise TypeError('xy_remap must have a size of 2 along the first dimension (X and Y)')
        
        if remap_len_y == self.dims[0] and remap_len_x == self.dims[1]:
            # flat remapping
            self.data = sparse.vstack([
                map_footprints(A, (xy_remap[0], xy_remap[1])) for A in self.planes
            ], format="csc")

        elif remap_len_y == self.dims[0] and remap_len_x == self.dims[1] * self.n_planes:
            # planewise remapping
            self.data = map_footprints(self.data, (xy_remap[0], xy_remap[1]))

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

            for kplane in range(fp.n_planes):
                if not fp.nonempty[kplane, sess_cell_ind]:
                    continue
                footprint_2d = fp.plane_data(kplane)[:, sess_cell_ind].toarray().reshape(dims, order='F')
                footprint_window = footprint_2d[yslice, xslice]
                maxproj = np.maximum(maxproj, footprint_window)
        return maxproj

    if cached:
        make_max_projection = cache(make_max_projection)

    return make_max_projection


# tagged union indicating what type of footprint interpolation to use and the settings
@dataclass
class InterpSettings:
    pass

@dataclass
class SmoothingSplineInterpSettings(InterpSettings):
    z_border: float = 20.  # distance beyond data extent in z where output may be nonzero
    lam: Optional[float] = None  # smoothing parameter

@dataclass
class RBFInterpSettings(InterpSettings):
    um_per_pixel_x: float
    um_per_pixel_y: float
    smoothing: float = 1.  # how closely to adhere to original data (does not have to do with spread)
    expansion_pix: int = 2  # how many pixels to potentially expand footprints by in X and Y
    reduce_neighbors_based_on_query_zs: bool = True
    max_dist: float = 20.  # only used for neighbors; should match max dist used for matching
    query_zs: Optional[onp.Array1D[np.floating]] = None  # only used for neighbors

@dataclass
class GaussianAverageSettings(InterpSettings):
    sigma: float  # standard deviation


def footprint_interpolator_per_cell(
        footprints: Sequence[FootprintsPerPlane], matchings: Sequence[np.ndarray],
        borders: Sequence[BorderSpec], interp_settings: InterpSettings, cached=False
        ) -> tuple[Callable[[int], Callable[[float], np.ndarray]], Sequence[BorderSpec]]:
    """
    Make a function that maps cell IDs to Z-axis footprint interpolators, optionally with a cache.
        - footprints: list of FootprintsPerPlane objects (1 per session) that are all mapped to the same session
        - matchings: for each session, the union cell ids that each cell in footprints corresponds to
        - borders: list of BorderSpec objects (1 per cell) specifying what region of the image to fit each interpolator to
        - cached: whether to cache the mapping from cell to interpolator.
        - interp_settings: interpolation settings for the desired interpolator type

    Returns a function that produces the interpolator given a union cell ID and a list of
        BorderSpecs for the interpolation output
    """
    validate_all_mapped_to_same_session(footprints)

    match interp_settings:
        case SmoothingSplineInterpSettings():
            interpolator_fn = make_smoothing_spline_footprint_interpolator
            borders_exp = borders  # no expansion
        case RBFInterpSettings(expansion_pix=expansion_pix):
            interpolator_fn = make_rbf_footprint_interpolator
            borders_exp = [border.decreased(expansion_pix) for border in borders]
        case GaussianAverageSettings():
            interpolator_fn = make_gaussian_average_interpolator
            borders_exp = borders
        case other:
            if other == InterpSettings():
                raise TypeError('A subclass of InterpSettings matching the interpolator type to use must be passed')
            else:
                raise TypeError(f'Unknown InterpSettings subclass {other.__class__.__name__}')

    def likelihood_map_interpolator(cell_id: int) -> Callable[[float], np.ndarray]:
        """Cache mappings from z position to likelihood maps"""
        dims = footprints[0].dims
        plot_slices = borders_exp[cell_id].slices(dims)
        zs: list[float] = []
        all_footprints: list[np.ndarray] = []

        for fp, this_matchings in zip(footprints, matchings):
            sess_cell_inds = np.flatnonzero(this_matchings == cell_id)
            if len(sess_cell_inds) == 0:
                continue
            sess_cell_ind = sess_cell_inds[0]

            for kplane in range(fp.n_planes):
                if not fp.nonempty[kplane, sess_cell_ind]:
                    continue

                zs.append(float(fp.z_positions[kplane]))
                footprint_2d = fp.plane_data(kplane)[:, sess_cell_ind].toarray().reshape(dims, order='F')
                footprint_window = footprint_2d[plot_slices]
                all_footprints.append(footprint_window)

        if len(all_footprints) == 0:
            return lambda z: np.zeros(borders_exp[cell_id].center_shape(dims))
        else:
            return interpolator_fn(all_footprints, zs, **asdict(interp_settings))

    if cached:
        likelihood_map_interpolator = cache(likelihood_map_interpolator)

    return likelihood_map_interpolator, borders_exp


def make_gaussian_average_interpolator(
    footprints: onp.ToArray1D, zs: Union[Sequence[float], onp.Array1D[np.floating]],
    sigma: float) -> Callable[[float], np.ndarray]:
    """
    Make function to not really interpolate, but just take a Gaussian-weighted average
    of the given footprints along z, with Gaussian centered at the query Z.
    """
    footprints = np.asarray(footprints)

    if footprints.shape[0] != len(zs):
        raise ValueError('Exactly one z value must be provided for each footprint')

    zs_arr = np.asarray(zs)

    def averager(query_z: float) -> np.ndarray:
        # compute Gaussian weights
        norm_dist = (zs_arr - query_z) / sigma
        wts = np.exp(-0.5 * norm_dist ** 2)  # keep unnormalized, max = 1
        return np.tensordot(wts, footprints, axes=1)

    return averager


def make_rbf_footprint_interpolator(
    footprints: onp.ToArrayStrict3D, zs: Union[Sequence[float], onp.Array1D[np.floating]],
    um_per_pixel_x: float, um_per_pixel_y: float, smoothing=1., expansion_pix=2,
    reduce_neighbors_based_on_query_zs=True, max_dist=20., query_zs: Optional[onp.Array1D[np.floating]] = None,
    min_val: Optional[float] = 0., max_val: Optional[float] = 1.) -> Callable[[float], np.ndarray]:
    """
    Make function to interpolate between multiple footprints at different z locations, to a new z location,
        using SciPy's RBFInterpolator (radial basis function) with a linear kernel.
    footprints should be a 3D matrix, Z x Y x X, where the Zs match the passed zs and do not need to be sorted or unique.
    um_per_pixel_x and um_per_pixel_y are needed so that the kernel is isotropic with respect to real distance,
        assuming that zs are also in um.
    smoothing: see RBFInterpolator.
    reduce_neighbors_based_on_query_zs: If True, based on Zs that will be queried later (passed in query_zs)
        and the max distance at which a match is possible, computes a reduced max number of neighbors to use in the
        interpolator that is also based on the size of the footprint.
    min_val, max_val: if not None, clip to these values after interpolating (necessary since it's cubic)
    The return value of the interpolator matches the shape of the input past the 1st dimension.
    """
    # construct grid of points
    footprints = np.asarray(footprints)
    nz, ny, nx = footprints.shape

    if nz != len(zs):
        raise ValueError('Exactly one z value must be provided for each footprint')
    
    pts_z, pts_y, pts_x = np.meshgrid(zs, np.arange(ny) * um_per_pixel_y, np.arange(nx) * um_per_pixel_x, indexing='ij')
    pts = np.stack([pts_z.ravel(), pts_y.ravel(), pts_x.ravel()], axis=1)
    vals = footprints.ravel()

    if reduce_neighbors_based_on_query_zs and query_zs is not None and query_zs.size > 0:
        # compute max neighbors we will need
        neighbors = 0
        zs_arr = np.asarray(zs)
        xy_dists_sq_max = pts_x.ravel() ** 2 + pts_y.ravel() ** 2

        for qz in query_zs:
            zdists_sq = (zs_arr - qz) ** 2
            zdists_sq = zdists_sq[zdists_sq <= max_dist ** 2]
            if zdists_sq.size == 0:
                continue

            # find max Euclidean distance to allow based on expansion_pix
            max_dist_sq = np.max(zdists_sq) + expansion_pix**2

            # find how many pixels in each used plane should be included
            this_neighbors = 0
            for zdist_sq in zdists_sq:
                this_neighbors += math.ceil(np.sum((xy_dists_sq_max + zdist_sq) <= max_dist_sq))

            neighbors = max(neighbors, this_neighbors)
    else:
        neighbors = None

    interpolant = RBFInterpolator(pts, vals,  smoothing=smoothing, neighbors=neighbors, kernel='linear')

    # points for interpolating with a single z
    pts_yx = np.stack([pts_y[0].ravel(), pts_x[0].ravel()], axis=1)

    def interpolator(z: float) -> np.ndarray:
        pts = np.concatenate([np.tile([[z]], (pts_yx.shape[0], 1)), pts_yx], axis=1)
        vals_interp = interpolant(pts)
        return np.clip(vals_interp, min_val, max_val).reshape((ny, nx))

    return interpolator


def make_smoothing_spline_footprint_interpolator(
    footprints: onp.ToArray2D, zs: Union[Sequence[float], onp.Array1D[np.floating]], z_border: float = 20,
    min_val: Optional[float] = 0., max_val: Optional[float] = 1., lam: Optional[float] = None
    ) -> Callable[[float], np.ndarray]:
    """
    Make function to interpolate between multiple footprints at different z locations, to a new z location,
        using SciPy's make_smoothing_spline.
    Each row of footprints should be a single footprint across space, of length prod(dims) in F-order.
    N-D arrays are also supported, with z along the first axis.
    The length of zs should equal footprints.shape[0]. Typically this is in um although it only has to be in the 
        same unit as z_border.
    z_border is the maximum distance past the edge where the footprint might be nonzero;
        z values less than min(zs) - z_border or greater than max(zs) + z_border will always yield all 0s.
    mss_kwargs are passed on to make_smoothing_spline. For example, "lam" controls the smoothness (larger = smoother);
        by default this is estimated automatically.
    min_val, max_val: if not None, clip to these values after interpolating (necessary since it's cubic)
    The return value of the interpolator matches the shape of the input past the 1st dimension..
    """
    footprints = np.asarray(footprints)
    if footprints.shape[0] != len(zs):
        raise ValueError('Exactly one z value must be provided for each footprint')
    
    if z_border < 0:
        raise ValueError('z_border must be non-negative')
    
    pixel_shape = footprints.shape[1:]
    n_pixels = int(np.prod(pixel_shape))

    # find which pixels to include in interpolator
    inds_to_interp = np.flatnonzero(np.any(footprints != 0, axis=0))
    data_for_interp = footprints.reshape((len(zs), -1))[:, inds_to_interp]
    min_data_z, max_data_z = np.min(zs), np.max(zs)
    min_z = min_data_z - z_border
    max_z = max_data_z + z_border

    # augment data so that cubic interpolation works
    data_aug, zs_aug = augment_data_for_interpolation(   # make_smoothing_spline requires at least 5 points
        data_for_interp, zs, z_border=z_border, n_border_points=0, min_total_points=5)

    # sort and average any repeats
    z_sorted, inv_inds, counts = np.unique(zs_aug, return_inverse=True, return_counts=True)
    data_sorted = np.zeros((len(z_sorted), data_aug.shape[1]))
    for inv_ind, data in zip(inv_inds, data_aug):  # index into z_sorted and data_sorted for this footprint
        data_sorted[inv_ind] += data / counts[inv_ind]

    # make functions for interpolation
    interpolant = make_smoothing_spline(z_sorted, data_sorted, lam=lam)

    def interpolator(z: float) -> np.ndarray:
        output_vals = np.zeros(n_pixels)
        if z > min_data_z and z < max_data_z:
            output_vals[inds_to_interp] = interpolant(z)
        elif z > min_z and z <= min_data_z:
            minvals = interpolant(min_data_z)
            output_vals[inds_to_interp] = minvals * (z - min_z) / z_border
        elif z < max_z and z >= max_data_z:
            maxvals = interpolant(max_data_z)
            output_vals[inds_to_interp] = maxvals * (max_z - z) / z_border
        return np.clip(output_vals, min_val, max_val).reshape(pixel_shape)
    return interpolator


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
        for k_comp, (plane_rois, planes) in enumerate(zip(merged_ROIs, roi_planes)):
            for roi, plane in zip(plane_rois, planes):
                if plane == k_plane:
                    A_merged[:, k_comp] += A_flat[:, roi]  # type: ignore

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
        sessinfo.make_df_over_f()
        dff = sessinfo.cnmf_fit.estimates.F_dff_denoised
    assert dff is not None
    assert (A_orig := sessinfo.cnmf_fit.estimates.A) is not None
    A_orig = sparse.csc_array(A_orig)

    # subset based on which components are in both planes
    in_planes_inds = np.flatnonzero([all(np.isin(planes_to_merge, planes)) for planes in roi_planes])
    A0_merged = A0_merged[:, in_planes_inds]
    A1_merged = A1_merged[:, in_planes_inds]

    with mplstyle.context('dark_background'):  # type: ignore
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
