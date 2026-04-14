"""Customized versions of caiman functions (visualizations etc.)"""
from itertools import repeat
import logging
from typing import cast, Sequence, Union, Literal, Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import optype.numpy as onp
from scipy.ndimage import label
from scipy.sparse import csc_array, csc_matrix, lil_array
from scipy.stats import gamma, norm, loggamma
from skimage.morphology import remove_small_objects, remove_small_holes, dilation, closing

from caiman.components_evaluation import estimate_baseline
from caiman.source_extraction.cnmf import cnmf
from caiman.utils.visualization import get_contours


def my_plot_contours(est: cnmf.Estimates, img=None, idx=None, thr_method='max',
                     thr=0.2, display_numbers=True, params=None, cmap='viridis',
                     accept_color='m', reject_color='r', vmin=None, vmax=None,
                     ax: Optional[Axes] = None):
    """
    View contours of all spatial footprints. If idx is provided,
    plot accepted and rejected contours in different colors rather than on differen axes.

    Args:
        img :   np.ndarray
            background image for contour plotting. Default is the mean
            image of all spatial components (d1 x d2)
        idx :   list
            list of accepted components
        thr_method : str
            thresholding method for computing contours ('max', 'nrg')
            if list of coordinates self.coordinates is None, i.e. not already computed
        thr : float
            threshold value
            only effective if self.coordinates is None, i.e. not already computed
        display_numbers :   bool
            flag for displaying the id number of each contour
        params : params object
            set of dictionary containing the various parameters
    """
    assert est.dims is not None and est.A is not None, 'Estimates elements should not be None - has CNMF been run?'

    if 'csc_matrix' not in str(type(est.A)):
        A = csc_matrix(est.A)
    else:
        A = cast(csc_matrix, est.A)

    if img is None:
        img = np.reshape(np.array(A.mean(1)), est.dims, order='F')
    if est.coordinates is None:  # not hasattr(est, 'coordinates'):
        est.coordinates = my_get_contours(A, img.shape, thr=thr, thr_method=thr_method)
    
    if ax is None:
        fig, ax = plt.subplots()
        assert isinstance(ax, Axes)
    else:
        plt.sca(ax)

    if params is not None:
        fig.suptitle('min_SNR=%1.2f, rval_thr=%1.2f, use_cnn=%i'
                        %(params.quality['min_SNR'],
                        params.quality['rval_thr'],
                        int(params.quality['use_cnn'])))
    if idx is None:
        my_vis_plot_contours(A, img, coordinates=est.coordinates,
                             display_numbers=display_numbers,
                             cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if not isinstance(idx, list):
            idx = idx.tolist()
        coor_g = [est.coordinates[cr] for cr in idx]
        bad = list(set(range(len(est.coordinates))) - set(idx))
        coor_b = [est.coordinates[cr] for cr in bad]
        my_vis_plot_contours(A[:, idx], img,    # type: ignore
                             coordinates=coor_g,
                             display_numbers=display_numbers,
                             inds_for_numbers=idx,
                             colors=accept_color,
                             cmap=cmap, vmin=vmin, vmax=vmax)

        my_vis_plot_contours(A[:, bad], img,    # type: ignore
                             coordinates=coor_b,
                             display_numbers=display_numbers,
                             inds_for_numbers=bad,
                             colors=reject_color,
                             cmap=cmap, vmin=vmin, vmax=vmax)
        
        # remove second plotted image to unblock accepted components
        ims = ax.get_images()
        ims[1].remove()
    return plt.gcf()


def my_vis_plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                         cmap=None, colors='w', vmin=None, vmax=None, coordinates=None, inds_for_numbers=None,
                         contour_args={}, number_args={}, **kwargs):
    """
    Modifications from visualization.plot_contours:
     - A/Cn can be 3D; the resulting plot has planes concatenated along the X axis.
     - remove support for swap_dim as an option; A is assumed to be in F order (with z as last dimension if 3D)
    -----
    Plots contour of spatial components against a background image and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)
    
         Cn:  np.ndarray (2D or 3D)
                   Background image (e.g. mean, correlation)
    
         thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy
    
         maxthr: [optional] scalar
                    Threshold of max value
    
         nrgthr: [optional] scalar
                    Threshold of energy
    
         thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)
                   Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr
    
         display_number:     Boolean
                   Display number of ROIs if checked (default True)
    
         max_number:    int
                   Display the number for only the first max_number components (default None, display all numbers)

         inds_for_numbers: [optional] vector
                   What numbers to display (defaults to range(min(nr, max_number)) if None, else overrides max_number)
    
         cmap:     string
                   User specifies the colormap (default None, default colormap)

     Returns:
          coordinates: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    if thr is None:
        try:
            thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
        except KeyError:
            thr = maxthr
    else:
        thr_method = 'nrg'

    for key in ['c', 'colors', 'line_color']:
        if key in kwargs.keys():
            colors = kwargs[key]
            kwargs.pop(key)

    dims = np.shape(Cn)
    Cn_flat = np.reshape(Cn, (dims[0], -1), order='F')

    ax = plt.gca()
    if vmax is None and vmin is None:
        ax.imshow(Cn_flat, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
                  vmax=np.percentile(Cn[~np.isnan(Cn)], 99.96))
    else:
        ax.imshow(Cn_flat, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

    if coordinates is None:
        coordinates = my_get_contours(A, dims, thr=thr, thr_method=thr_method)

    if max_number is None:
        max_number = len(coordinates)
    if inds_for_numbers is None:
        inds_for_numbers = range(max_number)

    for num, c in zip(inds_for_numbers, coordinates):
        ax.plot(*c['coordinates'].T, c=colors, **contour_args)
        if display_numbers:
            ax.text(c['CoM'][1], c['CoM'][0], str(num), color=colors, clip_on=True, **number_args)

    return coordinates


def my_get_contours(A, dims, thr: Optional[float] = None, thr_method: Optional[str] = None):
    """
    Get contours and flatten the coordinates along x axis if 3D
    threshold defaults to 0.5/max if data is bool (i.e., contour around true pixels)
    else, 0.9/nrg.
    """
    if thr_method is None:
        thr_method = 'max' if A.dtype == bool else 'nrg'

    if thr is None:
        thr = 0.5 if (thr_method == 'max' and A.dtype == bool) else 0.9

    is3D = len(dims) == 3
    # need to reverse dims and use transpose of A here (swap_dim) if not on latest version of CaImAn
    try:
        coordinates = get_contours(A, dims, thr, thr_method, slice_dim=None)
        flipped = False
    except TypeError:
        coordinates = get_contours(A, dims[::-1], thr, thr_method, swap_dim=True)
        flipped = True

    for c in coordinates:
        if c['coordinates'].size == 0 or all(np.all(np.isnan(coords)) for coords in c['coordinates']):
            # empty contour
            c['coordinates'] = np.array([])
            c['CoM'] = np.array([np.nan, np.nan])
            c['bbox'] = [np.nan] * 4
            if is3D:
                c['coordinates_3d'] = [np.array([]) for _ in dims[2]]
            continue

        if is3D:
            c['coordinates_3d'] = [np.fliplr(v) if flipped else v for v in c['coordinates']]  # un-reverse dimensions (to X, Y)
            # flatten planes along X-axis, note there should already be NaN rows at beginning and end
            # so this should work without merging components
            c['coordinates'] = np.concatenate([v + np.array([[kplane * dims[1], 0]])
                                               for kplane, v in enumerate(c['coordinates_3d'])], axis=0)
            # flatten and un-reverse centers of mass (to Y, X)
            com_y, com_x, com_z = c['CoM'][::-1] if flipped else c['CoM']
            com_plane = round(com_z)
            c['CoM'] = np.array([com_y, com_x + com_plane * dims[1]])
        elif flipped:
            c['coordinates'] = np.fliplr(c['coordinates'])  # un-reverse dimensions (to X, Y)
            c['CoM'] = c['CoM'][::-1]

        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    return coordinates


def my_com(A, *dims: Union[int, Sequence, np.ndarray, slice], order: Literal['C', 'F'] = 'F') -> np.ndarray:
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K); column indices map to dimensions according to F-order.

         *dims: D ints or Sequences
              each argument after A defines the size/spacing of one of the dimensions of the matrix.
              an int is interpreted as the number of uniformly-spaced pixels in this dimension.
              a sequence (should be monotonicaly increasing) is interpreted as the location of each pixel in this dimension.
            
         order: 'C' or 'F'
              how each column of A should be reshaped to match dims.

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x D)
    """
    if A.ndim == 1:
        A = A[:, np.newaxis]

    if 'csc_array' not in str(type(A)):
        A = csc_array(A)

    # convert each 'dims' argument to a (possibly unevenly-spaced) range from 0 to npixels-1
    def toarr(dim: Union[int, Sequence, np.ndarray, slice]) -> np.ndarray:
        if isinstance(dim, slice):
            return np.arange(dim.start, dim.stop, dim.step)
        elif isinstance(dim, Sequence):
            return np.array(dim)
        elif isinstance(dim, np.ndarray):
            return dim
        else:
            return np.arange(dim)

    dim_ranges = [toarr(dim) for dim in dims]

    # make coordinate arrays where coor[d] increases from 0 to npixels[d]-1 along the dth axis
    coors = np.meshgrid(*dim_ranges, indexing='ij')
    coor = np.stack([c.ravel(order=order) for c in coors])

    # take weighted sum of pixel positions along each coordinate
    with np.errstate(invalid='ignore'):
        cm = (coor @ A / A.sum(axis=0)).T
    return np.array(cm)


def compute_matching_performance(n1: int, n2: int, n_matched: int) -> dict[str, float]:
    """
    Given a number of cells (or anything) in 2 sessions and the number that were matched,
    compute the following metrics:
    - recall == n_matched / n1
    - precision == n_matched / n2
    - accuracy == n_matched / total cells
    - f1_score = 2*n_matched / (n1 + n2) (including duplicates)
    """
    TP = n_matched
    FN = n1 - n_matched
    FP = n2 - n_matched
    TN = 0

    performance = dict()
    with np.errstate(divide='ignore', invalid='ignore'):  # allow division by 0
        performance['recall'] = np.array(TP) / (TP + FN)
        performance['precision'] = np.array(TP) / (TP + FP)
        performance['accuracy'] = np.array(TP + TN) / (TP + FP + FN + TN)
        performance['f1_score'] = np.array(2 * TP) / (2 * TP + FP + FN)
    return performance


def my_extract_binary_masks_from_structural_channel(Y: Union[onp.Array2D[np.floating], onp.Array3D[np.floating]],
                                                    blur_type: Literal['box', 'gaussian'] = 'gaussian',
                                                    blur_gSig_multiple: Optional[float] = None,
                                                    min_area_size: int = 30,
                                                    min_hole_size: int = 15,
                                                    gSig: Union[int, Sequence[int]] = 5,
                                                    expand_method: str = 'closing',
                                                    selem: np.ndarray = np.ones((3, 3))) -> tuple[csc_array[np.bool_], np.ndarray]:
    """
    Extract binary masks by using adaptive thresholding on a structural channel
    My version allows using multiple gSigs, where each subsequent iteration only considers areas not included in
    any ROIs from previous iterations.

    Args:
        Y:                  caiman movie object
                            movie of the structural channel (assumed motion corrected)
        
        blur_type:          'box' | 'gaussian'
                            type of blurring to use
        
        blur_gSig_multiple: float
                            what to multiply (each) gSig by to get blur sigma or size

        min_area_size:      int
                            ignore components with smaller size

        min_hole_size:      int
                            fill in holes up to that size (donuts)

        gSig:               int
                            average radius of cell

        expand_method:      string
                            method to expand binary masks (morphological closing or dilation)

        selem:              np.array
                            structuring element ('selem') with which to expand binary masks

    Returns:
        A:                  sparse column format matrix
                            matrix of binary masks to be used for CNMF seeding

        mR:                 np.ndarray
                            mean image used to detect cell boundaries
    """
    if isinstance(gSig, int):
        gSig = (gSig,)

    if blur_gSig_multiple is None:
        blur_gSig_multiple = 1 if blur_type == 'box' else 0.75

    if onp.is_array_3d(Y):
        mR = np.mean(Y, axis=0)
    else:
        mR = Y

    n_pix = np.prod(mR.shape).item()
    occupied = np.zeros(n_pix, dtype=bool)
    A = lil_array((0, n_pix), dtype=bool)
    features_added = 0

    for single_gSig in gSig:
        if blur_gSig_multiple > 0:
            blur_sig = blur_gSig_multiple * single_gSig
            img = np.empty_like(mR, order='C')
            if blur_type == 'box':
                blur_sig = round(blur_sig)
                cv2.blur(mR, (blur_sig, blur_sig), dst=img)
            else:
                cv2.GaussianBlur(mR, (0, 0), blur_sig, dst=img)
        else:
            img = np.copy(mR)

        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.
        img = img.astype(np.uint8)

        th = np.empty_like(img)
        cv2.adaptiveThreshold(img, float(np.max(img)), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, single_gSig, 0, dst=th)
        th = remove_small_holes(th > 0, area_threshold=min_hole_size)
        th: onp.Array2D[np.uint8] = remove_small_objects(th, min_size=min_area_size)
        labeled_array, num_features = label(th)

        for i in range(num_features):
            temp = (labeled_array == i + 1)
            if expand_method == 'dilation':
                temp = dilation(temp, footprint=selem)
            elif expand_method == 'closing':
                temp = closing(temp, footprint=selem)
            
            temp_flat = temp.flatten('F')
            # only add if these pixels are not already occuied by another feature
            if not np.any(occupied[temp_flat]):
                occupied[temp_flat] = True
                A.resize(features_added + 1, n_pix)
                A.getrowview(features_added)[:] = temp_flat  # type: ignore
                features_added += 1

    return A.tocsr().T, mR


def compute_snr_gamma(C: onp.Array2D, YrA: onp.Array2D, use_loggamma=True,
                      remove_baseline=True, N=5, sigma_factor=3., dview=None) -> np.ndarray:
    """
    Compute an SNR measure based on fitting a gamma distribution to the residuals (YrA)
    of each component, then using the gamma ppf to evaluate exceptionality of events
    in the full traces (C + YrA).
    See evaluate_components, compute_event_exceptionality in components_evaluation.py
    
    Args:
        C: ndarray
            denoised traces (output of CNMF)
        
        YrA: ndarray
            residuals (output of CNMF)
        
        use_loggamma: bool
            whether to fit loggamma function instead of gamma (avoid underflow w/ small shape param)
        
        remove_baseline: bool
            whether to remove the baseline from YrA in a rolling fashion (8th percentile)
        
        N: int
            N number of consecutive events probability multiplied
        
        sigma_factor: float
            multiplicative factor for spread of gamma distribution (higher = higher criterion)
    """
    logger = logging.getLogger('caiman')

    if N == 0:
        # Without this, numpy ranged syntax does not work correctly, and also N=0 is conceptually incoherent
        raise Exception("FATAL: N=0 is not a valid value for compute_event_exceptionality()")
    
    if remove_baseline:
        logger.debug('Removing baseline from YrA')
        YrA = YrA - estimate_baseline(YrA, YrA.shape[1], slow_baseline=False)

    logger.debug('Computing event exceptionality with gamma distribution')
    args = zip(C, YrA, repeat(sigma_factor))
    logsf = np.empty_like(C)
    logsf_fn = get_loggamma_logsfs if use_loggamma else get_gamma_logsfs
    if dview is None:
        for out, arg_tuple in zip(logsf, args):
            out[:] = logsf_fn(arg_tuple)
    else:
        logger.info('SNR calculation in parallel')
        map_fn = dview.map if 'multiprocessing' in str(type(dview)) else dview.map_sync
        for out, res in zip(logsf, map_fn(logsf_fn, args)):
            out[:] = res
    
    # find most improbable sequence of N events using moving sum
    logsf_cum = np.cumsum(logsf, 1)
    logsf_seq = logsf_cum[:, N:] - logsf_cum[:, :-N]

    # select the minimum value of log-probability for each trace
    fitness = np.nanmin(logsf_seq, 1)
    # technically maybe weird to use normal distribution here, but
    # not sure how to deal with shape parameter if using gamma and 
    # I think it makes sense to use the same transformation to SNR values as regular SNR
    comp_SNR = -norm.ppf(np.exp(fitness / N))
    return comp_SNR


def get_gamma_logsfs(args: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    """
    Helper to compute the log of survival function (called "erf" in compute_event_exceptionality)
    for each sample in C_one + Yr_one based on fitting a gamma distribution to Yr_one.
    Inputs are vectors of C and YrA for a single ROI.
    """
    C_one, YrA_one, sigma_factor = args
    a, loc, scale = gamma.fit(YrA_one)
    scale *= sigma_factor  # artificially increase probabilities
    return gamma.logsf(C_one + YrA_one, a=a, loc=loc, scale=scale)


def get_loggamma_logsfs(args: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    """
    Like get_gamma_logsfs, but uses loggamma distribution instead.
    According to Wikipedia (https://en.wikipedia.org/wiki/Gamma_distribution#Caveat_for_small_shape_parameter),
    this can help avoid underflow.
    """
    C_one, YrA_one, sigma_factor = args
    # offset YrA so that it is in range, then take log
    # First scale by sigma_factor
    YrA_scaled = YrA_one * sigma_factor
    offset = 1 - np.min(YrA_scaled)
    YrA_offset = YrA_scaled + offset
    YrA_log = np.log(YrA_offset)

    # fit loggamma distribution
    params = loggamma.fit(YrA_log)

    # now apply same transformation (except scaling) to C + YrA and get survival function values
    CYrA_offset = YrA_one + C_one + offset
    CYrA_log = np.log(CYrA_offset)
    return loggamma.logsf(CYrA_log, *params)
