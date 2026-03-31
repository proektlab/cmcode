"""Wrappers for CNMF and Estimates classes"""
from copy import copy
from dataclasses import dataclass, asdict
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Literal, Union

import numpy as np
from numpy.typing import NDArray
import optype.numpy as onp
from pandas import NA
from scipy import sparse
from scipy.interpolate import PchipInterpolator, interp1d
from caiman.source_extraction.cnmf import cnmf, params, merging

from cmcode import alignment, caiman_params as cmp
from cmcode.cmcustom import compute_snr_gamma
from cmcode.util import paths, types

@dataclass
class MetricInfo:
    name: str
    vals: Optional[NDArray[np.floating]]
    accept_thresh_name: str
    min_thresh_name: str

class EstimatesExt(cnmf.Estimates):
    """
    Estimates object that allows access to added fields
    I wrote this in a weird way that behaves like inheritance externally but is actually composition
    to satisfy static type checkers while keeping the base Estimates object available
    """
    __slots__ = ('estimates', 'accepted_list', 'rejected_list', 'structural_reg_res',
                 'F_dff_denoised', 'snr_type', 'snr_gamma_vals', 'crossplane_merge_thr_used')

    def __new__(cls, _: cnmf.Estimates):
        self = object.__new__(cls)  # don't actually create an estimates object under the hood
        return self
    
    def __init__(self, base_obj: cnmf.Estimates):
        self.estimates = base_obj
        # use getattr to copy fields from base_obj that might have been saved alongside Estimates fields
        if hasattr(base_obj, 'Cn'):
            logging.info('Ignoring saved correlation image (deprecated)')

        self.accepted_list: onp.Array1D[np.integer] = getattr(base_obj, 'accepted_list', np.array([], dtype=int))
        self.rejected_list: onp.Array1D[np.integer] = getattr(base_obj, 'rejected_list', np.array([], dtype=int))
        self.structural_reg_res: Optional[alignment.RegisterROIsResults] = getattr(base_obj, 'structural_reg_res', None)
        self.F_dff_denoised: Optional[np.ndarray] = getattr(base_obj, 'F_dff_denoised', None)
        self.snr_type: Literal['normal', 'gamma'] = getattr(base_obj, 'snr_type', 'normal')
        self.snr_gamma_vals: Optional[np.ndarray] = getattr(base_obj, 'snr_gamma_vals', None)
        self.crossplane_merge_thr_used: Optional[float] = getattr(base_obj, 'crossplane_merge_thr_used', None)
    
    def __getattribute__(self, name):
        """defer to estimates object"""
        if hasattr(EstimatesExt, name):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self.estimates, name)

    def __setattr__(self, name, val):
        """defer to estimates object"""
        if hasattr(EstimatesExt, name):
            object.__setattr__(self, name, val)
        else:
            object.__setattr__(self.estimates, name, val)

    @property
    def A(self) -> Optional[types.MaybeSparse[np.floating]]:
        """Just provide more precise type for estimates.A"""
        return self.estimates.A  # type: ignore

    @A.setter
    def A(self, val):
        self.estimates.A = val

    @property
    def C(self) -> Optional[onp.Array2D[np.floating]]:
        """Just provide more precise type for estimates.C"""
        return self.estimates.C  # type: ignore

    @C.setter
    def C(self, val):
        self.estimates.C = val

    @property
    def idx_components(self) -> Optional[onp.Array1D[np.integer]]:
        """Combines automatic evaluation and manual curation to produce idx_components"""
        if self.idx_components_eval is None:
            return None
        return np.setdiff1d(np.union1d(self.accepted_list, self.idx_components_eval), self.rejected_list)

    @idx_components.setter
    def idx_components(self, _):
        raise NotImplementedError('idx_components is a read-only property on EstimatesExt; use idx_components_eval')
    
    @property
    def idx_components_bad(self) -> Optional[onp.Array1D[np.integer]]:
        """Combines automatic evaluation and manual curation to produce idx_components_bad"""
        if self.idx_components_bad_eval is None:
            return None
        return np.setdiff1d(np.union1d(self.rejected_list, self.idx_components_bad_eval), self.accepted_list)
    
    @idx_components_bad.setter
    def idx_components_bad(self, _):
        raise NotImplementedError('idx_components_bad is a read-only property on EstimatesExt; use idx_components_bad_eval')

    @property
    def idx_components_eval(self) -> Optional[onp.Array1D[np.integer]]:
        return self.estimates.idx_components
    
    @idx_components_eval.setter
    def idx_components_eval(self, val):
        self.estimates.idx_components = val
    
    @property
    def idx_components_bad_eval(self) -> Optional[onp.Array1D[np.integer]]:
        return self.estimates.idx_components_bad
    
    @idx_components_bad_eval.setter
    def idx_components_bad_eval(self, val):
        self.estimates.idx_components_bad = val

    @property
    def structural_reg_idx_used(self) -> Optional[np.ndarray]:
        if self.structural_reg_res is None or self.A is None:
            return None
        elif self.structural_reg_res.components_used is not None:
            return self.structural_reg_res.components_used
        else:
            return np.arange(self.A.shape[1])

    @property
    def idx_components_marked(self) -> Optional[onp.Array1D[np.integer]]:
        comps_used = self.structural_reg_idx_used
        if comps_used is None or self.structural_reg_res is None:
            return None
        return comps_used[self.structural_reg_res.matched1]
    
    @property
    def idx_components_unmarked(self) -> Optional[onp.Array1D[np.integer]]:
        comps_used = self.structural_reg_idx_used
        if comps_used is None or self.structural_reg_res is None:
            return None
        return comps_used[self.structural_reg_res.unmatched1]


    # richer information about metrics
    @property
    def SNR_comp(self) -> Optional[np.ndarray]:
        """In EstimatesExt, this points to either the normal or gamma distribution-based SNR"""
        if self.snr_type == 'normal':
            return self.estimates.SNR_comp  # type: ignore
        else:
            return self.snr_gamma_vals

    @property
    def snr(self) -> MetricInfo:
        return MetricInfo(
            name=f'SNR ({self.snr_type})',
            vals=self.SNR_comp,
            accept_thresh_name='min_SNR',
            min_thresh_name='SNR_lowest'
        )
    
    @property
    def spatial_corr(self) -> MetricInfo:
        assert isinstance(self.estimates.r_values, Optional[np.ndarray])
        return MetricInfo(
            name='Spatial correlation (r)',
            vals=self.estimates.r_values,
            accept_thresh_name='rval_thr',
            min_thresh_name='rval_lowest'
        )
    
    @property
    def cnn_score(self) -> MetricInfo:
        cnn_preds = self.estimates.cnn_preds
        if isinstance(cnn_preds, list):
            if len(cnn_preds) == 0:
                cnn_preds = None  # occurs if use_cnn is false
            else:
                cnn_preds = np.array(cnn_preds)
        assert isinstance(cnn_preds, (Optional[np.ndarray]))
        return MetricInfo(
            name='CNN score',
            vals=cnn_preds,
            accept_thresh_name='min_cnn_thr',
            min_thresh_name='cnn_lowest'
        )

    
    def populate_snr_gamma(self, params: params.CNMFParams, use_loggamma=True, recalc=True, dview=None):
        if recalc or self.snr_gamma_vals is None:
            if self.C is None or self.YrA is None:
                raise RuntimeError('CNMF not done')
            frate = params.data['fr']
            decay_time = params.data['decay_time']
            N = np.ceil(frate * decay_time).astype(int)
            self.snr_gamma_vals = compute_snr_gamma(self.C, self.YrA, use_loggamma=use_loggamma, N=N, dview=dview)

    def evaluate_components(self, imgs: np.ndarray, params: params.CNMFParams, dview=None):
        """do normal evaluate_components on base object, then add gamma SNR and re-filter if needed"""
        self.estimates.evaluate_components(imgs, params, dview=dview)
        if self.snr_type == 'gamma':
            self.populate_snr_gamma(params, recalc=True, dview=dview)
            self.filter_components(imgs, params, dview=dview)
    
    def filter_components(self, imgs: np.ndarray, params: params.CNMFParams, new_dict={}, dview=None):
        """Overrides Estimates.filter_components to not recompute cnn_preds unless necessary"""
        curr_cnn_preds: Optional[np.ndarray] = self.cnn_preds  # type: ignore
        if curr_cnn_preds is None:
            # imitate estimates.filter_components by using an empty array
            curr_cnn_preds = np.array([])

        using_cnn = ('use_cnn' in new_dict and new_dict['use_cnn']) or (
            'use_cnn' not in new_dict and params.quality['use_cnn'])
        gsig_range_changed = 'gSig_range' in new_dict and new_dict['gSig_range'] != params.quality['gSig_range']

        # whether we can reuse the CNN results without recalculating
        faking_no_cnn = (using_cnn and not gsig_range_changed and
                         curr_cnn_preds is not None and len(curr_cnn_preds) > 0)
        if not faking_no_cnn:
            # either not using the CNN or using it differently than before
            # no disadvantage to using the original method, but restore cnn preds afterwards
            self.estimates.filter_components(imgs, params, new_dict, dview)
            if not using_cnn:
                self.cnn_preds = curr_cnn_preds
        else:        
            # keep our cnn_preds and then first run filter_components without cnn, then incorporate CNN info
            self.estimates.filter_components(imgs, params, {**new_dict, 'use_cnn': False}, dview)
            self.cnn_preds = curr_cnn_preds
            params.change_params({'quality': {'use_cnn': True}})

        if faking_no_cnn or self.snr_type == 'gamma':
            # update idx_components and idx_components_bad
            if self.snr_type == 'gamma' and self.snr_gamma_vals is None:
                self.populate_snr_gamma(params, dview=dview)

            is_good = ((self.r_values >= params.quality[self.spatial_corr.accept_thresh_name]) |
                       (self.SNR_comp >= params.quality[self.snr.accept_thresh_name]))
            if using_cnn:
                is_good |= (self.cnn_preds >= params.quality[self.cnn_score.accept_thresh_name])
            
            is_bad = ((self.r_values <= params.quality[self.spatial_corr.min_thresh_name]) |
                      (self.SNR_comp <= params.quality[self.snr.min_thresh_name]))
            if using_cnn:
                is_bad |= self.cnn_preds <= params.quality[self.cnn_score.min_thresh_name]

            b_keep = is_good & ~is_bad
            self.estimates.idx_components = np.flatnonzero(b_keep)
            self.estimates.idx_components_bad = np.flatnonzero(~b_keep)
        return self
    
    def interpolate_t(self, upsample_factor: int, frames_per_trial: np.ndarray, method='pchip'):
        """
        Interpolate each variable in the time dimension to upsample by upsample_factor
        Do separately for each trial; the final length of each trial is given in frames_per_trial.
        """
        if self.C is None:
            raise RuntimeError('CNMF not run?')

        # infer interpolation information for each trial
        total_frames = sum(frames_per_trial)
        total_frames_ds = self.C.shape[1]
        if total_frames == total_frames_ds:
            logging.info('Estimates have already been interpolated')
            return

        frames_per_trial_ds = [math.ceil(frames / upsample_factor) for frames in frames_per_trial]
        if sum(frames_per_trial_ds) != total_frames_ds:
            raise RuntimeError('Number of frames in results does not match what is expected from downsampling')
        
        split_indices = np.cumsum(frames_per_trial[:-1])
        split_indices_ds = np.cumsum(frames_per_trial_ds[:-1])
        
        # create replacement for each variable with time dimension
        def interpolate_each_trial(var: onp.Array2D[np.floating], method, axis=1) -> onp.Array2D[np.floating]:
            var_out = np.empty((var.shape[0], total_frames), dtype=var.dtype)
            var_trials = np.split(var, split_indices_ds, axis=axis)
            var_out_trials = np.split(var_out, split_indices, axis=axis)

            for trial_in, trial_out in zip(var_trials, var_out_trials):
                xin = range(0, upsample_factor*trial_in.shape[axis], upsample_factor)
                xout = range(trial_out.shape[axis])
                if method == 'zero_fill':
                    trial_out[:] = 0
                    indices = [slice(None)] * trial_out.ndim
                    indices[axis] = slice(xin.start, xin.stop, xin.step)
                    trial_out[tuple(indices)] = trial_in
                elif method == 'pchip':
                    trial_out[:] = PchipInterpolator(xin, trial_in, axis=axis, extrapolate=True)(xout)
                else:
                    trial_out[:] = interp1d(xin, trial_in, kind=method, axis=axis,
                                            fill_value='extrapolate')(xout)  # type: ignore
            return var_out

        self.C = interpolate_each_trial(self.C, method=method)
        if self.f is not None:
            self.f = interpolate_each_trial(self.f, method=method)
        if self.YrA is not None:
            self.YrA = interpolate_each_trial(self.YrA, method=method)
            self.R = self.YrA
        if self.S is not None:
            # spikes is different, interpolate with 0s since we don't want to create new events
            self.S = interpolate_each_trial(self.S, method='zero_fill')
        if self.F_dff is not None:
            self.F_dff = interpolate_each_trial(self.F_dff, method=method)
        if self.F_dff_denoised is not None:
            self.F_dff_denoised = interpolate_each_trial(self.F_dff_denoised, method=method)
    
    def merge_components_crossplane(self, n_planes: int, params: params.CNMFParams, thr=0.7) -> int:
        """
        Merge components across planes as if they were in the same plane, by flattening A
        Returns number of sets that were merged.
        """
        if self.crossplane_merge_thr_used is not None:
            raise RuntimeError(f'Crossplane merging already run with threshold {self.crossplane_merge_thr_used}; cannot re-run.')
        
        self.crossplane_merge_thr_used = thr

        est = self.estimates
        assert est.A is not None and self.C is not None and est.R is not None and est.S is not None, 'CNMF not run?'
        A = est.A
        if not isinstance(A, sparse.csc_matrix):
            A = sparse.csc_matrix(A)
        plane_pixels = A.shape[0] // n_planes
        plane_per_comp = A.indices[A.indptr[:-1]] // plane_pixels  # assumes each comp is just in one plane
        A_flat = sparse.csc_matrix((A.data, A.indices % plane_pixels, A.indptr), shape=(plane_pixels, A.shape[1]))

        # find which components to merge
        rois_to_merge = merging.get_ROIs_to_merge(A_flat, self.C, thr=thr)[0]

        # only keep groups that span more than one plane
        rois_to_merge = [group for group in rois_to_merge if not np.all(np.diff(plane_per_comp[group]) == 0)]

        n_to_merge = len(rois_to_merge)
        if n_to_merge > 0:
            # now merge the original components using manual_merge
            self.estimates.manual_merge(rois_to_merge, params=params)

            # invalidate evaluation stuff which will have to be re-run
            for field in ['F_dff', 'SNR_comp', 'r_values', 'cnn_preds', 'idx_components', 'idx_components_bad']:
                setattr(self.estimates, field, None)
            
            self.F_dff_denoised = None
            self.snr_gamma_vals = None
            self.accepted_list = np.array([], dtype=int)
            self.rejected_list = np.array([], dtype=int)
        return n_to_merge


class CNMFExt(cnmf.CNMF):
    """CNMF object with EstimatesExt along with Estimates"""
    def __init__(self, *args, copy_from: Optional[cnmf.CNMF] = None, **kwargs):
        if copy_from is not None:
            # just copy from existing CNMF object
            for key, val in copy_from.__dict__.items():
                if key == 'estimates':
                    self.estimates_base = val
                else:
                    setattr(self, key, val)
            self.estimates = EstimatesExt(self.estimates_base)

            # if estimates_ext attr exists, populate information in estimates
            ext_data: Optional[dict] = getattr(self, 'estimates_ext', None)
            if ext_data is not None:
                for name, val in ext_data.items():
                    # deal with sparse matrices as special cases
                    if name == 'structural_reg_res' and isinstance(val, dict):
                        for Akey in ['A1', 'A2', 'A2_orig']:
                            if Akey in val and isinstance(A := val[Akey], dict):
                                csc_mat = sparse.csc_matrix((A['data'], A['indices'], A['indptr']), A['shape'])
                                val[Akey] = csc_mat
                        val = alignment.RegisterROIsResults(**val)

                    setattr(self.estimates, name, val)
            if hasattr(self, 'estimates_ext'):
                delattr(self, 'estimates_ext')
        else:
            super().__init__(*args, **kwargs)
            # make it so that 'estimates' gets the extended object, but allow using the base type when necessary
            # this finagling is mostly so that filter_components and related functions don't break
            self.estimates_base = self.estimates
            self.estimates = EstimatesExt(self.estimates_base)
    
    def _to_base_CNMF(self) -> cnmf.CNMF:
        """
        Make a proxy object where estimates is an instance of Estimates (i.e. idx_components can be assigned to)
        Warning, re-assigning fields of this proxy object will not update the calling instance.
        """
        cnmf_obj = copy(self)
        cnmf_obj.estimates = self.estimates_base
        del cnmf_obj.estimates_base
        return cnmf_obj
    
    @classmethod
    def create_fit(cls, images, indices=(slice(None), slice(None)), **construct_args) -> 'CNMFExt':
        """Create and fit CNMFExt in one operation (avoids trickiness with copy from to_base_CNMF)"""
        self = cls(**construct_args)
        base_res = self._to_base_CNMF().fit(images, indices=indices)
        return CNMFExt(copy_from=base_res)
    
    def refit(self, images, dview=None) -> 'CNMFExt':
        base_res = self._to_base_CNMF().refit(images, dview=dview)
        ext_res = CNMFExt(copy_from=base_res)
        return ext_res

    def save(self, filename: str, safe: bool = True):
        """
        Save in a way that is compatible with CNMF.save (which does not recognize EstimatesExt)
        If safe is true (default) and the file already exists, saves to temp file first and then moves.
        """
        if '.hdf5' not in filename:
            raise Exception('File extension not supported for cnmf.save')

        obj_dict = self.__dict__.copy()
        obj_dict['estimates'] = obj_dict['estimates_base']
        del obj_dict['estimates_base']

        obj_dict['estimates_ext'] = {
            key: getattr(self.estimates, key)
            for key in EstimatesExt.__slots__
            if key != 'estimates'
        }

        if isinstance(srr := obj_dict['estimates_ext']['structural_reg_res'], alignment.RegisterROIsResults):
            # convert to dict so that it can be saved to HDF5
            obj_dict['estimates_ext']['structural_reg_res'] = {
                k: v for (k, v) in asdict(srr).items() if v is not NA
            }

        if safe and os.path.exists(filename):
            # save to temp file and then move
            save_dir = os.path.split(filename)[0]
            with NamedTemporaryFile(dir=save_dir, delete=False) as tempf:
                try:
                    cnmf.save_dict_to_hdf5(obj_dict, tempf)  # type: ignore
                    try:
                        os.remove(filename)
                    except FileNotFoundError:
                        pass
                    os.rename(tempf.name, filename)  # should work b/c on same filesystem
                except Exception as e:
                    raise RuntimeError('Saving failed; may exist as temp file at ' + tempf.name) from e
        else:
            cnmf.save_dict_to_hdf5(obj_dict, filename)


def load_CNMFExt(filename: Union[Path, str], dview=None, quiet=True) -> CNMFExt:
    logger = logging.getLogger('caiman')
    old_level = logger.level
    if quiet:
        logger.setLevel(logging.WARNING)
    cnmf_obj_ext = _load_CNMFExt(str(filename))
    if quiet:
        logger.setLevel(old_level)

    if dview is not None:
        cnmf_obj_ext.dview = dview

    return cnmf_obj_ext


@lru_cache(maxsize=10)
def _load_CNMFExt(filename: str) -> CNMFExt:
    """Cached version of load_CNMFExt"""
    cnmf_obj = cnmf.load_CNMF(filename)
    cnmf_obj_ext = CNMFExt(copy_from=cnmf_obj)

    # to account for not setting crossplane_merge_thr_used previously, set it here if needed
    if hasattr(cnmf_obj, 'estimates_ext') and 'crossplane_merge_thr_used' not in getattr(cnmf_obj, 'estimates_ext'):
        params_path = paths.params_file_for_result(filename)
        try:
            saved_params = cmp.UpToEvalParamStruct.read_from_file(params_path)
        except FileNotFoundError:
            pass
        else:
            cnmf_obj_ext.estimates.crossplane_merge_thr_used = saved_params.cnmf_extra.crossplane_merge_thr

    return cnmf_obj_ext


def clear_cnmf_cache():
    _load_CNMFExt.cache_clear()
