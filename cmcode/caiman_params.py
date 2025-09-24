from typing import Optional, Literal, Union

import numpy as np
from caiman.source_extraction.cnmf import params


quality_params = {
    'normal': {
        'SNR_lowest': 2,
        'cnn_lowest': 0.1,
        'gSig_range': [[7, 7], [8, 8], [9, 9]],
        'min_SNR': 2.6,
        'min_cnn_thr': 0.99,
        'rval_lowest': 0,
        'rval_thr': 0.8,
        'use_cnn': False,
        'use_ecc': False,
        'max_ecc': 3
    },
    'gamma': {
        'SNR_lowest': 0.5,
        'min_SNR': 1.2,
        'rval_lowest': 0.1,
        'rval_thr': 0.85,
        'cnn_lowest': 0.1,
        'min_cnn_thr': 0.99,
        'gSig_range': [[7, 7], [8, 8], [9, 9]],
        'use_cnn': False,
        'use_ecc': False,
        'max_ecc': 3
    }
}


def get_dxy_and_scale(metadata: dict):
    """
    Get um per pixel in X and Y and zoom/scale relative to 
    the reference magnification of 1.2x.
    """
    reference_um_per_pixel = (1.0825, 1.1394)  # for 1.2x magnification
    if 'um_per_pixel_x' in metadata:
        dxy = (metadata['um_per_pixel_x'], metadata['um_per_pixel_y'])
    else:
        dxy = reference_um_per_pixel  # from before I set it to be saved in the mat file

    scalex = reference_um_per_pixel[0] / dxy[0]  # ratio of pixel size
    scaley = reference_um_per_pixel[1] / dxy[1]
    scale = (scalex + scaley) / 2

    return dxy, scale


def round_to_odd(x: Union[float, np.ndarray]):
    if isinstance(x, np.ndarray):
        return np.vectorize(round_to_odd)(x)
    
    return round((x - 1) / 2) * 2 + 1


def make_cnmf_params(metadata: dict, dims: int, tif_file: Optional[str] = None,
                     snr_type: Literal['normal', 'gamma'] = 'gamma', downsample_factor: Optional[int] = None) -> params.CNMFParams:
    p = 1     # order of the autoregressive system (set p=2 if there is visible rise time in data)
    dxy, scale = get_dxy_and_scale(metadata)    
    
    if downsample_factor is None:
        downsample_factor = 1

    params_dict: dict[str, dict] = {
        # general dataset-dependent parameters
        'data': {
            'fnames': [tif_file] if tif_file is not None else None,
            'fr': metadata['frame_rate'] / downsample_factor,  # imaging rate in frames per second
            # 'decay_time': 0.4,            # length of a typical transient in seconds
            'decay_time': 0.2,                         # for jGCaMP8f
            'dxy': dxy
        },
        
        # motion correction parameters
        'motion': {
            # start a new patch for pw-rigid motion correction every x pixels
            'strides': (48, 48, metadata['num_planes'])[:dims],
            'overlaps': (24, 24, 0)[:dims],     # overlap between patches (width of patch = strides+overlaps)
            'max_shifts': (round(15 * scale), round(15 * scale), 0)[:dims],   # maximum allowed rigid shifts (in pixels)
            'max_deviation_rigid': round(10 * scale),  # maximum shifts deviation allowed for patch with respect to rigid shifts
            'pw_rigid': True,                   # flag for performing non-rigid motion correction
            'is3D': dims == 3,
            'indices': (slice(None),) * dims,
            'border_nan': 'copy'
        },
        
        'preprocess': {
            'p': p
        },
        
        'init': {
            'nb': 1,                      # number of global background components (set to 1 or 2) 11/6/24: seems that 1 works fine and is faster
            'K': 10,                      # number of components per patch
            'gSig': round_to_odd(np.array([9, 9]) * scale),     # expected half-width of neurons in pixels (Gaussian kernel standard deviation)
            'method_init': 'greedy_roi',  # initialization method (if analyzing dendritic data see demo_dendritic.ipynb)
            'ssub': 1,                    # spatial subsampling during initialization 
            'tsub': 1,                    # temporal subsampling during intialization
        },

        'spatial': {
            'nrgthr': 0.9
        },
        
        'temporal': {
            'p': p,
            'bas_nonneg': True      # enforce nonnegativity constraint on calcium traces (technically on baseline)
        },
        
        'merging': {
            'merge_thr': 0.8             # merging threshold, max correlation allowed
        },
        
        'patch': {
            # 'rf': 22,      # half-size of the patches in pixels (patch width is rf*2 + 1)
            # 'stride': 15   # amount of overlap between the patches in pixels (overlap is stride+1) 
            'rf': 60,
            'stride': 40
        },

        # parameters for component evaluation
        'quality': quality_params[snr_type]
    }

    params_dict['init']['gSiz'] = 2*params_dict['init']['gSig'] + 1           # Gaussian kernel width and height
    return params.CNMFParams(params_dict=params_dict) # CNMFParams is the parameters class


def get_default_seed_params(metadata: dict) -> dict:
    """Default seed parameters; use empty dict for non-seeded CNMF"""
    _, scale = get_dxy_and_scale(metadata)
    return {
        'type': 'mean',
        'norm_medw': 25,
        'gSig': np.unique(round_to_odd(np.array([5, 7, 9]) * scale))
    }