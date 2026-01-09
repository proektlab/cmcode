from abc import ABC, abstractmethod
from copy import copy, deepcopy
from datetime import date
from enum import IntEnum
from itertools import pairwise
import os
from pathlib import Path

from typing import (Iterable, Sequence, Optional, Literal, Protocol, overload, cast,
                    Union, Any, Type, TypedDict, Mapping, TypeVar)

import msgspec
from msgspec.structs import fields, replace
import numpy as np
from caiman.source_extraction.cnmf import params
from mesmerize_core.utils import get_params_diffs

from cmcode.util.image import BorderSpec


# parameters to not show in diffs between CNMF runs
EXCLUDE_FROM_DIFFS = [
    'online.path_to_model',  # depends on caiman data dir
    'online.init_batch',
    'online.movie_name_online',
    'preprocess.n_pixels_per_process',
    'spatial.n_pixels_per_process',
    'patch.n_processes',
    'data.caiman_version',  # could be relevant at some point, but don't show for now
    'data.last_commit',
    'data.fnames',
    'quality'
]


# for parameter serialization:

def enc_hook(obj: Any) -> Any:
    """
    Extend msgspec encoding to work with additional types
    Based on caiman.source_extraction.cnmf.params.CNMFParams.to_json.NumpyEncoder,
    so it should produce json files compatible with the caiman functions.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, slice):
        return [obj.start, obj.stop, obj.step]
    else:
        raise NotImplementedError(f'Encoding objects of type {type(obj)} is not supported')

def dec_hook(decl_type: Type, obj: Any) -> Any:
    """Extend msgspec decoding to work with additional types"""
    if decl_type is np.ndarray:
        return np.array(obj)
    elif issubclass(decl_type, np.integer) or issubclass(decl_type, np.floating):
        return decl_type(obj)
    elif decl_type is slice:
        start, stop, step = obj
        return slice(start, stop, step)
    else:
        raise NotImplementedError(f'Decoding objects of type {decl_type} is not supported')

param_encoder = msgspec.json.Encoder(enc_hook=enc_hook)


Self = TypeVar('Self', bound='StageParams')

class StageParams(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """
    msgspec.Struct holding params for a particular analysis stage (abstract)
    This is like a dataclass, but it is faster to use with msgspec, and also
    has the feature forbid_unknown_fields which raises an error if we try to 
    deserialize an instance with unknonwn param names (like due to typos)
    """
    def get_differing_params(self: Self, other: Self, metadata: dict[str, Any]) -> Iterable[str]:
        yield from ()
        """Find names of parameters that are different between self and other"""

    def matches(self: Self, other: Self, metadata: dict[str, Any]) -> bool:
        """Find whether any parameters differ between self and other"""
        return not any(self.get_differing_params(other, metadata=metadata))
    
    def replace(self: Self, replacements: Mapping[str, Any]) -> Self:
        """Make copy with replacements; like dataclasses.replace but recursing on other StageParams"""
        new_args: dict[str, Any] = {}
        for key, val in replacements.items():
            if not hasattr(self, key):
                raise KeyError(f'Cannot replace unknown key {key}')

            my_val = getattr(self, key)
            if isinstance(my_val, StageParams) and isinstance(val, Mapping):
                new_args[key] = my_val.replace(val)
            else:
                # possibly do delayed conversion from dict format
                new_args[key] = msgspec.convert(val, type=self.__annotations__[key], dec_hook=dec_hook)
        return replace(self, **new_args)


# note: in the below params classes, defaults reflect the original values of each parameter/the
# value used if it is left unspecified when calling subroutines,
# which is not necessarily the same as the default when creating a new SessionAnalysis object.

# TODO: add stage params for file discovery (START stage) including trials_to_exclude, raw_dir, etc.


class ConversionParams(StageParams, frozen=True):
    """Settings for convert_to_tif"""
    crop: BorderSpec = BorderSpec()           # borders to crop out of image
    downsample_factor: Optional[int] = None
    keep_3d: bool = False

    # arguments to sbx_chain_to_tif
    channel: int = 0
    odd_row_offset: Optional[int] = None      # horizontal pixel offset of odd rows to correct
    # number of saturated pixels at the left of odd rows for each file (before any cropping)
    odd_row_ndead: Optional[list[int]] = None
    bigtiff: bool = True
    imagej: bool = False
    to32: Optional[bool] = None
    chunk_size: Optional[int] = 100
    force_estim_ndead_offset: bool = False
    interp: bool = True
    dead_pix_mode: Union[bool, Literal['copy', 'min']] = 'copy'

    def get_differing_params(self, other: 'ConversionParams', metadata: dict[str, Any]) -> Iterable[str]:
        """Determine whether loaded params are compatible with current ones"""
        # build list of fields that don't matter for outputs with current settings (or are checked separately)
        irrelevant_fields = ['chunk_size', 'odd_row_offset', 'odd_row_ndead', 'force_estim_ndead_offset']

        other_max_ndead = None if other.odd_row_ndead is None else max(other.odd_row_ndead)
        self_max_ndead = None if self.odd_row_ndead is None else max(self.odd_row_ndead)
        other_offset = other.odd_row_offset
        self_offset = self.odd_row_offset

        if metadata['scanning_mode'] == 'unidirectional':
            # set odd_row_offset and odd_row_ndead as they are in the actual function
            if not other.force_estim_ndead_offset:
                if other_max_ndead is None:
                    other_max_ndead = 0
                if other_offset is None:
                    other_offset = 0
            
            if not self.force_estim_ndead_offset:
                if self_max_ndead is None:
                    self_max_ndead = 0
                if self_offset is None:
                    self_offset = 0
        
        if self_max_ndead == 0 and other_max_ndead == 0 and self_offset == 0 and other_offset == 0:
            # no correction to be done, can ignore dead pixel fields
            irrelevant_fields.extend(['interp', 'dead_pix_mode'])

        # if either is None, be permissive, assume that the other was correctly estimated
        if self_max_ndead is not None and other_max_ndead is not None and self_max_ndead != other_max_ndead:
            yield 'odd_row_ndead'

        if self_offset is not None and other_offset is not None and self_offset != other_offset:
            yield 'odd_row_offset'

        for field in fields(other):
            if field.name in irrelevant_fields:
                continue

            # use array_equal to compare without caring about e.g. the sequence type
            if not np.array_equal(getattr(other, field.name), getattr(self, field.name)):
                yield field.name


class McorrParamsExtra(StageParams, frozen=True):
    """Extra params for the motion correction step"""
    # whether to automatically adjust indices to take odd_row_ndead, odd_row_offset and crop into account
    auto_adjust_indices: bool = True
    # flag that is set when indices are adjusted and unset when parames affecting them are changed
    _indices_are_adjusted: bool = False

    def get_differing_params(self, other: 'McorrParamsExtra', metadata: dict[str, Any]) -> Iterable[str]:
        # only care about auto_adjust_indices if indices are out of date, meaning that
        # matching motion.indices can't be trusted.
        if self.auto_adjust_indices and not other.auto_adjust_indices and not self._indices_are_adjusted:
            yield 'auto_adjust_indices'
        elif other.auto_adjust_indices and not self.auto_adjust_indices and not other._indices_are_adjusted:
            yield 'auto_adjust_indices'


class TranspositionParams(StageParams, frozen=True):
    """Settings for transpose/concat planes"""
    highpass_cutoff: float = 0
    highpass_order: int = 4
    add_to_mov: float = 0
    blur_kernel_size: int = 1  # same as blur_size in SeedParams

    def get_differing_params(self, other: 'TranspositionParams', metadata: dict[str, Any]) -> Iterable[str]:
        irrelevant_fields = ['conversion_params', 'highpass_cutoff']
        if self.highpass_cutoff != other.highpass_cutoff:
            yield 'highpass_cutoff'
        
        if self.highpass_cutoff == 0:
            irrelevant_fields.append('highpass_order')

        for field in fields(self):
            if field.name in irrelevant_fields:
                continue

            # use array_equal to compare without caring about e.g. the sequence type
            if not np.array_equal(getattr(other, field.name), getattr(self, field.name)):
                yield field.name


class SeedParams(StageParams, frozen=True):
    """Params for producing spatial seed"""
    type: str = 'none' # e.g. "mean", "skew"; "none" means don't use a seed at all
    norm_medw: Optional[int] = None
    blur_size: int = 1  # 1 = blurring in projection disabled
    border: Union[int, BorderSpec, None] = None  # None = auto-fill from mcorr

    # defaults from my_extract_binary_masks_from_structural_channel
    gSig: Union[int, Sequence[int]] = 5  # neuron pixel size
    blur_type: Literal['box', 'gaussian'] = 'gaussian'
    blur_gSig_multiple: Optional[float] = None  # this is for the extract_binary_masks step, not projection
    min_area_size: int = 30
    min_hole_size: int = 15
    expand_method: Literal['closing', 'dilation'] = 'closing'
    selem: np.ndarray = np.ones((3, 3))

    @classmethod
    def infer_from_seed_path(cls, path_to_seed: Union[str, Path]) -> 'SeedParams':
        """Infer seed params from deprecated filename encoding, partially based on modification time"""
        mtime = date.fromtimestamp(os.stat(path_to_seed).st_mtime)
        # date when default changed and I did not otherwise save this info
        after_blur_type_change = mtime >= date(2025, 5, 1)
        seed_params: dict[str, Any] = {'blur_type': 'gaussian' if after_blur_type_change else 'box'}

        # decode filename
        filename = os.path.splitext(os.path.split(path_to_seed)[1])[0]
        if not filename.startswith('Ain_caiman_from'):
            raise RuntimeError('Seed filename does not follow expected format')
        params_part = 'type' + filename.removeprefix('Ain_caiman_from')
        
        for key, val in pairwise(params_part.split('_')):
            if key == 'type':
                seed_params['type'] = val
            elif key == 'medw':
                seed_params['norm_medw'] = int(val)
            elif key == 'blur':
                seed_params['blur_size'] = int(val)
            elif key == 'gSig':
                seed_params['gSig'] = tuple(int(sig) for sig in val.split(','))
            elif key == 'blurmult':
                seed_params['blur_gSig_multiple'] = float(val)
        return cls(**seed_params)
    
    @classmethod
    def default(cls, metadata: dict):
        """Default seed parameters; use empty dict for non-seeded CNMF"""
        _, scale = get_dxy_and_scale(metadata)
        return cls(
            type='mean',
            norm_medw=25,
            gSig=tuple(np.unique(round_to_odd(np.array([5, 7, 9]) * scale))),
            )

    def get_differing_params(self, other: 'SeedParams', metadata: dict[str, Any]) -> Iterable[str]:
        if self.type != 'none' or other.type != 'none':
            for field in fields(self):
                val1 = getattr(self, field.name)
                val2 = getattr(other, field.name)

                if field.name == 'gSig':
                    # treat 1-element sequences the same as scalars
                    val1 = np.atleast_1d(val1)
                    val2 = np.atleast_1d(val2)
                    
                # need to use array_equal because there is an ndarray
                if not np.array_equal(val1, val2):
                    yield field.name


class CNMFParamsExtra(StageParams, frozen=True):
    """Extra parameters for operations associated with CNMF"""
    seed_params: SeedParams = SeedParams()
    crossplane_merge_thr: Optional[float] = None

    def get_differing_params(self, other: 'CNMFParamsExtra', metadata: dict[str, Any]) -> Iterable[str]:
        for param in self.seed_params.get_differing_params(
            other.seed_params, metadata=metadata):
            yield 'seed_params.' + param
        if self.crossplane_merge_thr != other.crossplane_merge_thr:
            yield 'crossplane_merge_thr'


class EvalParamsExtra(StageParams, frozen=True):
    """Extra parameters for CNMF evaluation (do not require redoing CNMF)"""
    snr_type: Literal['normal', 'gamma'] = 'normal'

    def get_differing_params(self, other: 'EvalParamsExtra', metadata: dict[str, Any]) -> Iterable[str]:
        if self.snr_type != other.snr_type:
            yield 'snr_type'


class AnalysisStage(IntEnum):
    """
    Stages of analysis; ordered: if one stage is invalidated, then
    all higher ones should also be invalidated.
    In practice, the current stage is determined by which fields of
    CNMFAnalysis are set (and they can be unset to invalidate past a given stage).
    """
    START = 0       # raw data file identification
    CONVERT = 1     # file format conversion
    MCORR = 2       # motion correction
    TRANSPOSE = 3   # concat/transpose to C order/possibly filter
    CNMF = 4        # CNMF or CNMFE
    EVAL = 5        # component quality evaluation
    FINAL = 6

    @property
    def name(self) -> str:
        return [
            'file discovery',
            'file conversion',
            'motion correction',
            'transposition/plane concatenation',
            'CNMF',
            'component evaluation',
            'final',
            ][self]


# recursive typeddicts for serializing/deserializing parameters for each stage
class ConvertParamDict(TypedDict):
    conversion: ConversionParams

UpToConvertParamDict = ConvertParamDict

class McorrParamDict(TypedDict):
    motion: dict[str, Any]
    mcorr_extra: McorrParamsExtra

class UpToMcorrParamDict(UpToConvertParamDict, McorrParamDict):
    pass

class TransposeParamDict(TypedDict):
    transposition: TranspositionParams

class UpToTransposeParamDict(UpToMcorrParamDict, TransposeParamDict):
    pass

class CNMFParamDict(TypedDict):
    data: dict[str, Any]
    patch: dict[str, Any]
    preprocess: dict[str, Any]
    init: dict[str, Any]
    spatial: dict[str, Any]
    temporal: dict[str, Any]
    merging: dict[str, Any]
    online: dict[str, Any]
    ring_CNN: dict[str, Any]
    cnmf_extra: CNMFParamsExtra

class UpToCNMFParamDict(UpToTransposeParamDict, CNMFParamDict):
    pass

class EvalParamDict(TypedDict):
    quality: dict[str, Any]
    eval_extra: EvalParamsExtra

class UpToEvalParamDict(UpToCNMFParamDict, EvalParamDict):
    pass

FullParamDict = UpToEvalParamDict


# general serialization, deserialization, and comparison logic
def serialize_params(params: Mapping[str, Any], pretty=True) -> bytes:
    enc_bytes = param_encoder.encode(params)
    if pretty:
        enc_bytes = msgspec.json.format(enc_bytes)
    return enc_bytes

def write_params(params: Mapping[str, Any], path: Union[str, Path], pretty=True):
    data = serialize_params(params, pretty=pretty)
    with open(path, mode='wb') as file:
        file.write(data)


@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.START], data: bytes) -> Mapping[str, Any]:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.CONVERT], data: bytes) -> UpToConvertParamDict:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.MCORR], data: bytes) -> UpToMcorrParamDict:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.TRANSPOSE], data: bytes) -> UpToTransposeParamDict:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.CNMF], data: bytes) -> UpToCNMFParamDict:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.EVAL], data: bytes) -> UpToEvalParamDict:
    ...
@overload
def deserialize_params_up_to_stage(stage: Literal[AnalysisStage.FINAL], data: bytes) -> FullParamDict:
    ...

def deserialize_params_up_to_stage(stage: AnalysisStage, data: bytes) -> Mapping[str, Any]:
    """Attempt to decode data representing params for a given stage into one of the TypedDict types"""
    match stage:
        case AnalysisStage.START:
            return {}
        case AnalysisStage.CONVERT:
            deser_type = UpToConvertParamDict
        case AnalysisStage.MCORR:
            deser_type = UpToMcorrParamDict
        case AnalysisStage.TRANSPOSE:
            deser_type = UpToTransposeParamDict
        case AnalysisStage.CNMF:
            deser_type = UpToCNMFParamDict
        case AnalysisStage.EVAL:
            deser_type = UpToEvalParamDict
        case AnalysisStage.FINAL:
            deser_type = FullParamDict  
    return msgspec.json.decode(data, type=deser_type, dec_hook=dec_hook)


@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.START], path: Union[str, Path]) -> Mapping[str, Any]:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.CONVERT], path: Union[str, Path]) -> UpToConvertParamDict:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.MCORR], path: Union[str, Path]) -> UpToMcorrParamDict:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.TRANSPOSE], path: Union[str, Path]) -> UpToTransposeParamDict:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.CNMF], path: Union[str, Path]) -> UpToCNMFParamDict:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.EVAL], path: Union[str, Path]) -> UpToEvalParamDict:
    ...
@overload
def read_params_up_to_stage(stage: Literal[AnalysisStage.FINAL], path: Union[str, Path]) -> FullParamDict:
    ...
    
def read_params_up_to_stage(stage: AnalysisStage, path: Union[str, Path]) -> Mapping[str, Any]:
    """Attempt to read params for a given stage from a file"""
    with open(path, mode='rb') as file:
        data = file.read()
    return deserialize_params_up_to_stage(stage, data)


def get_differing_params(first: Mapping[str, Any], second: Mapping[str, Any], metadata: dict[str, Any],
                         include_different_toplevel=False) -> Iterable[str]:
    """
    Return flattened names of params that differ between first and second
    include_different_toplevel: include top-level keys if one is present and the other is not.
        By default, only compares top-level keys that are present in both dicts.
    """
    for key in set(first.keys()) | set(second.keys()):
        if key in EXCLUDE_FROM_DIFFS:
            continue

        if (key not in first or key not in second) and include_different_toplevel:
            yield key
        else:
            val1, val2 = first[key], second[key]
            if isinstance(val1, dict):
                assert isinstance(val2, dict),  'same key should contain same types'
                diff_subkeys = get_params_diffs([val1, val2])[0].keys()
                for subkey in diff_subkeys:
                    # skip ones that we don't care about
                    subkey_flat = key + '.' + subkey
                    if subkey_flat not in EXCLUDE_FROM_DIFFS:
                        yield subkey_flat
            else:
                # annoyingly it seems like Mapping[str, Union[dict, StageParams]] doesn't work due to @dataclass
                assert isinstance(val1, StageParams), 'params dict should only contain subdicts and StageParams objects'
                assert isinstance(val2, type(val1)),  'same key should contain same types'
                for param in val1.get_differing_params(val2, metadata=metadata):
                    yield key + '.' + param
    
def do_params_match(first: Mapping[str, Any], second: Mapping[str, Any], metadata: dict[str, Any]) -> bool:
    return not any(get_differing_params(first, second, metadata=metadata))


class SessionAnalysisParams:
    """
    Object that contains CNMFParams as well as other parameters used in the SessionAnalysis pipeline.
    These params should not be modified directly; SessionAnalysis fields must be
    invalidated depending on what is changed. Instead, use the SessionAnalysis.update_params method.
    To read params, the SessionAnalysis.read_params method can be used.
    """
    def __init__(self,
        cnmf: params.CNMFParams,
        conversion: ConversionParams = ConversionParams(),
        mcorr_extra: McorrParamsExtra = McorrParamsExtra(),
        transposition: TranspositionParams = TranspositionParams(),
        cnmf_extra: CNMFParamsExtra = CNMFParamsExtra(),
        eval_extra: EvalParamsExtra = EvalParamsExtra()
        ):
        self._cnmf = cnmf
        self._conversion = conversion
        self._mcorr_extra = mcorr_extra
        self._transposition = transposition
        self._cnmf_extra = cnmf_extra
        self._eval_extra = eval_extra

    @classmethod
    def from_dict(cls, input_dict: FullParamDict) -> 'SessionAnalysisParams':
        """
        Construct from deserialized or updated dict (like returned from read_all())
        Must at least have CNMFParams fields.
        """
        # remove nb from spatial and temporal so caiman doesn't complain
        spatial_dict = copy(input_dict['spatial'])
        spatial_dict.pop('nb', None)
        temporal_dict = copy(input_dict['temporal'])
        temporal_dict.pop('nb', None)

        cnmf_params = params.CNMFParams(params_dict={
            'data': input_dict['data'],
            'patch': input_dict['patch'],
            'preprocess': input_dict['preprocess'],
            'init': input_dict['init'],
            'spatial': spatial_dict,
            'temporal': temporal_dict,
            'merging': input_dict['merging'],
            'online': input_dict['online'],
            'motion': input_dict['motion'],
            'ring_CNN': input_dict['ring_CNN']
        })

        obj = cls(
            cnmf=cnmf_params,
            conversion=input_dict['conversion'],
            mcorr_extra=input_dict['mcorr_extra'],
            transposition=input_dict['transposition'],
            cnmf_extra=input_dict['cnmf_extra'],
            eval_extra=input_dict['eval_extra']
        )

        return obj

    
    @classmethod
    def from_metadata(
        cls, metadata: dict, dims: int, tif_file: Optional[str] = None, channel=0, odd_row_offset: Optional[int] = None,
        odd_row_ndead: Optional[list[int]] = None, crop: BorderSpec = BorderSpec(), downsample_factor: Optional[int] = None,
        extra_conversion_params: Optional[dict[str, Any]] = None, seed_params: Optional[dict[str, Any]] = None,
        snr_type: Literal['normal', 'gamma'] = 'gamma', crossplane_merge_thr: Optional[float] = 0.7,
        highpass_cutoff: float = 0, highpass_order=4, add_to_mov=0
        ) -> 'SessionAnalysisParams':
        """Constructor that makes CNMFParams based on other params (ensuring it is consistent)"""
        if extra_conversion_params is None:
            extra_conversion_params = {}

        if seed_params is None:
            seed = SeedParams.default(metadata)
        else:
            seed = SeedParams(**seed_params)

        return cls(
            conversion=ConversionParams(
                channel=channel, odd_row_offset=odd_row_offset, odd_row_ndead=odd_row_ndead, crop=crop,
                downsample_factor=downsample_factor, **extra_conversion_params),
            transposition=TranspositionParams(highpass_cutoff=highpass_cutoff, highpass_order=highpass_order, add_to_mov=add_to_mov),
            cnmf=make_cnmf_params(
                metadata, dims=dims, tif_file=tif_file, snr_type=snr_type, downsample_factor=downsample_factor),
            cnmf_extra=CNMFParamsExtra(seed_params=seed, crossplane_merge_thr=crossplane_merge_thr),
            eval_extra=EvalParamsExtra(snr_type=snr_type)
        )


    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.START]) -> Mapping[str, Any]:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.CONVERT]) -> ConvertParamDict:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.MCORR]) -> McorrParamDict:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.TRANSPOSE]) -> TransposeParamDict:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.CNMF]) -> CNMFParamDict:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.EVAL]) -> EvalParamDict:
        ...
    @overload
    def get_params_for_stage(self, stage: Literal[AnalysisStage.FINAL]) -> Mapping[str, Any]:
        ...

    def get_params_for_stage(self, stage: AnalysisStage) -> Mapping[str, Any]:
        """Get copy of only params that relate to this specific stage"""
        match stage:
            case AnalysisStage.START:
                return {}
            case AnalysisStage.CONVERT:
                return ConvertParamDict(conversion=deepcopy(self._conversion))
            case AnalysisStage.MCORR:
                return McorrParamDict(motion=deepcopy(self._cnmf.motion), mcorr_extra=deepcopy(self._mcorr_extra))
            case AnalysisStage.TRANSPOSE:
                return TransposeParamDict(transposition=deepcopy(self._transposition))
            case AnalysisStage.CNMF:
                # make cnmf_params dict with motion and quality fields removed
                cnmf_params_dict = deepcopy(self._cnmf.to_dict())
                del cnmf_params_dict['motion']
                del cnmf_params_dict['quality']
                return CNMFParamDict(**cnmf_params_dict, cnmf_extra=deepcopy(self._cnmf_extra))
            case AnalysisStage.EVAL:
                return EvalParamDict(
                    quality=deepcopy(self._cnmf.quality),
                    eval_extra=deepcopy(self._eval_extra))
            case AnalysisStage.FINAL:
                return {}


    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.START]) -> Mapping[str, Any]:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.CONVERT]) -> UpToConvertParamDict:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.MCORR]) -> UpToMcorrParamDict:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.TRANSPOSE]) -> UpToTransposeParamDict:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.CNMF]) -> UpToCNMFParamDict:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.EVAL]) -> UpToEvalParamDict:
        ...
    @overload
    def get_params_up_to_stage(self, stage: Literal[AnalysisStage.FINAL]) -> FullParamDict:
        ...

    def get_params_up_to_stage(self, stage: AnalysisStage) -> Mapping[str, Any]:
        """
        Read params as a dict, including all information that could be used to invalidate the given stage.
        """
        curr_stage = AnalysisStage.START
        params = cast(dict, self.get_params_for_stage(curr_stage))

        while curr_stage < stage:
            curr_stage = AnalysisStage(curr_stage + 1)
            params.update(self.get_params_for_stage(curr_stage))
        
        return params


    def read_all(self) -> FullParamDict:
        return self.get_params_up_to_stage(AnalysisStage.FINAL)


    # Serialization
    def write_params_for_stage(self, stage: AnalysisStage, path: Union[str, Path], pretty=True):
        """Write JSON of params for a given stage to file"""
        write_params(self.get_params_up_to_stage(stage), path=path, pretty=pretty)
    
    def write_json_file(self, path: Union[str, Path], pretty=True):
        self.write_params_for_stage(AnalysisStage.FINAL, path=path, pretty=pretty)


    # Param comparison
    def get_differing_params(self, stage: AnalysisStage, other_params: Mapping[str, Any],
                             metadata: dict[str, Any]) -> Iterable[str]:
        this_params = self.get_params_up_to_stage(stage)
        return get_differing_params(this_params, other_params, metadata=metadata)

    def get_differing_params_from_file(self, stage: AnalysisStage, path: Union[str, Path],
                                       metadata: dict[str, Any]) -> Iterable[str]:
        other_params = read_params_up_to_stage(stage, path)
        return self.get_differing_params(stage, other_params, metadata=metadata)
    
    def does_params_file_match(self, stage: AnalysisStage, path: Union[str, Path],
                               metadata: dict[str, Any]) -> bool:
        return not any(self.get_differing_params_from_file(stage, path, metadata=metadata))


    # Updating, safely
    def change_params_and_get_stage_to_invalidate(
            self, changes: Mapping[str, Mapping[str, Any]], metadata: dict[str, Any]
            ) -> tuple['SessionAnalysisParams', Optional[AnalysisStage]]:
        """
        Make a copy with new parameters, and also return which analysis stage should be invalidated
        to ensure we are using these new parameters in future operations (if any).
        """
        # strategy: iterate through stages from first to last
        # at each stage, make updates pertaining to this stage to the copy, 
        # and test whether that stage needs to be invalidated.
        # Once a stage to be invalidated has been found, the rest of the tests can be skipped.
        stage = AnalysisStage.START
        new_params = cast(dict, self.get_params_for_stage(stage))
        invalid_stage: Optional[AnalysisStage] = None
        changes = dict(changes)

        while stage < AnalysisStage.FINAL:
            stage = AnalysisStage(stage + 1)
            curr_stage_params = self.get_params_for_stage(stage)
            new_stage_params = dict(curr_stage_params)
            for key, subparams in new_stage_params.items():
                if key in changes:
                    change_subparams = changes.pop(key)
                    if isinstance(subparams, dict):
                        if key in ('spatial', 'temporal') and 'nb' in change_subparams:
                            raise ValueError('Cannot set nb under "spatial" or "temporal" - use init.nb instead.')

                        # set to an updated copy
                        new_subparams = copy(subparams)
                        for subkey, change_subval in change_subparams.items():
                            if subkey not in subparams:
                                raise ValueError(f'Cannot set unknown parameter {key}.{subkey}')
                            new_subparams[subkey] = change_subval
                        new_stage_params[key] = new_subparams
                    else:
                        assert isinstance(subparams, StageParams), 'Unexpected params type'
                        assert isinstance(change_subparams, Mapping), \
                             'Changes should always be dicts to avoid replacing non-specified params'
                        new_stage_params[key] = subparams.replace(change_subparams)

            new_params.update(new_stage_params)

            if invalid_stage is None and not do_params_match(curr_stage_params, new_stage_params, metadata):
                invalid_stage = stage

                # unset _indices_are_adjusted flag if necessary
                if 'conversion' in curr_stage_params:
                    assert isinstance(curr_stage_params['conversion'], ConversionParams)
                    assert isinstance(new_stage_params['conversion'], ConversionParams)
                    for differing_param in curr_stage_params['conversion'].get_differing_params(
                        new_stage_params['conversion'], metadata
                    ):
                        if differing_param in ['crop', 'odd_row_offset', 'odd_row_ndead']:
                            if 'mcorr_extra' not in changes:
                                changes['mcorr_extra'] = {'_indices_are_adjusted': False}
                            else:
                                changes['mcorr_extra'] = {**changes['mcorr_extra'], '_indices_are_adjusted': False}
                            break
        
        new_params = FullParamDict(**new_params)
        return SessionAnalysisParams.from_dict(new_params), invalid_stage

    
    def read_cnmf_params(self) -> params.CNMFParams:
        """Copy and return CNMFParams object"""
        return deepcopy(self._cnmf)
    
    def __repr__(self) -> str:
        dict_rep = self.read_all()
        return f'SessionAnalysisParams.from_dict({repr(dict_rep)})'
    
    def __str__(self) -> str:
        return repr(self)

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
        'min_SNR': 1.0,
        'rval_lowest': 0.1,
        'rval_thr': 0.6,
        'cnn_lowest': 0.1,
        'min_cnn_thr': 0.99,
        'gSig_range': [[7, 7], [8, 8], [9, 9]],
        'use_cnn': False,
        'use_ecc': False,
        'max_ecc': 3
    }
}


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