from collections.abc import Container, Iterator, Sequence, Mapping
from copy import deepcopy
from datetime import date
from enum import IntEnum
from functools import cache
import h5py
from itertools import pairwise
import json
import os
from pathlib import Path
from typing import Optional, Literal, Union, Any, Type, TypeVar, Annotated, cast
import warnings

from caiman.source_extraction.cnmf import params
from caiman.source_extraction.cnmf.utilities import all_same
from caiman.utils.utils import recursively_load_dict_contents_from_group
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter, BeforeValidator, Field, PrivateAttr, computed_field, model_validator
from pydantic.dataclasses import dataclass
from pydantic.json_schema import SkipJsonSchema, PydanticJsonSchemaWarning

from cmcode.util import types, paths
from cmcode.util.image import BorderSpec


# pydantic helpers
def list_from_ndarray(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

ReadNDArray = BeforeValidator(list_from_ndarray)


Self = TypeVar('Self', bound='StageParams')

@dataclass(kw_only=True, frozen=True)
class StageParams:
    """Params for a particular analysis stage (abstract base class)"""
    __pydantic_config__ = ConfigDict(extra='forbid', serialize_by_alias=True)

    @classmethod
    @cache
    def input_params(cls) -> set[str]:
        """Param names that can be used in constructor etc. (excludes purely computed fields)"""
        ta = TypeAdapter(cls)
        # we don't care if some defaults aren't serializable
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PydanticJsonSchemaWarning, message='Default value')
            return set(ta.json_schema(mode='validation')['properties'].keys())
    
    @classmethod
    @cache
    def params(cls) -> set[str]:
        """Parameters available to read from this group"""
        # Use the JSON schema to ensure we respect excluded fields, etc
        ta = TypeAdapter(cls)
        # we don't care if some defaults aren't serializable
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PydanticJsonSchemaWarning, message='Default value')
            return set(ta.json_schema(mode='serialization')['properties'].keys())


    def get_differing_params(self: Self, other: Self, metadata: dict[str, Any]) -> Iterator[str]:
        yield from ()
        """Find names of parameters that are different between self and other"""

    def matches(self: Self, other: Self, metadata: dict[str, Any]) -> bool:
        """Find whether any parameters differ between self and other"""
        return not any(self.get_differing_params(other, metadata=metadata))
    
    def replace(self: Self, **replacements) -> Self:
        """Make copy with replacements; like dataclasses.replace but recursing on other StageParams"""
        ta = TypeAdapter(type(self))
        param_dict = ta.dump_python(self, round_trip=True)

        # update the dict while recursing into other StageParams instances
        for key, val in replacements.items():
            if not hasattr(self, key):
                raise KeyError(f'Cannot replace unknown key {key}')

            my_val = getattr(self, key)
            if isinstance(my_val, StageParams) and isinstance(val, Mapping):
                val = my_val.replace(**val)
            param_dict[key] = val
        
        return ta.validate_python(param_dict)

    # support copy.replace (for 3.13 and above)
    __replace__ = replace


# note: in the below params classes, defaults reflect the original values of each parameter/the
# value used if it is left unspecified when calling subroutines,
# which is not necessarily the same as the default when creating a new SessionAnalysis object.

# TODO: add stage params for file discovery (START stage) including trials_to_exclude, raw_dir, etc.


@dataclass(kw_only=True, frozen=True)
class ConversionParams(StageParams):
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
    dead_pix_mode: Union[bool, params.LitStr[Literal['copy', 'min']]] = 'copy'

    def get_differing_params(self, other: 'ConversionParams', metadata: dict[str, Any]) -> Iterator[str]:
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

        for param in other.params():
            if param in irrelevant_fields:
                continue

            # use all_same to compare without caring about e.g. the sequence type
            if not all_same(getattr(other, param), getattr(self, param)):
                yield param


@dataclass(kw_only=True, frozen=True)
class McorrParamsExtra(StageParams):
    """Extra params for the motion correction step"""
    # reference to SessionAnalysisParams to compute indices_exclude_fringe
    _mcorr_params: 'SkipJsonSchema[Optional[McorrParamStruct]]' = Field(
        default=None, init=False, exclude=True, repr=False
    )

    # whether to automatically adjust indices to take odd_row_ndead, odd_row_offset and crop into account
    # default = only for rigid motion correction.
    _indices_exclude_fringe: Optional[bool] = Field(default=None, alias='indices_exclude_fringe')
    # flag that is set when indices are adjusted and unset when params affecting them are changed
    _indices_are_adjusted: bool = False

    @computed_field
    @property
    def indices_exclude_fringe(self) -> bool:
        """Determine indices_exclude_fringe from pw_rigid if None"""
        if self._indices_exclude_fringe is not None:
            return self._indices_exclude_fringe
        
        if self._mcorr_params is None:
            raise RuntimeError('Cannot compute indices_exclude_fringe without reference to mcorr params')
        
        return not self._mcorr_params.motion.pw_rigid


    def get_differing_params(self, other: 'McorrParamsExtra', metadata: dict[str, Any]) -> Iterator[str]:
        # only care about indices_exclude_fringe if indices are out of date, meaning that
        # matching motion.indices can't be trusted.

        if self.indices_exclude_fringe and not other.indices_exclude_fringe and not self._indices_are_adjusted:
            yield 'indices_exclude_fringe'
        elif other.indices_exclude_fringe and not self.indices_exclude_fringe and not other._indices_are_adjusted:
            yield 'indices_exclude_fringe'


@dataclass(kw_only=True, frozen=True)
class TranspositionParams(StageParams):
    """Settings for transpose/concat planes"""
    highpass_cutoff: float = 0.
    highpass_order: int = 4
    add_to_mov: float = 0.
    blur_kernel_size: int = 1  # same as blur_size in SeedParams

    def get_differing_params(self, other: 'TranspositionParams', metadata: dict[str, Any]) -> Iterator[str]:
        irrelevant_fields = ['conversion_params', 'highpass_cutoff']
        if self.highpass_cutoff != other.highpass_cutoff:
            yield 'highpass_cutoff'
        
        if self.highpass_cutoff == 0:
            irrelevant_fields.append('highpass_order')

        for param in self.params():
            if param in irrelevant_fields:
                continue

            if not all_same(getattr(other, param), getattr(self, param)):
                yield param


@dataclass(kw_only=True, frozen=True)
class SeedParams(StageParams):
    """Params for producing spatial seed"""
    type: str = 'none' # e.g. "mean", "skew"; "none" means don't use a seed at all
    norm_medw: Optional[int] = None
    blur_size: int = 1  # 1 = blurring in projection disabled
    borders: Optional[list[BorderSpec]] = None  # None = auto-fill from mcorr

    # defaults from my_extract_binary_masks_from_structural_channel
    gSig: Annotated[Union[None, int, Sequence[int]], ReadNDArray] = 5  # neuron pixel size (None = same as CNMF gSig)
    blur_type: params.LitStr[Literal['box', 'gaussian']] = 'gaussian'
    blur_gSig_multiple: Optional[float] = None  # this is for the extract_binary_masks step, not projection
    min_area_size: int = 30
    min_hole_size: int = 15
    expand_method: params.LitStr[Literal['closing', 'dilation']] = 'closing'
    selem: params.NDArray = Field(default_factory=lambda: np.ones((3, 3)))

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
            gSig=np.unique(round_to_odd(np.array([5, 7, 9]) * scale)).tolist(),
            )

    def get_differing_params(self, other: 'SeedParams', metadata: dict[str, Any]) -> Iterator[str]:
        if self.type != 'none' or other.type != 'none':
            for param in self.params():
                val1 = getattr(self, param)
                val2 = getattr(other, param)

                if param == 'gSig':
                    # treat 1-element sequences the same as scalars
                    val1 = np.atleast_1d(val1)
                    val2 = np.atleast_1d(val2)
                    
                # need to use array_equal because there is an ndarray
                if not all_same(val1, val2):
                    yield param


@dataclass(kw_only=True, frozen=True)
class CNMFParamsExtra(StageParams):
    """Extra parameters for operations associated with CNMF"""
    seed_params: SeedParams = SeedParams()
    crossplane_merge_thr: Optional[float] = None

    def get_differing_params(self, other: 'CNMFParamsExtra', metadata: dict[str, Any]) -> Iterator[str]:
        for param in self.seed_params.get_differing_params(
            other.seed_params, metadata=metadata):
            yield 'seed_params.' + param
        if self.crossplane_merge_thr != other.crossplane_merge_thr:
            yield 'crossplane_merge_thr'


@dataclass(kw_only=True, frozen=True)
class EvalParamsExtra(StageParams):
    """Extra parameters for CNMF evaluation (do not require redoing CNMF)"""
    snr_type: params.LitStr[Literal['normal', 'gamma']] = 'normal'

    def get_differing_params(self, other: 'EvalParamsExtra', metadata: dict[str, Any]) -> Iterator[str]:
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


# recursive models for serializing/deserializing parameters for each stage
SelfPS = TypeVar('SelfPS', bound='ParamStruct')

class ParamStruct(BaseModel):
    @classmethod
    def read_from_file(cls: Type[SelfPS], path: Union[str, Path]) -> SelfPS:
        with open(path, mode='r') as file:
            data = file.read()
        return cls.model_validate_json(data)
    
    def serialize_params(self, stage: Optional[AnalysisStage] = None, pretty=True) -> str:
        """Convert a group of params to JSON, optionally specifying the subset of params to serialize using params_type."""
        if stage is None:
            encoded = self.model_dump(mode='json', round_trip=True)
        else:
            # use TypeAdapter to serialize only params relevant for the stage
            ta = TypeAdapter(stage_cumulative_params[stage])
            encoded = ta.dump_python(self, mode='json', round_trip=True)

        # use json library for dumping b/c it allows nans and infs
        return json.dumps(encoded, indent=4 if pretty else None)

    def write_params(self, path: Union[str, Path], stage: Optional[AnalysisStage] = None, pretty=True):
        data = self.serialize_params(stage=stage, pretty=pretty)
        with open(path, mode='w') as file:
            file.write(data)

    
    def get_differing_params(
            self, other: Union['ParamStruct', params.CNMFParams], metadata: dict[str, Any], stage: Optional[AnalysisStage] = None,
            include_different_toplevel=False, ignore_prereq_stages=False,
            params_to_exclude: Optional[Container[str]] = None, exclude_quality=True) -> Iterator[str]:
        """
        Return flattened names of params that differ between first and second
        include_different_toplevel: include top-level keys if one is present and the other is not.
            By default, only compares top-level keys that are present in both dicts.
        """
        if params_to_exclude is None:
            params_to_exclude = {
                'online.path_to_model',  # depends on caiman data dir
                'online.init_batch',
                'online.movie_name_online',
                'preprocess.n_pixels_per_process',
                'spatial.n_pixels_per_process',
                'patch.n_processes',
                'data.caiman_version',  # could be relevant at some point, but don't show for now
                'data.last_commit',
                'data.fnames',
            }

        # infer whether CNMF is seeded because in this case we don't care about patch.rf and patch.only_init
        seeded = True
        for struct in self, other:
            if not isinstance(struct, CNMFParamStruct) or struct.cnmf_extra.seed_params.type == 'none':
                seeded = False

        first_fields = type(self).model_fields.keys()
        if isinstance(other, params.CNMFParams):
            second_fields = set(other.groups)
        else:
            second_fields = type(other).model_fields.keys()

        if stage is not None:
            if ignore_prereq_stages:
                # subset to fields for this stage
                stage_fields = stage_only_params[stage].model_fields.keys()
            else:
                # subset to fields used up to this stage
                stage_fields = stage_cumulative_params[stage].model_fields.keys()
            first_fields &= stage_fields
            second_fields &= stage_fields

        for key in first_fields | second_fields:
            if key in params_to_exclude or (exclude_quality and key == 'quality'):
                continue

            if (key not in first_fields or key not in second_fields):
                if include_different_toplevel:
                    yield key
                continue

            val1, val2 = getattr(self, key), getattr(other, key)
            if isinstance(val1, params.GroupParams):
                assert isinstance(val2, type(val1)),  'same key should contain same types'
                for param, _, _ in val1.get_differing_params(val2):
                    # skip ones that we don't care about
                    param_flat = key + '.' + param
                    if param_flat in params_to_exclude:
                        continue

                    # ignore rf and only_init if we are using seeded CNMF
                    if seeded and key == 'patch' and param in ['rf', 'only_init']:
                        continue

                    yield param_flat
            else:
                assert isinstance(val1, StageParams), 'params dict should only contain subdicts and StageParams objects'
                assert isinstance(val2, type(val1)),  'same key should contain same types'
                for param in val1.get_differing_params(val2, metadata=metadata):
                    yield key + '.' + param
        
    def do_params_match(self, other: Union['ParamStruct', params.CNMFParams], metadata: dict[str, Any],
                        stage: Optional[AnalysisStage] = None, ignore_prereq_stages=False) -> bool:
        return not any(self.get_differing_params(other, metadata=metadata, stage=stage, ignore_prereq_stages=ignore_prereq_stages))

    
    def get_differing_params_from_file(
            self, path: Union[str, Path], metadata: dict[str, Any], stage: Optional[AnalysisStage] = None) -> Iterator[str]:
        if stage is None:
            other_params = type(self).read_from_file(path)
        else:
            other_params = stage_cumulative_params[stage].read_from_file(path)
        return self.get_differing_params(other_params, metadata=metadata, stage=stage)
    
    def does_params_file_match(self, path: Union[str, Path], metadata: dict[str, Any], stage: Optional[AnalysisStage] = None) -> bool:
        return not any(self.get_differing_params_from_file(path, metadata=metadata, stage=stage))


class ConvertParamStruct(ParamStruct):
    conversion: ConversionParams

UpToConvertParamStruct = ConvertParamStruct

class McorrParamStruct(ParamStruct):
    motion: params.MotionParams
    mcorr_extra: McorrParamsExtra

    @model_validator(mode='after')
    def _set_ref(self):
        """Set reference to full params struct"""
        object.__setattr__(self.mcorr_extra, '_mcorr_params', self)
        return self

class UpToMcorrParamStruct(UpToConvertParamStruct, McorrParamStruct):
    pass

class TransposeParamStruct(ParamStruct):
    transposition: TranspositionParams

class UpToTransposeParamStruct(UpToMcorrParamStruct, TransposeParamStruct):
    pass

class CNMFParamStruct(ParamStruct):
    data: params.DataParams
    patch: params.PatchParams
    preprocess: params.PreprocessParams
    init: params.InitParams
    spatial: params.SpatialParams
    temporal: params.TemporalParams
    merging: params.MergingParams
    online: params.OnlineParams
    ring_CNN: params.RingCNNParams
    cnmf_extra: CNMFParamsExtra

class UpToCNMFParamStruct(UpToTransposeParamStruct, CNMFParamStruct):
    pass

class EvalParamStruct(ParamStruct):
    quality: params.QualityParams
    eval_extra: EvalParamsExtra

class UpToEvalParamStruct(UpToCNMFParamStruct, EvalParamStruct):
    pass


class SessionAnalysisParams(UpToEvalParamStruct):
    """
    Object that contains CNMFParams as well as other parameters used in the SessionAnalysis pipeline.
    These params should not be modified directly; SessionAnalysis fields must be
    invalidated depending on what is changed. Instead, use the SessionAnalysis.update_params method.
    """
    _cnmf: params.CNMFParams = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        """Make CNMFParams object to manage changes to these params"""
        super().model_post_init(context)
        
        self._cnmf = params.CNMFParams(
            data=self.data,
            patch=self.patch,
            preprocess=self.preprocess,
            init=self.init,
            spatial=self.spatial,
            temporal=self.temporal,
            merging=self.merging,
            online=self.online,
            motion=self.motion,
            ring_CNN=self.ring_CNN,
            quality=self.quality
        )

    @classmethod
    def from_cnmf_params(cls,
        cnmf: params.CNMFParams,
        conversion: ConversionParams = ConversionParams(),
        mcorr_extra: McorrParamsExtra = McorrParamsExtra(),
        transposition: TranspositionParams = TranspositionParams(),
        cnmf_extra: CNMFParamsExtra = CNMFParamsExtra(),
        eval_extra: EvalParamsExtra = EvalParamsExtra()
        ):

        return cls(
            conversion=conversion,
            motion=cnmf.motion,
            mcorr_extra=mcorr_extra,
            transposition=transposition,
            data=cnmf.data,
            patch=cnmf.patch,
            preprocess=cnmf.preprocess,
            init=cnmf.init,
            spatial=cnmf.spatial,
            temporal=cnmf.temporal,
            merging=cnmf.merging,
            online=cnmf.online,
            ring_CNN=cnmf.ring_CNN,
            cnmf_extra=cnmf_extra,
            quality=cnmf.quality,
            eval_extra=eval_extra
        )

    @classmethod
    def from_metadata(
        cls, metadata: dict, dims: int, tif_file: Optional[str] = None, channel=0, odd_row_offset: Optional[int] = None,
        odd_row_ndead: Optional[list[int]] = None, crop: BorderSpec = BorderSpec(), downsample_factor: Optional[int] = None,
        extra_conversion_params: Optional[dict[str, Any]] = None, seed_params: Optional[dict[str, Any]] = None,
        snr_type: Optional[Literal['normal', 'gamma']] = None, crossplane_merge_thr: Optional[float] = 0.7,
        highpass_cutoff: float = 0, highpass_order=4, add_to_mov=0
        ) -> 'SessionAnalysisParams':
        """Constructor that makes CNMFParams based on other params (ensuring it is consistent)"""
        if extra_conversion_params is None:
            extra_conversion_params = {}

        if seed_params is None:
            seed = SeedParams.default(metadata)
        else:
            seed = SeedParams(**seed_params)

        if snr_type is None:
            snr_type = 'gamma'

        return cls.from_cnmf_params(
            conversion=ConversionParams(
                channel=channel, odd_row_offset=odd_row_offset, odd_row_ndead=odd_row_ndead, crop=crop,
                downsample_factor=downsample_factor, **extra_conversion_params),
            transposition=TranspositionParams(highpass_cutoff=highpass_cutoff, highpass_order=highpass_order, add_to_mov=add_to_mov),
            cnmf=make_cnmf_params(
                metadata, dims=dims, tif_file=tif_file, snr_type=snr_type, downsample_factor=downsample_factor),
            cnmf_extra=CNMFParamsExtra(seed_params=seed, crossplane_merge_thr=crossplane_merge_thr),
            eval_extra=EvalParamsExtra(snr_type=snr_type)
        )


    def get_first_nonmatching_stage(self, other: 'SessionAnalysisParams', metadata: dict[str, Any]) -> AnalysisStage:
        """Just identify the first invalid/nonmatching stage compared to another params object"""
        for stage_num in range(int(AnalysisStage.START) + 1, int(AnalysisStage.FINAL)):
            curr_stage = AnalysisStage(stage_num)

            if not self.do_params_match(other, metadata=metadata, stage=curr_stage, ignore_prereq_stages=True):
                return curr_stage

        return AnalysisStage.FINAL


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
        new_params = deepcopy(self)
        invalid_stage: Optional[AnalysisStage] = None
        changes = dict(changes)

        while stage < AnalysisStage.FINAL:
            stage = AnalysisStage(stage + 1)

            for key in stage_only_params[stage].model_fields.keys() & changes.keys():
                change_subparams = changes.pop(key)
                if not isinstance(change_subparams, Mapping):
                        raise TypeError('Changes should always be dicts to avoid replacing non-specified params')

                # set new_params attribute to an updated copy
                curr_subparams: Union[params.GroupParams, StageParams] = getattr(new_params, key)
                new_subparams = curr_subparams.replace(**change_subparams)
                setattr(new_params, key, new_subparams)

                # special case: unset _indices_are_adjusted flag if necessary
                if isinstance(curr_subparams, ConversionParams):
                    assert isinstance(new_subparams, ConversionParams)
                    for differing_param in curr_subparams.get_differing_params(new_subparams, metadata=metadata):
                        if differing_param in ['crop', 'odd_row_offset', 'odd_row_ndead']:
                            # add to mcorr_extra changes to apply in future loop iteration
                            if 'mcorr_extra' not in changes:
                                changes['mcorr_extra'] = {'_indices_are_adjusted': False}
                            else:  # note we make a new dict since the value is not guaranteed to be mutable
                                changes['mcorr_extra'] = {**changes['mcorr_extra'], '_indices_are_adjusted': False}
                            break

            if invalid_stage is None and not self.do_params_match(new_params, metadata=metadata, stage=stage, ignore_prereq_stages=True):
                invalid_stage = stage

        if changes:
            raise RuntimeError('These groups in change dict did not match: ' + ', '.join(changes.keys()))

        return new_params, invalid_stage

    
    def read_cnmf_params(self) -> params.CNMFParams:
        """Copy and return CNMFParams object"""
        # fix some fields that aren't relevant/depend on current environment
        default_params = params.CNMFParams()
        updates = {
            'patch': {
                'n_processes': default_params.patch['n_processes']  # set in mesmerize
            },
            'online': {
                'movie_name_online': default_params.online['movie_name_online'],
                'path_to_model': default_params.online['path_to_model'],
                'init_batch': default_params.online['init_batch']
            }
        }

        # fix rf and only_init depending on whether we are doing seeded CNMF
        if self.cnmf_extra.seed_params.type != 'none':
            updates['patch']['rf'] = None
            updates['patch']['only_init'] = False

        run_params = deepcopy(self._cnmf)
        run_params.change_params(updates)
        return run_params


    def copy_with_mesmerize_run_differences(self) -> 'SessionAnalysisParams':
        """
        Return a copy with changes to reflect the parameters used during a mesmerize-core run,
        before finish_cnmf_processing is called.
        """
        new_params = deepcopy(self)
        new_params.cnmf_extra = self.cnmf_extra.replace(crossplane_merge_thr=None)
        new_params.eval_extra = self.eval_extra.replace(snr_type='normal')
        return new_params
    

# convenience mappings to find params container for a given analysis stage
# (however, it is not useful for type checking/generics)
stage_only_params: dict[AnalysisStage, Type[ParamStruct]] = {
    AnalysisStage.START: ParamStruct,
    AnalysisStage.CONVERT: ConvertParamStruct,
    AnalysisStage.MCORR: McorrParamStruct,
    AnalysisStage.TRANSPOSE: TransposeParamStruct,
    AnalysisStage.CNMF: CNMFParamStruct,
    AnalysisStage.EVAL: EvalParamStruct,
    AnalysisStage.FINAL: ParamStruct
}


stage_cumulative_params: dict[AnalysisStage, Type[ParamStruct]] = {
    AnalysisStage.START: ParamStruct,
    AnalysisStage.CONVERT: UpToConvertParamStruct,
    AnalysisStage.MCORR: UpToMcorrParamStruct,
    AnalysisStage.TRANSPOSE: UpToTransposeParamStruct,
    AnalysisStage.CNMF: UpToCNMFParamStruct,
    AnalysisStage.EVAL: UpToEvalParamStruct,
    AnalysisStage.FINAL: SessionAnalysisParams
}


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
            'border_nan': 'copy',
            'shifts_interpolate': True
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


# more utilities for loading

def load_params_from_cnmf_h5(filename: Union[Path, str]) -> params.CNMFParams:
    """Load just params from CNMF HDF5 file"""
    with h5py.File(filename, 'r') as h5file:
        params_grp = recursively_load_dict_contents_from_group(h5file, '/params/')
    loaded_params = params.CNMFParams(**params_grp)
    # blank out fnames to avoid data validation issue
    loaded_params.change_params({'data': {'fnames': None}})
    return loaded_params


def load_params_from_batch_item(item: pd.Series) -> Union[SessionAnalysisParams, params.CNMFParams]:
    """
    Load params used for this batch item, either from the full params JSON file (if saved)
    or from the params saved with the CNMF run.
    """
    item = cast(types.MescoreSeries, item)
    cnmf_path = str(item.cnmf.get_output_path())
    params_path = paths.params_file_for_result(cnmf_path)

    try:
        return SessionAnalysisParams.read_from_file(params_path)
    except FileNotFoundError:
        return load_params_from_cnmf_h5(cnmf_path)
