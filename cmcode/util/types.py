from typing import Optional, Union, TYPE_CHECKING, TypeVar, Type, Callable

import numpy as np
import optype.numpy as onp
from scipy import sparse
import pandas as pd

if TYPE_CHECKING:  # avoid costly import at runtime
    import mesmerize_core as mc


class UnknownHostError(KeyError):
    def __init__(self, host: str):
        super().__init__(f'Information on host "{host}" not found')


class UnknownPartitionError(KeyError):
    def __init__(self, host: str, partition: str):
        super().__init__(f'Information on partition {partition} not found for host {host}')


class NoMatchingResultError(RuntimeError):
    pass

class NoMultisessionResults(RuntimeError):
    def __init__(self, msg: Optional[str] = None):
        if msg is None:
            msg = 'No saved multisession results found'
        super().__init__(msg)


class NoBatchFileError(RuntimeError):
    def __init__(self):
        super().__init__('Batch file does not exist')


class BadFitError(RuntimeError):
    pass


# MaybeSparse type that can contain any number or boolean
ST = TypeVar('ST', bound=Union[np.number, np.bool_])
MaybeSparse = Union[onp.Array2D[ST], sparse.csc_matrix[ST], sparse.csc_array[ST]]

# we need moar dimensions
Array4D = onp.Array[tuple[int, int, int, int], onp._array._SCT]


# helpers for mesmerize-core
if TYPE_CHECKING:
    class MescoreBatch(pd.DataFrame):
        @property
        def _constructor(self):
            return MescoreBatch

        @property
        def _constructor_sliced(self):
            return MescoreSeries

        # columns
        algo: pd.Series[str]
        item_name: pd.Series[str]
        input_movie_path: pd.Series[str]
        params: pd.Series
        outputs: pd.Series
        added_time: pd.Series[str]
        ran_time: pd.Series
        algo_duration: pd.Series
        comments: pd.Series
        uuid: pd.Series[str]
        
        # accessors
        paths: mc.batch_utils.PathsDataFrameExtension
        caiman: mc.caiman_extensions.CaimanDataFrameExtensions

    class MescoreSeries(pd.Series):
        @property
        def _constructor(self):
            return MescoreSeries
        
        @property
        def _constructor_expanddim(self):
            return MescoreBatch
        
        # fields
        algo: str
        item_name: str
        input_movie_path: str
        params: dict
        outputs: Optional[dict]
        added_time: str
        ran_time: Optional[str]
        algo_duration: Optional[str]
        comments: Optional[str]
        uuid: str

        # accessors
        paths: mc.batch_utils.PathsSeriesExtension
        caiman: mc.caiman_extensions.CaimanSeriesExtensions
        mcorr: mc.caiman_extensions.MCorrExtensions
        cnmf: mc.caiman_extensions.CNMFExtensions
else:
    class MescoreBatch:
        pass

    class MescoreSeries:
        pass

