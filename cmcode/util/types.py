from typing import Optional, Union, TYPE_CHECKING
import numpy as np
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


MaybeSparse =  Union[np.ndarray, sparse.csc_matrix, sparse.csc_array]


# helpers for mesmerize-core
if TYPE_CHECKING:
    class MescoreBatch(pd.DataFrame):
        paths: mc.batch_utils.PathsDataFrameExtension
        caiman: mc.caiman_extensions.CaimanDataFrameExtensions

    class MescoreSeries(pd.DataFrame):
        paths: mc.batch_utils.PathsSeriesExtension
        caiman: mc.caiman_extensions.CaimanSeriesExtensions
        mcorr: mc.caiman_extensions.MCorrExtensions
        cnmf: mc.caiman_extensions.CNMFExtensions
else:
    class MescoreBatch:
        pass

    class MescoreSeries:
        pass
