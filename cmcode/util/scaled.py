from typing import Union, Sequence, Literal, Optional, Type, Iterable, Hashable, Mapping, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.core.dtypes.dtypes import NumpyEADtype  # type: ignore
from pandas.core.arrays import NumpyExtensionArray  # type: ignore


class ScaledArray(NumpyExtensionArray):
    """Extension of the existing pandas Numpy array type that can have a ScaledOffset type"""
    def __init__(self, values: Union[np.ndarray, NumpyExtensionArray],
                 dtype: 'Optional[ScaledOffsetDtype]' = None, copy=False):
        if not isinstance(values, np.ndarray) and hasattr(values, '_ndarray'):
            values = values._ndarray
            
        super().__init__(values, copy=copy)
        self._dist_type = dtype if dtype is not None else None

    @property
    def dtype(self) -> 'Union[NumpyEADtype, ScaledOffsetDtype]':
        if self._dist_type is not None:
            return self._dist_type
        else:
            return super().dtype

    # override methods that use cls or type(self) without dtype
    @classmethod
    def _from_sequence(cls, scalars: Sequence, *, dtype: 'Optional[ScaledOffsetDtype]'=None, copy=False):
        if dtype is None:
            raise ValueError('Cannot assume dtype for ScaledArray')
        array = super()._from_sequence(scalars, dtype=dtype.sub_dtype, copy=copy)
        return cls(array, dtype=dtype)
 
    def _from_backing_data(self, arr: np.ndarray):
        return type(self)(arr, dtype=self.dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if ufunc.nout > 1:
            # multiple return values
            return (type(self)(x, dtype=self.dtype) for x in result)
        elif isinstance(result, NumpyExtensionArray):
            return type(self)(result, dtype=self.dtype)
        else:
            return result
    
    def __invert__(self):
        return type(self)(super().__invert__(), dtype=self.dtype)
    
    def __neg__(self):
        return type(self)(super().__invert__(), dtype=self.dtype)
    
    def __pos__(self):
        return type(self)(super().__pos__(), dtype=self.dtype)
    
    def __abs__(self):
        return type(self)(super().__abs__(), dtype=self.dtype)

    def _wrap_ndarray_result(self, result: np.ndarray):
        result = super()._wrap_ndarray_result(result)
        # if the result is the same type of value, e.g. not the result of a comparison, cast to this type
        if isinstance(result, NumpyExtensionArray) and result.dtype.numpy_dtype == self.dtype.numpy_dtype:
            result = type(self)(result, dtype=self.dtype)
        return result

    # override some operation behavior
    def _cmp_method(self, other, op):
        if isinstance(other, ScaledArray) and other.dtype.unit != self.dtype.unit:
            raise TypeError('Cannot compare arrays with different units directly')
        return super()._cmp_method(other, op)
    
    def _arith_method(self, other, op):
        if isinstance(other, ScaledArray):
            if other.dtype.unit != self.dtype.unit:
                raise TypeError('Cannot combine arrays with different units directly')
            res_type = self.dtype._get_common_dtype([self.dtype, other.dtype])
        else:
            res_type = None
        result = super()._arith_method(other, op)
        # use the appropriate type for the result
        if res_type is not None:
            result = type(self)(result, dtype=res_type)
        return result


@pd.api.extensions.register_extension_dtype
class ScaledOffsetDtype(pd.api.extensions.ExtensionDtype):
    _metadata = ('unit', 'um_per_pixel', 'dtype')
    _na_value = np.nan
    name = 'ScaledOffset'

    def __init__(self, unit: Optional[Literal['pixels', 'um']], um_per_pixel: Optional[float],
                 dtype: Union[npt.DTypeLike, NumpyEADtype, None] = None):
        self.sub_dtype = NumpyEADtype(dtype)
        self.dtype: np.dtype = self.sub_dtype.numpy_dtype
        self.unit: Optional[Literal['pixels', 'um']] = unit
        self.um_per_pixel = um_per_pixel
    
    @property
    def type(self) -> str:
        return self.sub_dtype.type

    @property
    def numpy_dtype(self) -> np.dtype:
        return self.sub_dtype.numpy_dtype
    
    @property
    def na_value(self) -> float:
        return self._na_value

    def construct_array_type(self) -> Type[pd.api.extensions.ExtensionArray]:
        return ScaledArray
    
    def __repr__(self) -> str:
        if self.unit is None:
            return 'ScaledOffset'
        else:
            classname = 'ScaledPixels' if self.unit == 'pixels' else 'ScaledUm'

        if self.um_per_pixel is None:
            return f'{classname}(None, {self.sub_dtype.name})'
        else:
            return f'{classname}({self.um_per_pixel}, {self.sub_dtype.name})'
    
    def _get_common_dtype(self, dtypes: Sequence[Union[np.dtype, pd.api.extensions.ExtensionDtype]]):
        """
        This allows e.g. a row to still have a ScaledOffset dtype if the column types are different
        instances of ScaledOffset. For example if we have a row of um measurements with different resolutions,
        we should still be able to get the distance.
        """
        common_unit = self.unit
        common_umpp = self.um_per_pixel
        dts = set([self.dtype])
        all_scaled_offset = True

        for dt in dtypes:
            if not isinstance(dt, ScaledOffsetDtype):
                all_scaled_offset = False
                break
            
            if dt.unit != common_unit:
                common_unit = None

            if dt.um_per_pixel != common_umpp:
                common_umpp = None

            dts.add(dt.dtype)
        
        if not all_scaled_offset:
            # fall back to numpy common type
            dtypes_base = [dt.sub_dtype if isinstance(dt, ScaledOffsetDtype) else dt for dt in dtypes]
            return self.sub_dtype._get_common_dtype(dtypes_base)
        else:   
            common_dt = np.result_type(*dts)
            return ScaledOffsetDtype(unit=common_unit, um_per_pixel=common_umpp, dtype=common_dt)

# for convenience:
def ScaledPixels(um_per_pixel: Optional[float], dtype: Union[npt.DTypeLike, NumpyEADtype, None] = None):
        return ScaledOffsetDtype(unit='pixels', um_per_pixel=um_per_pixel, dtype=dtype)

def ScaledUm(um_per_pixel: Optional[float], dtype: Union[npt.DTypeLike, NumpyEADtype, None] = None):
        return ScaledOffsetDtype(unit='um', um_per_pixel=um_per_pixel, dtype=dtype)


class ScaledSeries(pd.Series):
    """This is a series of a scaleddataframe, which just adds a couple of methods to a regular Series"""
    @property
    def _constructor(self):
        return ScaledSeries

    @property
    def _constructor_expanddim(self):
        return ScaledDataFrame
    
    @property
    def unit(self) -> Optional[Literal['pixels', 'um']]:
        if not isinstance(self.dtype, ScaledOffsetDtype):
            return None
        return self.dtype.unit
    
    @property
    def um_per_pixel(self) -> Optional[float]:
        if not isinstance(self.dtype, ScaledOffsetDtype):
            return None
        return self.dtype.um_per_pixel

    @um_per_pixel.setter
    def um_per_pixel(self, new: Optional[float]):
        if not isinstance(self.dtype, ScaledOffsetDtype):
            raise RuntimeError('Cannot set um_per_pixel on series that is not a ScaledOffset type')
        self.dtype.um_per_pixel = new
    
    def to_unit(self, unit: Literal['pixels', 'um'], always_copy=False) -> 'ScaledSeries':
        """Return a series that is converted to the given unit, copying if requested"""
        if not isinstance(self.dtype, ScaledOffsetDtype):
            raise TypeError('Cannot convert series that is not a ScaledOffset type')

        if self.dtype.unit != unit:
            if self.dtype.unit is None or self.dtype.um_per_pixel is None:
                raise TypeError('Cannot convert series of heterogeneous ScaledOffset type')
            op = np.multiply if unit == 'um' else np.divide
            converted_vals = op(self.to_numpy(), self.dtype.um_per_pixel)
            dtype = ScaledOffsetDtype(unit=unit, um_per_pixel=self.dtype.um_per_pixel, dtype=converted_vals.dtype)
            return type(self)(data=converted_vals, index=self.index, dtype=dtype, name=self.name, copy=False)
        elif always_copy:
            return type(self)(self.copy())
        else:
            return self

    def to_um(self, always_copy=False) -> 'ScaledSeries':
        """Return a series that is converted to um, copying if requested"""
        return self.to_unit('um', always_copy=always_copy)
        
    def to_pixels(self, always_copy=False) -> 'ScaledSeries':
        """Return a series that is converted to pixels, copying if requested"""
        return self.to_unit('pixels', always_copy=always_copy)
    
    def distance(self) -> float:
        if not isinstance(self.dtype, ScaledOffsetDtype):
            raise TypeError('Cannot get distance of series that is not a ScaledOffset type')
        if self.dtype.unit is None:
            raise TypeError('Cannot get distance of series with heterogneous units')
        return float(np.linalg.norm(self.to_numpy()))


class ScaledDataFrame(pd.DataFrame):
    """dataframe whose columns are ScaledOffset variables"""
    @property
    def _constructor(self):
        return ScaledDataFrame
    
    @property
    def _constructor_sliced(self):
        return ScaledSeries
    
    @property
    def um_per_pixel(self) -> dict[str, Optional[float]]:
        dtypes = self._check_all_scaled()
        return {col: dt.um_per_pixel for col, dt in zip(self.columns, dtypes)}
    
    def _check_all_scaled(self) -> Sequence[ScaledOffsetDtype]:
        dtypes: list[ScaledOffsetDtype] = []
        for dtype in self.dtypes:
            if not isinstance(dtype, ScaledOffsetDtype):
                raise TypeError('Not all columns are of type ScaledOffset')
            dtypes.append(dtype)
        return dtypes

    def __getitem__(self, index):
        """Override default behavior to select the row as a dataframe when indexing by a single int"""
        if isinstance(index, int):
            return self.iloc[[index]]
        return super().__getitem__(index)

    def iterpoints(self) -> 'Iterable[tuple[Hashable, ScaledDataFrame]]':
        """Iterate over points in the dataframe (i.e., rows) without collapsing them into Series"""
        for ind in self.index:
            yield ind, self.loc[[ind]]

    def to_numpy(self, dtype: Optional[np.dtype] = None, copy=False, na_value=ScaledOffsetDtype._na_value) -> np.ndarray:
        # set a default dtype for converting to array to get around https://github.com/pandas-dev/pandas/issues/22791
        if dtype is None and self.shape[1] > 0:
            for col_dtype in self.dtypes:
                if isinstance(col_dtype, ScaledOffsetDtype):
                    common_dtype = col_dtype._get_common_dtype(list(self.dtypes))
                    dtype = common_dtype.numpy_dtype
                    break
        return super().to_numpy(dtype=dtype, copy=copy, na_value=na_value)
    
    def to_unit(self, unit: Literal['pixels', 'um'], always_copy=False) -> 'ScaledDataFrame':
        """Return version with all columns in given unit"""
        df = self.copy(deep=always_copy)
        for colname in df.columns:
            col: ScaledSeries = getattr(df, colname)
            if isinstance(col.dtype, ScaledOffsetDtype) and col.dtype.unit != unit:
                setattr(df, colname, col.to_unit(unit, always_copy=always_copy))
        return df

    def to_um(self, always_copy=False) -> 'ScaledDataFrame':
        """Return version with all columns in um"""
        return self.to_unit('um', always_copy=always_copy)

    def to_pixels(self, always_copy=False) -> 'ScaledDataFrame':
        """Return version with all columns in um"""
        return self.to_unit('pixels', always_copy=always_copy)
    
    def distance(self, unit: Optional[Literal['um', 'pixels']] = None) -> ScaledSeries:
        """
        Compute distance across each row. If unit is specified, will attempt to convert
        to this unit first. 
        """
        if self.shape[1] == 0:
            raise ValueError('Cannot compute distance with 0 columns')
        
        if unit is not None:
            df = self.to_unit(unit, always_copy=False)
        else:
            df = self

        scaled_dtypes = df._check_all_scaled()
        common_dtype = scaled_dtypes[0]._get_common_dtype(scaled_dtypes)
        unit = common_dtype.unit
        if unit is None:
            raise ValueError('Cannot compute distance with heterogeneous units; specify unit to convert')                   
        
        # OK we are safe to just compute the distance
        dist = np.linalg.norm(df.to_numpy(), axis=1)
        return ScaledSeries(data=dist, index=df.index, dtype=common_dtype, name=f'distance ({unit})', copy=False)


# convenience functions to make dataframes with scaling info
def make_scaled_df(
        data: Union[Mapping[str, npt.ArrayLike], pd.DataFrame, npt.ArrayLike], unit: Literal['pixels', 'um'],
        dim_names: Optional[Sequence[str]] = None, index=None,
        pixel_size: Union[Mapping[str, Optional[float]], Sequence, pd.Series, float, None] = None) -> ScaledDataFrame:
    """Construct a ScaledDataFrame with the given data and potentially different pixel sizes"""
    # convert what was passed into a mapping of names to (data, Series)
    # first we need to be able to iterate over the columns and column names
    if isinstance(data, (Mapping, pd.DataFrame)):
        if dim_names is not None:
            raise TypeError('dim_names not expected for mapping or dataframe types')
        if isinstance(data, Mapping):
            dim_names = list(data.keys())
            columns = (np.asarray(v) for v in data.values())
        else:
            dim_names = list(data.columns)
            columns = (data.loc[:, name].to_numpy() for name in dim_names)
    else:
        array = np.atleast_2d(data)
        if array.ndim > 2:
            raise TypeError('Data should be a matrix (at most 2D)')
        columns = array.T

        if dim_names is None:
            # try to grab variables from pixel_size
            if isinstance(pixel_size, Mapping) and len(pixel_size) == array.shape[1]:
                dim_names = list(pixel_size.keys())
            elif isinstance(pixel_size, pd.Series) and len(pixel_size) == array.shape[1]:
                dim_names = list(pixel_size.index)
            else:
                raise RuntimeError('Cannot infer dimension names - dim_names not provided and cannot infer from pixel_size')

    df_builder: dict[str, pd.Series] = {}
    for i, (dim_name, column) in enumerate(zip(dim_names, columns)):
        # figure out the dtype for each column
        if isinstance(pixel_size, Mapping):
            this_umpp = pixel_size[dim_name]
        elif isinstance(pixel_size, Sequence):
            this_umpp = pixel_size[i]
        elif isinstance(pixel_size, pd.Series):
            if pixel_size.index.dtype.kind == 'i':
                this_umpp = pixel_size.iloc[i]
            else:
                this_umpp = pixel_size.loc[dim_name]
        else:
            this_umpp = pixel_size
        dtype = ScaledOffsetDtype(unit=unit, um_per_pixel=this_umpp, dtype=column.dtype)
        df_builder[dim_name] = pd.Series(column, dtype=dtype)
    df = ScaledDataFrame(df_builder)
    if index is not None:
        df.index = index
    return df


def make_pixel_df(data: Union[Mapping[str, npt.ArrayLike], pd.DataFrame, npt.ArrayLike], dim_names: Optional[Sequence[str]] = None,
                  index=None, pixel_size: Union[Mapping[str, Optional[float]], Sequence, pd.Series, float, None] = None) -> ScaledDataFrame:
    """
    Create a dataframe of pixel measurements with scaling information. Inputs:
     - data: existing data containing values in units of pixels, in each of one or more columns corresponding to 
             image dimensions. Dicts, dataframes, and array-likes are suppored. If passing an array-like,
             also pass dim_names to specify the name of each dimension, or it may also be inferred from
             pixel_size.
     - dim_names: sequence of dimension names corresponding to columns of an array-like data input.
     - pixel_size: dict or sequence of um per pixel factors for each dimension in data.
    """
    return make_scaled_df(data, unit='pixels', dim_names=dim_names, index=index, pixel_size=pixel_size)


def make_um_df(data: Union[Mapping[str, npt.ArrayLike], pd.DataFrame, npt.ArrayLike], dim_names: Optional[Sequence[str]] = None,
               index=None, pixel_size: Union[Mapping[str, Optional[float]], Sequence, pd.Series, float, None] = None) -> ScaledDataFrame:
    """
    Create a dataframe of um measurements with pixel scaling information. Inputs:
     - data: existing data containing values in units of um, in each of one or more columns corresponding to 
             image dimensions. Dicts, dataframes, and array-likes are suppored. If passing an array-like,
             also pass dim_names to specify the name of each dimension, or it may also be inferred from
             pixel_size.
     - dim_names: sequence of dimension names corresponding to columns of an array-like data input.
     - pixel_size: dict or sequence of um per pixel factors for each dimension in data.
    """
    return make_scaled_df(data, unit='um', dim_names=dim_names, index=index, pixel_size=pixel_size)
