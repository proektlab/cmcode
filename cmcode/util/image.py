# Plotting/image helpers
from dataclasses import dataclass
from functools import partialmethod, partial, reduce
import math
from typing import Union, Sequence, Optional, Callable, Any

import cv2
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import ArrayLike
from optype import CanFloat
import optype.numpy as onp
import pandas as pd
from scipy import ndimage, interpolate
from scipy.ndimage import median_filter

from cmcode.util.scaled import ScaledDataFrame, ScaledSeries, ScaledPixels


COLORS = {
    'red':    (1., 0., 0.),
    'green':  (0., 1., 0.),
    'blue':   (0., 0., 1.),
    'yellow': (1., 1., 0.),
    'cyan':   (0., 1., 1.),
    'magenta':(1., 0., 1.)
}
# add single-letter aliases
single_letter_colors = {}
for (name, val) in COLORS.items():
    if name[0] not in single_letter_colors:
        single_letter_colors[name[0]] = val
COLORS.update(single_letter_colors)


@dataclass(init=False, frozen=True)
class BorderSpec:
    """For specifying the border of a 2D image unambiguously"""
    left_subpix: float
    right_subpix: float
    top_subpix: float
    bottom_subpix: float

    def __init__(
            self, left: CanFloat = 0., right: CanFloat = 0.,
            top: CanFloat = 0., bottom: CanFloat = 0.):
        object.__setattr__(self, 'left_subpix', float(left))
        object.__setattr__(self, 'right_subpix', float(right))
        object.__setattr__(self, 'top_subpix', float(top))
        object.__setattr__(self, 'bottom_subpix', float(bottom))

    # if just accessing left/right/top/bottom, round up to nearest int
    @property
    def left(self) -> int:
        return math.ceil(self.left_subpix)
    
    @property
    def right(self) -> int:
        return math.ceil(self.right_subpix)
    
    @property
    def top(self) -> int:
        return math.ceil(self.top_subpix)
    
    @property
    def bottom(self) -> int:
        return math.ceil(self.bottom_subpix)
    
    # make compatible with old version when pickling/unpickling
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for side in ['left', 'right', 'top', 'bottom']:
            state[side] = getattr(self, side)
        return state

    def __setstate__(self, state: dict[str, Any]):
        for side in ['left', 'right', 'top', 'bottom']:
            if side + '_subpix' not in state:
                state[side + '_subpix'] = float(state[side])
        self.__dict__.update(state)

    
    def _arithmetic_op(self, op: str, other: Union[CanFloat, 'BorderSpec']) -> 'BorderSpec':
        """Apply arithmetic operation to all sides"""
        sides = ['left', 'right', 'top', 'bottom']
        op_method = getattr(float, op)
        if isinstance(other, CanFloat):
            res = BorderSpec(**{side: op_method(getattr(self, side + '_subpix'), float(other)) for side in sides})
        else:
            res = BorderSpec(**{side: op_method(
                getattr(self, side + '_subpix'), getattr(other, side + '_subpix')
                ) for side in sides})
        return BorderSpec.max(res, BorderSpec.equal(0))

    def _comparison_op(self, op: str, other: Union[CanFloat, 'BorderSpec']) -> bool:
        """Delegate comparisons to int comparisons of all sides"""
        sides = ['left_subpix', 'right_subpix', 'top_subpix', 'bottom_subpix']
        op_method = getattr(float, op)
        if isinstance(other, (int, CanFloat)):
            return all(op_method(getattr(self, side), other) for side in sides)
        else:
            return all(op_method(getattr(self, side), getattr(other, side)) for side in sides)

    for op in ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']:
        vars()[op] = partialmethod(_comparison_op, op)

    @classmethod
    def equal(cls, border: CanFloat) -> 'BorderSpec':
        return cls(left=border, right=border, top=border, bottom=border)
    
    @classmethod
    def maximal(cls, shape: tuple[int, int]) -> 'BorderSpec':
        return cls(left=shape[1], right=shape[1], top=shape[0], bottom=shape[0])
    
    @classmethod
    def combine(cls, fn: Callable[[float, float], float],
                border1: Union[None, CanFloat, 'BorderSpec'], border2: Union[None, CanFloat, 'BorderSpec']) -> 'BorderSpec':
        """Combine 2 borders into a BorderSpec by applying the given function to each pair of matching dimensions.""" 
        if isinstance(border1, CanFloat):
            border1 = cls.equal(border1)
        if isinstance(border2, CanFloat):
            border2 = cls.equal(border2)

        if border1 is None:
            if border2 is None:
                raise TypeError('Both borders cannot be None')
            return border2
        if border2 is None:
            return border1
        
        return BorderSpec(
            left=fn(border1.left_subpix, border2.left_subpix),
            right=fn(border1.right_subpix, border2.right_subpix),
            top=fn(border1.top_subpix, border2.top_subpix),
            bottom=fn(border1.bottom_subpix, border2.bottom_subpix)
        )

    @classmethod
    def max(cls, first: Union[CanFloat, 'BorderSpec'], *others: Union[CanFloat, 'BorderSpec']) -> 'BorderSpec':
        """Make a BorderSpec for the maximum border along each side"""
        if isinstance(first, CanFloat):
            first = cls.equal(first)
        return reduce(partial(cls.combine, max), others, first)

    @classmethod
    def min(cls, first: Union[CanFloat, 'BorderSpec'], *others: Union[CanFloat, 'BorderSpec']) -> 'BorderSpec':
        """Make a BorderSpec for the minimum border along each side"""
        if isinstance(first, CanFloat):
            first = cls.equal(first)
        return reduce(partial(cls.combine, min), others, first)
    
    def is_center_nonempty(self, shape: tuple[int, int]) -> bool:
        """Whether there are any pixels left in the center, given shape"""
        center_shape = self.center_shape(shape)
        return center_shape[0] > 0 and center_shape[1] > 0
    
    def increased(self, other: Union[CanFloat, 'BorderSpec'], shape: Optional[tuple[int, int]] = None) -> 'BorderSpec':
        """Increase borders, limiting result to maximal borders if shape is given"""
        res = self._arithmetic_op('__add__', other)
        if shape is not None:
            res = BorderSpec.min(self, BorderSpec.maximal(shape=shape))
        return res
    
    def decreased(self, other: Union[CanFloat, 'BorderSpec']) -> 'BorderSpec':
        """Decrease borders"""
        return self._arithmetic_op('__sub__', other)
    
    def enclosing_square(self, shape: tuple[int, int]) -> 'BorderSpec':
        """
        Transform into another BorderSpec that removes fewer or equal rows and columns than this one
        and has a square center, if possible. Does nothing if the center is empty.
        """
        if not self.is_center_nonempty(shape):
            return self
        
        height, width = self.center_shape_subpix(shape)
        if height == width:
            return self
        
        def distribute_difference(diff: float, border1: float, border2: float) -> tuple[float, float]:  # (newborder1, newborder2)
            half_diff = diff / 2
            if border1 >= half_diff and border2 >= half_diff:
                # typical case
                return border1 - half_diff, border2 - half_diff
            elif border1 + border2 >= diff:
                # can make it square, just won't be symmetric
                if border1 < half_diff:
                    return 0, border2 - (diff - border1)
                else:
                    return border1 - (diff - border2), 0
            else:
                # just return minimum borders
                return 0, 0

        if height > width:
            new_left, new_right = distribute_difference(height - width, self.left_subpix, self.right_subpix)
            return BorderSpec(top=self.top_subpix, bottom=self.bottom_subpix, left=new_left, right=new_right)
        else:
            new_top, new_bottom = distribute_difference(width - height, self.top_subpix, self.bottom_subpix)
            return BorderSpec(top=new_top, bottom=new_bottom, left=self.left_subpix, right=self.right_subpix)


    def slices(self, shape: Sequence[int]) -> tuple[slice, slice]:
        """Make indexing slices for these borders given an image shape"""
        shape2d = shape[:2]
        return (slice(self.top, shape2d[0]-self.bottom), slice(self.left, shape2d[1]-self.right))
    
    def center_shape(self, shape: tuple[int, int]) -> tuple[int, int]:
        """Make shape of center with borders removed"""
        return (max(shape[0] - self.top - self.bottom, 0),
                max(shape[1] - self.left - self.right, 0))

    def center_shape_subpix(self, shape: tuple[int, int]) -> tuple[float, float]:
        """Make shape of center with borders removed, using subpixels"""
        return (max(shape[0] - self.top_subpix - self.bottom_subpix, 0),
                max(shape[1] - self.left_subpix - self.right_subpix, 0))

    def flatmask(self, shape: tuple[int, ...], order: onp.OrderKACF = 'F') -> np.ndarray:
        """Make a boolean vector for masking a flattened array within these borders."""
        mask = np.zeros(shape, dtype=bool)
        mask[self.slices(shape)] = True
        return mask.ravel(order=order)

@dataclass(frozen=True)
class BorderedImage:
    """Image that also keeps track of some border through shifts and remaps"""
    image: onp.Array2D
    border: BorderSpec

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray):
            object.__setattr__(self, 'image', np.asarray(self.image))

        if isinstance(self.border, CanFloat):
            object.__setattr__(self, 'border', BorderSpec.equal(float(self.border)))

    # for convenience, provide shape directly to BorderSpec methods
    @property
    def plane_shape(self) -> tuple[int, int]:
        return (self.image.shape[0], self.image.shape[1]) 
    
    @property
    def center_shape(self) -> tuple[int, int]:
        return self.border.center_shape(self.plane_shape)
    
    @property
    def center_shape_subpix(self) -> tuple[float, float]:
        return self.border.center_shape_subpix(self.plane_shape)
    
    @property
    def center(self) -> np.ndarray:
        """View into the center of this image (according to border)"""
        slices = self.border.slices(self.plane_shape)
        return self.image[slices]
    

    def shifted(self, x_shift: float, y_shift: float) -> 'BorderedImage':
        """Shift image and also adjust border"""
        shifted_image = shift_image(self.image, x_shift=x_shift, y_shift=y_shift)
        
        max_y, max_x = self.image.shape
        new_left = np.clip(self.border.left_subpix + x_shift, 0, max_x)
        new_right = np.clip(self.border.right_subpix - x_shift, 0, max_x)
        new_top = np.clip(self.border.top_subpix + y_shift, 0, max_y)
        new_bottom = np.clip(self.border.bottom_subpix - y_shift, 0, max_y)
        shifted_border = BorderSpec(left=new_left, right=new_right, top=new_top, bottom=new_bottom)
        
        return BorderedImage(image=shifted_image, border=shifted_border)
    

    def remapped(self, x_remap: Optional[onp.Array2D[np.floating]], y_remap: Optional[onp.Array2D[np.floating]]) -> 'BorderedImage':
        """Nonrigidly remap image and also adjust border"""
        remapped_image = remap_image(self.image, x_remap=x_remap, y_remap=y_remap)
        if x_remap is None:
            assert y_remap is None  # should be checked in remap_image
            return BorderedImage(image=remapped_image, border=self.border)
        assert y_remap is not None # should be checked in remap_image

        # infer new borders to fully contain shifted image
        # first get a boolean mask of where in the new image corresponds to valid pixels of the old image
        sz_y, sz_x = self.image.shape
        x_remap = np.asarray(x_remap)
        y_remap = np.asarray(y_remap)
        within_borders_x = (x_remap >= self.border.left_subpix) & (x_remap <= sz_x - 1 - self.border.right_subpix)
        within_borders_y = (y_remap >= self.border.top_subpix) & (y_remap <= sz_y - 1 - self.border.bottom_subpix)
        within_borders = within_borders_x & within_borders_y

        # then collapse along x and y and make new borders
        inds_within_x = np.flatnonzero(np.any(within_borders, axis=0))
        new_left = np.min(inds_within_x, initial=sz_x)
        new_right = np.min(sz_x - 1 - inds_within_x, initial=sz_x)

        inds_within_y = np.flatnonzero(np.any(within_borders, axis=1))
        new_top = np.min(inds_within_y, initial=sz_y)
        new_bottom = np.min(sz_y - 1 - inds_within_y, initial=sz_y)

        remapped_border = BorderSpec(left=new_left, right=new_right, top=new_top, bottom=new_bottom)
        return BorderedImage(image=remapped_image, border=remapped_border)


def colorize(im: onp.Array2D[np.number], color: Union[Sequence[float], str],
             clip_percentile: Union[float, tuple[float, float]] = (40, 99.7)) -> onp.Array3D[np.float32]:
    """
    Helper function to create an RGB image from a single-channel image using a 
    specific color.
    Source: https://bioimagebook.github.io/chapters/1-concepts/4-colors/python.html
    """
    # Check that we do just have a 2D image
    if len(s := im.shape) > 2 and s[2] != 1:
        raise ValueError('This function expects a single-channel image!')

    # Check string colors
    if isinstance(color, str):
        try:
            color_seq = COLORS[color]
        except KeyError:
            raise ValueError(f'{color} is not a recognized color')
    else:
        color_seq = color

    if not isinstance(clip_percentile, Sequence):
        clip_percentile = (clip_percentile, 100-clip_percentile)

    # Rescale the image according to how we want to display it
    im_scaled = im.astype(np.float32)
    im_scaled -= np.percentile(im_scaled, clip_percentile[0])
    im_scaled /= np.percentile(im_scaled, clip_percentile[1])
    im_scaled = np.clip(im_scaled, 0, 1)
    
    # Need to make sure we have a channels dimension for the multiplication to work
    im_scaled = np.atleast_3d(im_scaled)
    
    # Reshape the color (here, we assume channels last)
    color_seq = np.asarray(color_seq).reshape((1, 1, -1))
    return im_scaled * color_seq


def make_merge(im1: np.ndarray, im2: np.ndarray,
               color1: Union[Sequence[float], str] = 'g',
               color2: Union[Sequence[float], str] = 'r',
               clip_percentile: Union[float, tuple[float, float]] = (40, 99.7)) -> np.ndarray:
    # colorize 2 images and merge them
    rgb1 = colorize(im1, color1, clip_percentile=clip_percentile)
    rgb2 = colorize(im2, color2, clip_percentile=clip_percentile)
    return np.clip(rgb1 + rgb2, 0, 1)


def shift_image(image: onp.ToFloat2D, x_shift: float, y_shift: float) -> np.ndarray:
    """Shift image by a number of pixels in X and Y"""
    shifts = (y_shift, x_shift)
    return ndimage.shift(image, shifts)


def shift_image_location(image: onp.ToFloat2D, start_loc: ScaledDataFrame, end_loc: ScaledDataFrame):
    """Shift image from start_loc to end_loc (both must have a single row)"""
    if start_loc.shape[0] != 1 or end_loc.shape[0] != 1:
        raise ValueError('start_loc and end_loc must both have a single row')

    start_pixels = start_loc.to_pixels().loc[:, ['x', 'y']].to_numpy().ravel()
    end_pixels = end_loc.to_pixels().loc[:, ['x', 'y']].to_numpy().ravel()
    x_shift, y_shift = end_pixels - start_pixels
    return shift_image(image, x_shift=float(x_shift), y_shift=float(y_shift))


def remap_image(image: np.ndarray, x_remap: Optional[onp.Array2D[np.floating]], y_remap: Optional[onp.Array2D[np.floating]]):
    """Use CV2 to remap an image according to remap function as returned from register_ROIs"""
    if x_remap is None or y_remap is None:
        if y_remap is not None or x_remap is not None:
            raise ValueError('x_remap and y_remap should both be either defined or not')
        return image
    return cv2.remap(image.astype(np.float32), x_remap, y_remap, cv2.INTER_CUBIC)


def invert_mapping(x_remap: onp.Array2D[np.floating], y_remap: onp.Array2D[np.floating]
                   ) -> tuple[onp.Array2D[np.floating], onp.Array2D[np.floating]]:
    y_grid: onp.Array2D[np.floating]
    x_grid: onp.Array2D[np.floating]
    y_grid, x_grid = np.meshgrid(
        np.arange(x_remap.shape[0], dtype=x_remap.dtype),
        np.arange(x_remap.shape[1], dtype=x_remap.dtype), indexing='ij')
    x_remap_inv = -(x_remap - x_grid) + x_grid
    y_remap_inv = -(y_remap - y_grid) + y_grid
    return x_remap_inv, y_remap_inv


def compose_mappings(*xy_remaps: Union[tuple[onp.Array2D, onp.Array2D], tuple[None, None]]
                     ) -> Union[tuple[onp.Array2D, onp.Array2D], tuple[None, None]]:
    """
    Compose together all of the passed mappings, each of which should be an (X, Y)
    tuple and have the same shape. e.g. [A -> B, B -> C, C -> D] becomes A -> D.
    """
    non_nops = [remap for remap in xy_remaps if remap[0] is not None]

    if len(non_nops) == 0:
        return None, None
    
    cum_remap_x, cum_remap_y = non_nops[0]
    for next_x, next_y in non_nops[1:]:
        cum_remap_x = remap_image(cum_remap_x, x_remap=next_x, y_remap=next_y)
        cum_remap_y = remap_image(cum_remap_y, x_remap=next_x, y_remap=next_y)
    
    return cum_remap_x, cum_remap_y


def inverse_remap_image(image: np.ndarray, x_remap: Optional[onp.Array2D], y_remap: Optional[onp.Array2D]):
    """Use CV2 to inverse-remap an image according to remap function as returned from register_ROIs"""
    if x_remap is None or y_remap is None:
        if y_remap is not None or x_remap is not None:
            raise ValueError('x_remap and y_remap should both be either defined or not')
        return image
    return remap_image(image, *invert_mapping(x_remap=x_remap, y_remap=y_remap))


def remap_points(points: onp.Array2D, x_remap: Optional[onp.Array2D], y_remap: Optional[onp.Array2D]) -> np.ndarray:
    """
    Map a set of points from one coordinate system to another using a nonrigid mapping
    There must be 2 columns and they are assumed to be (Y, X).
    """
    points = np.atleast_2d(points)
    if points.ndim > 2 or points.shape[1] != 2:
        raise ValueError('Points should be a matrix with 2 columns (Y and X)')
    yvals = points[:, 0]
    xvals = points[:, 1]

    if x_remap is None or y_remap is None:
        if y_remap is not None or x_remap is not None:
            raise ValueError('x_remap and y_remap should both be either defined or not')
        mapped_vals = (yvals, xvals)
    else:
        # take inverse of remapping so that we can do regular-grid interpolation
        # we want to map (y, x) locations to the coordinates in the new space (given by the inverted mapping)
        remap_inv_x, remap_inv_y = invert_mapping(x_remap=x_remap, y_remap=y_remap)
        interpolants = [
            interpolate.RectBivariateSpline(range(x_remap.shape[0]), range(x_remap.shape[1]), remap_dim)
            for remap_dim in (remap_inv_y, remap_inv_x)
        ]
        mapped_vals = [interpolant(yvals, xvals, grid=False) for interpolant in interpolants]
    
    return np.column_stack(mapped_vals)


def remap_points_from_df(df: pd.DataFrame, x_remap: Optional[np.ndarray],
                         y_remap: Optional[np.ndarray]) -> pd.DataFrame:
    """
    Map a set of points from one coordinate system to another using a nonrigid mapping
    There must be 2 columns called x and y. A copy will be returned with x and y mapped
    and all other columns unchanged.
    If input is a ScaledDataFrame, this function handles converting the x and y columns
    to pixels to do the mapping and then back to the original unit(s).
    """
    if 'y' not in df or 'x' not in df:
        raise ValueError('DataFrame must have y and x values')
    
    yxvals = df.loc[:, ['y', 'x']]
    if isinstance(yxvals, ScaledDataFrame):
        orig_dtypes = yxvals._check_all_scaled()        
        yxvals = yxvals.to_pixels()
    else:
        orig_dtypes = None

    mapped_vals = remap_points(yxvals.to_numpy(), x_remap=x_remap, y_remap=y_remap)
    df_mapped = df.copy()

    if orig_dtypes is not None:
        assert isinstance(yxvals, ScaledDataFrame)

        for orig_dtype, col_name, values in zip(orig_dtypes, ['y', 'x'], mapped_vals.T):
            col = ScaledSeries(
                values, dtype=ScaledPixels(um_per_pixel=orig_dtype.um_per_pixel, dtype=orig_dtype.dtype))
            if (unit := orig_dtype.unit) is None:
                raise RuntimeError(f'Inconsistent unit for columns {col_name}')
            setattr(df_mapped, col_name, col.to_unit(unit))
    else:
        df_mapped.loc[:, ['y', 'x']] = mapped_vals
    
    return df_mapped


def preprocess_proj_for_seed(mean_img: np.ndarray, med_w: int = 25,
                             borders: list[BorderSpec] = [BorderSpec.equal(0)]) -> np.ndarray:
    """Make a brightness-normalized mean image for CNMF seed calculation"""
    concat_planes = len(borders)

    mean_frac_img_planes: list[np.ndarray] = []
    # split apart to process each plane separately
    for plane, border in zip(np.split(mean_img, concat_planes, axis=1), borders):
        # exclude the border, if any, for normalization
        center_slices = border.slices(plane.shape)
        center_part = plane[center_slices]
        local_med_img = median_filter(center_part, size=med_w)
        mean_frac_img = center_part / local_med_img
        mean_frac_img[np.isnan(mean_frac_img)] = 0
        plane_corrected = np.zeros_like(plane)
        plane_corrected[center_slices] = mean_frac_img
        mean_frac_img_planes.append(plane_corrected)
    return np.concatenate(mean_frac_img_planes, axis=1)


def calc_weighted_median(data: np.ndarray, weights: np.ndarray, axis=0):
    """See: https://github.com/nudomarinero/wquantiles/blob/master/wquantiles.py"""
    ind_sorted = np.argsort(data, axis=axis)
    data_sorted = np.take_along_axis(data, ind_sorted, axis=axis)
    weights_sorted = weights[ind_sorted]

    Sn = np.cumsum(weights_sorted, axis=axis)
    Pn = (Sn - 0.5 * weights_sorted) / np.take(Sn, -1, axis=axis)

    # loop to use np.interp on vectors
    Ni, Nj = data.shape[:axis], data.shape[axis+1:]
    res = np.empty(Ni + Nj)
    for ii in np.ndindex(Ni):
        for jj in np.ndindex(Nj):
            res[ii + jj] = np.interp(0.5, Pn[ii + (slice(None),) + jj], data_sorted[ii + (slice(None),) + jj])

    return res


def imshow_scaled(ax: Axes, image: ArrayLike, vmin_pct=40., vmax_pct=99.7, axes_off=True, **kwargs):
    """Convenience function b/c I'm tired of typing np.percentile"""
    vmin, vmax = np.percentile(np.ravel(image), [vmin_pct, vmax_pct])
    ax.imshow(image, vmin=float(vmin), vmax=float(vmax), **kwargs)
    if axes_off:
        ax.set_axis_off()