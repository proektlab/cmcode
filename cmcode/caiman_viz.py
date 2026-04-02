from copy import deepcopy
from functools import partial
import logging
import math
import time
from typing import Optional, Union, cast, Callable, Literal, Any, Sequence

from bokeh.document import Document
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, CategoricalSlider, LinearColorMapper, CategoricalColorMapper
import bokeh.palettes
from bokeh.plotting import figure
from bokeh.themes import built_in_themes

import fastplotlib as fpl
from fastplotlib.layouts._frame._jupyter_output import JupyterOutputContext
import holoviews as hv
from holoviews import opts
from holoviews.operation import Operation
from holoviews.streams import Stream, param
from IPython.display import display
from ipywidgets import (HBox, VBox, IntSlider, Label, Layout, Button, Image as ImageWidget, 
                        Output, SelectMultiple, HTML, CallbackDispatcher)
import jupyter_bokeh as jbk
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType
import panel as pn
from pathlib import Path
from PIL import Image
from scipy import sparse
from sidecar import Sidecar

import caiman
from caiman.utils import sbx_utils
from caiman.source_extraction.cnmf import CNMF
from mesmerize_core import MCorrExtensions, CaimanSeriesExtensions
from mesmerize_viz._cnmf import CNMFVizContainer, CNMFDataFrameVizExtension, EvalController

import cmcode
from cmcode import caiman_analysis as cma, alignment
from cmcode.cnmf_ext import CNMFExt, MetricInfo, clear_cnmf_cache
from cmcode.cmcustom import my_get_contours, my_plot_contours, compute_matching_performance
from cmcode.mcorr import MCResult
from cmcode.util.footprints import (FootprintsPerPlane, collapse_footprints_to_xy,
                                    map_footprints, maxproj_per_cell, footprint_interpolator_per_cell)
from cmcode.util.image import remap_image, make_merge, BorderSpec, preprocess_proj_for_seed
from cmcode.util.sbx_data import average_raw_frames, find_sess_sbx_files
from cmcode.util.types import MaybeSparse

pn.extension()
hv.extension('bokeh')  # type: ignore

BokehPalette = Callable[[int], Sequence[str]]


class ButtonWithFeedback(HBox):
    """a button that shows a loading indicator and check/X emoji to give feedback on progress of the callback"""
    loading_gif_path = Path(next(iter(cmcode.__path__))).parent / 'assets' / 'loading.gif'

    def __init__(self, button_text: str):
        self.button = Button(description=button_text)
        self.feedback = Output()
        super().__init__([self.button, self.feedback])

        with open(self.loading_gif_path, 'rb') as loading_gif:
            loading_gif_data = loading_gif.read()
        self.loading = ImageWidget(value=loading_gif_data, format='gif', layout=Layout(width='28px'))
        self.done = Label(value='\u2714', style={'color': 'green', 'font-size': '28px'})
        self.failed = HBox([
            Label(value='\u2716', style={'color': 'red', 'font-size': '28px'}),
            Label(value='See console for details')
        ])

        self.click_handlers = CallbackDispatcher()
        self.button.on_click(self.button_cb)

    def on_click(self, callback: Callable[[Any], None], remove=False):
        self.click_handlers.register_callback(callback, remove=remove)

    def button_cb(self, obj: Any) -> None:
        self.feedback.clear_output()
        with self.feedback:
            display(self.loading)

        for callback in self.click_handlers.callbacks:
            try:
                callback(obj)
            except:
                # show X and directions to look at console
                self.feedback.clear_output()
                with self.feedback:
                    display(self.failed)
                    break
        else:  # (if no break/error)
            self.feedback.clear_output()
            with self.feedback:
                display(self.done)
            time.sleep(5)  # wait long enough to be seen, then disappear so next press also gets feedback
            self.feedback.clear_output()


class ManualCurationController:
    def __init__(self):
        self._viz_container: Optional[CNMFVizContainer] = None

        self.accept_list: set[int] = set()
        self.reject_list: set[int] = set()

        self._add_accept = Button(description='Add')
        self._add_accept.on_click(lambda _: self._add_cell(True))
        self._remove_accept = Button(description='Remove')
        self._remove_accept.on_click(lambda _: self._remove_cell(True))
        self._add_reject = Button(description='Add')
        self._add_reject.on_click(lambda _: self._add_cell(False))
        self._remove_reject = Button(description='Remove')
        self._remove_reject.on_click(lambda _: self._remove_cell(False))

        self._accept_buttons = HBox([self._add_accept, self._remove_accept])
        self._accept_selector = SelectMultiple()
        self._accept_pane = VBox([Label('Accepted cells:'), self._accept_buttons, self._accept_selector])

        self._reject_buttons = HBox([self._add_reject, self._remove_reject])
        self._reject_selector = SelectMultiple()
        self._reject_pane = VBox([Label('Rejected cells:'), self._reject_buttons, self._reject_selector])

        self._save_button = ButtonWithFeedback('Save to disk')
        # add callback initially to say that save_eval callback is not set yet
        def dummy_cb(_):
            raise RuntimeError('Save callback has not been set yet')
        self._save_button.on_click(dummy_cb)

        self.widget = VBox([HBox([self._accept_pane, self._reject_pane]), self._save_button])

        # for callbacks
        self._handlers: list[Callable[[], None]] = [self.update_controls]
        self._block_handlers = True  # wait until set_data is first called to enable

    def set_viz(self, viz: CNMFVizContainer):
        self._viz_container = viz
        # use the same callbacks as the eval save button for the manual curation save button
        self._save_button.click_handlers = viz._eval_controller.button_save_eval._click_handlers

    def set_data_from_cnmf(self, cnmf_obj: CNMFExt):
        """Grap accepted/rejected lists from CNMF object (e.g. after selecting a new run)"""
        self._block_handlers = True
        est = cnmf_obj.estimates
        if est.idx_components_eval is None or est.idx_components_bad_eval is None:
            raise RuntimeError('Eval must have been run')

        self.accept_list = set(est.accepted_list)
        self.reject_list = set(est.rejected_list)
        
        self.update_controls()        
        self._block_handlers = False
    
    def update_controls(self):
        for (cell_list, selector) in zip([self.accept_list, self.reject_list],
                                         [self._accept_selector, self._reject_selector]):
            # unselect any that are no longer in the list
            curr_selected = {int(cell) for cell in selector.value}
            selector.options = [str(cell) for cell in sorted(cell_list)]
            selector.value = [str(cell) for cell in curr_selected & cell_list]

    def apply_to_cnmf(self, cnmf_obj: CNMFExt):
        """Update idx_components and idx_components_bad in CNMF object"""
        est = cnmf_obj.estimates
        est.accepted_list = np.array(list(self.accept_list), dtype=int)
        est.rejected_list = np.array(list(self.reject_list), dtype=int)

    def get_data(self) -> dict[str, set[int]]:
        return {'accept': self.accept_list, 'reject': self.reject_list}
    
    def add_handler(self, func: Callable[[], None]):
        """Add a callback function for accept and reject lists changing"""
        self._handlers.append(func)
    
    def remove_handler(self, func: Callable[[], None]):
        self._handlers.remove(func)
    
    def clear_handlers(self):
        self._handlers = [self.update_controls]

    def _add_cell(self, b_accept: bool):
        """Add button callback"""
        if self._viz_container is None:
            raise RuntimeError('Cannot add cell until fully initialized')

        cell_list = self.accept_list if b_accept else self.reject_list
        other_list = self.reject_list if b_accept else self.accept_list
        curr_cell = self._viz_container.component_index
        if curr_cell in cell_list:
            # don't need to do anything
            return
        
        if curr_cell in other_list:
            # cell can only be in one list at a time
            self._remove_cell(not b_accept, {curr_cell})

        cell_list.add(curr_cell)
        self._call_handlers()

    def _remove_cell(self, b_accept: bool, which_cells: Optional[set[int]] = None):
        """Remove button callback"""
        cell_list = self.accept_list if b_accept else self.reject_list
        if which_cells is None:
            selector = self._accept_selector if b_accept else self._reject_selector
            which_cells = {int(cell) for cell in selector.value}
        cell_list -= which_cells
        self._call_handlers()
    
    def _call_handlers(self):
        if self._block_handlers:
            return

        for handler in self._handlers:
            handler()


class MetricStream(Stream):
    cell_id = param.Integer(default=0, doc='Current cell/component')
    curr_row = param.Integer(default=0, doc='Current CNMF run')
    use_cnn = param.Boolean(default=True, doc='Whether to use the CNN metric')
    min_SNR = param.Number(default=0, doc='Accept threshold for SNR')
    SNR_lowest = param.Number(default=0, doc='Minimum threshold for SNR')
    rval_thr = param.Number(default=0, doc='Accept threshold for spatial correlation')
    rval_lowest = param.Number(default=0, doc='Minimum threshold for spatial correlation')
    min_cnn_thr = param.Number(default=0, doc='Accept threshold for CNN')
    cnn_lowest = param.Number(default=0, doc='Minimum threshold for CNN')


class MetricHistogram:
    """Plots a histogram showing the values of one of the evaluation metrics that updates to show the value at the current cell"""

    def __init__(self, name: str, container: 'CNMFVizWideContainer', n_bins: int = 40):
        self.name = name
        self.container = container
        self.n_bins = n_bins
        self.histcounts: dict[int, tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    def get_histogram(self, stream: MetricStream) -> hv.DynamicMap:
        return hv.DynamicMap(self._build_histogram, streams=[stream]).opts(hv.opts.Histogram(framewise=True)) # type: ignore

    def _build_histogram(self, curr_row: int, cell_id: int, use_cnn: bool, **float_metrics: float) -> hv.Overlay:
        """Make histogram. Takes parameters from a MetricStream."""
        is_cnn = self.name == 'cnn_score'
        metric_info: MetricInfo = getattr(self.container._cnmf_obj_ext.estimates, self.name)
        if curr_row not in self.histcounts or (is_cnn and use_cnn and self.histcounts[curr_row][0] is None):
            # add row to histcounts
            if metric_info.vals is None or len(metric_info.vals) == 0:
                self.histcounts[curr_row] = (None, None)
            else:
                self.histcounts[curr_row] = np.histogram(metric_info.vals[np.isfinite(metric_info.vals)], self.n_bins)
            
        # gather information
        min_thresh = float_metrics[metric_info.min_thresh_name]
        accept_thresh = float_metrics[metric_info.accept_thresh_name]
        enabled = not is_cnn or use_cnn
        title = metric_info.name
        freqs, edges = self.histcounts[curr_row]
        if edges is None or freqs is None:
            edges = np.array([min_thresh - 1, accept_thresh + 1])
            freqs = np.array([0])

        # make the histogram
        if enabled:
            fill_color = hv.dim(title).bin(
                bins=[edges[0], min_thresh, accept_thresh, edges[-1]],
                labels=['red', 'yellow', 'green'])
            assert metric_info.vals is not None
            cell_val = metric_info.vals[cell_id]
        else:
            fill_color = 'gray'
            cell_val = 0

        histogram = hv.Histogram((edges, freqs), kdims=[title], vdims=['# of ROIs']).opts(
            title=title, fill_color=fill_color)
        min_line = hv.VLine(min_thresh).opts(color='red' if enabled else 'gray', line_dash='dashed')
        max_line = hv.VLine(accept_thresh).opts(color='green' if enabled else 'gray', line_dash='dashed')
        cell_line = hv.VLine(np.clip(cell_val, edges[0], edges[-1])).opts(color='white', line_width=4)
        return (histogram * min_line * max_line * cell_line).opts(  # type: ignore
                width=300, xlim=(edges[0], edges[-1]), ylim=(0, np.max(freqs) * 1.1 + 1), active_tools=[])


def get_coms_of_disconnected_coords(coords: np.ndarray, dist_thresh=0.) -> list[np.ndarray]:
    """
    Get approximate centers of mass from disconnected components of contour coordinates.
    If dist_thresh > 0, only includes a COM if it is at least that distance (in pixels) from
    all other COMs.
    """
    nan_entries = np.flatnonzero(np.any(np.isnan(coords), axis=1))
    connected_comps = np.split(coords, nan_entries, axis=0)
    coms: list[np.ndarray] = []
    
    for comp in connected_comps:
        com = np.nanmean(comp, axis=0)
        if dist_thresh > 0:
            dists = [np.linalg.norm(com - other) for other in coms]
            if any(d < dist_thresh for d in dists):
                continue
        coms.append(com)
    return coms


def my_set_limits(self: EvalController, cnmf_obj: CNMF):
    """Method to patch EvalController.set_limits, in this case to avoid infinite limits"""
    self._block_handlers = True
    for metric in self._float_metrics:
        metric_array = getattr(cnmf_obj.estimates, self._metric_array_mapping[metric])
        metric_array = np.asarray(metric_array)[np.isfinite(metric_array)]
        if len(metric_array) == 0:  # for example, cnn_preds is an empty array for CNMFE
            for ui_element in ["slider", "spinbox"]:
                self._widgets[metric][ui_element].disabled = True
        else:
            for ui_element in ["slider", "spinbox"]:
                self._widgets[metric][ui_element].disabled = False
                # allow 100 steps
                self._widgets[metric][ui_element].step = np.ptp(metric_array) / 100
                self._widgets[metric][ui_element].min = metric_array.min()
                self._widgets[metric][ui_element].max = metric_array.max()
                self._widgets[metric][ui_element].value = cnmf_obj.params.get_group("quality")[metric]

    if len(cnmf_obj.estimates.cnn_preds) == 0:  # type: ignore
        # cnn_preds is not present, force use_cnn = False, disable the UI
        self.use_cnn_checkbox.value = False
        self.use_cnn_checkbox.disabled = True
    else:
        # cnn_preds is a real array, allow it to be settable
        self.use_cnn_checkbox.disabled = False
        self.use_cnn_checkbox.value = cnmf_obj.params.get_group("quality")["use_cnn"]

    self._block_handlers = False


class CNMFVizWideContainer(CNMFVizContainer):
    """
    Drop-in replacement for CNMFVizContainer that works better for wide images
    structural_tif_path: pass a path to structural movie tif to enable the following:
    - Marked/unmarked contour visibility and color modes
    - strucural_input and structural_mean image data options (background images)
    """
    IMAGE_OPTIONS = CNMFVizContainer.IMAGE_OPTIONS + [
        'structural_input',
        'structural_mean',
        'corr',
        'mean_equalized',
        'max_equalized'
        ]
    
    MetricName = Literal['snr', 'spatial_corr', 'cnn_score']

    def __init__(
        self,
        dataframe: pd.DataFrame,
        start_index: Optional[int] = None,
        temporal_data_option: Optional[str] = None,
        image_data_options: Optional[list[str]] = None,
        temporal_kwargs: Optional[dict] = None,
        reset_timepoint_on_change: bool = False,
        input_movie_kwargs: Optional[dict] = None,
        image_widget_kwargs: Optional[dict] = None,
        data_grid_kwargs: Optional[dict] = None,
        structural_tif_path: Optional[str] = None,
        n_planes: int = 1,
        hist_nbins: int = 40
        ):
        # enforce that the image grid is N x 1
        # assumes that 4 images are plotted by default (true in 0.1.0)
        n_images = 4 if image_data_options is None else len(image_data_options)
        self.n_planes = n_planes
        
        if image_widget_kwargs is None:
            image_widget_kwargs = dict()

        if 'grid_shape' not in image_widget_kwargs:  # still allow overriding
            image_widget_kwargs['grid_shape'] = (n_images, 1)
        
        if 'grid_plot_kwargs' not in image_widget_kwargs:
            image_widget_kwargs['grid_plot_kwargs'] = dict()
            
        if 'size' not in image_widget_kwargs['grid_plot_kwargs']:
            image_widget_kwargs['grid_plot_kwargs']['size'] = (300 * n_planes, 450 * n_images)

        self.structural_tif_path = structural_tif_path

        self._manual_curation_controller = ManualCurationController()  # used in super().__init__ call

        # cache locations of contours in each subplot
        self._contour_coms = np.empty((0, 2))
        self._contour_rois = np.empty(0, dtype=int)

        super().__init__(
            dataframe, start_index=start_index, temporal_data_option=temporal_data_option,  # type: ignore
            image_data_options=image_data_options, temporal_kwargs=temporal_kwargs,  # type: ignore
            reset_timepoint_on_change=reset_timepoint_on_change, input_movie_kwargs=input_movie_kwargs,  # type: ignore
            image_widget_kwargs=image_widget_kwargs, data_grid_kwargs=data_grid_kwargs)  # type: ignore
        
        assert isinstance(self._cnmf_obj, CNMFExt), 'CNMF obj should be wrapped in CNMFExt here'

        self._manual_curation_controller.set_viz(self)

        # add callback to clear my CNMF cache
        self.on_save(lambda _: clear_cnmf_cache())

        # update eval controller to handle non-finite values correctly
        self._eval_controller.set_limits = partial(my_set_limits, self._eval_controller)

        # add manual accept/reject tab
        self._manual_curation_controller.set_data_from_cnmf(self._cnmf_obj)
        self._manual_curation_controller.add_handler(self._set_manual)

        self._tab_contours_eval.children += (self._manual_curation_controller.widget,)
        self._tab_contours_eval.titles = self._tab_contours_eval.titles[:2] + ('manual curation',)

        # reorganize top widget
        self._middle_widget = VBox(self._top_widget.children[1:])
        self._top_widget = self._top_widget.children[0]

        # histograms
        self._metric_stream = MetricStream()
        self._eval_controller.add_handler(lambda data: self._metric_stream.event(**data))
        self._eval_controller._call_handlers(None)  # make sure stream is up to date before making histograms
        self._metric_histogram_objs, self._metric_histograms = self.metric_histograms(n_bins=hist_nbins)
        self._dff_plot = self.make_dff_plot()

        # trigger callback on CNN checkbox - I think this was not done originally by mistake
        self._eval_controller.use_cnn_checkbox.observe(self._eval_controller._call_handlers, 'value')

        # render plots in a way that allows me to set the theme
        self._plots_doc = Document(theme=built_in_themes['dark_minimal'])
        self._eval_plots = column([hv.render(self._dff_plot), hv.render(self._metric_histograms)])
        self._plots_doc.add_root(self._eval_plots)

        
    @property
    def _cnmf_obj_ext(self) -> CNMFExt:
        # just to help the type checker, identical to _cnmf_obj 
        assert isinstance(self._cnmf_obj, CNMFExt), 'should always be wrapped in CNMFExt when loading'
        return self._cnmf_obj

    def show(self, sidecar: bool = False):
        if self.image_widget.gridplot.canvas.__class__.__name__ == "JupyterWgpuCanvas":
            # change default sizes so it lays out nicely
            self._plot_heatmap._starting_size = (400, 300)
            self._plot_temporal._starting_size = (800, 300)

            # put heatmap and temporal plots in a row at the top
            temporals = HBox([self._plot_heatmap.show(), self._plot_temporal.show()])
            self._widget = VBox([self._top_widget, temporals,
                                 HBox([self._image_widget.widget, jbk.BokehModel(self._eval_plots)]),
                                 self._middle_widget, self._tab_contours_eval])
            if sidecar:
                self._sidecar = Sidecar()
                with self._sidecar:
                    return display(self._widget)
            else:
                return self._widget
        elif self.image_widget.gridplot.canvas.__class__.__name__ == "QWgpuCanvas":
            self.plot_temporal.show()
            self.plot_heatmap.show()
            self.image_widget.show()

            self._widget = VBox([self._top_widget, self._middle_widget, self._tab_contours_eval])
            return self._widget
        else:
            raise EnvironmentError(
                "No available output context. Make sure you're running in jupyterlab or using %gui qt"
            )
            
    def _set_eval(self, eval_params: dict):
        index = self._get_selected_row()
        assert index is not None, 'Invalid/no selected row'

        cnmf_obj = self._cnmf_obj_ext

        cnmf_obj.estimates.filter_components(
            imgs=self._dataframe.iloc[index].caiman.get_input_movie(),
            params=cnmf_obj.params, new_dict=eval_params
        )

        # take accept/reject lists into account
        self._manual_curation_controller.apply_to_cnmf(cnmf_obj)

        # set the colors
        colors = self._dropdown_contour_colors.value
        assert isinstance(colors, str), 'Dropdown value should be a string'
        self.set_component_colors(colors)

        # update sliders in case CNN should be enabled/disabled
        self._eval_controller.set_limits(cnmf_obj)
        self._eval_controller.use_cnn_checkbox.disabled = False

    def on_save(self, callback: Callable[[Any], None], remove=False):
        """Register callback for eval and manual curation save buttons"""
        self._eval_controller.button_save_eval.on_click(callback, remove=remove)

    def _row_changed(self, *args):
        super()._row_changed(*args)
        self._metric_stream.event(curr_row=self.current_row)

    def _set_data(self, data_arrays: dict[str, np.ndarray]):
        super()._set_data(data_arrays)

        # cache locations of coms
        if len(self._image_data_options) > 0:
            first_plot = self._image_widget.gridplot[self._image_data_options[0]]
            self._cache_contour_coms(first_plot['contours'])

        # this may disable the use_cnn checkbox, but it should not
        self._eval_controller.use_cnn_checkbox.disabled = False

        self._manual_curation_controller.set_data_from_cnmf(self._cnmf_obj_ext)
        # decide whether to show "marked" option
        has_marked = self._cnmf_obj_ext.estimates.idx_components_marked is not None
        curr_options = self._dropdown_contour_colors.options

        if has_marked and 'marked' not in curr_options:
            self._dropdown_contour_colors.options = curr_options + ('marked',)
        elif not has_marked and 'marked' in curr_options:
            marked_ind = curr_options.index('marked')
            self._dropdown_contour_colors.options = curr_options[:marked_ind] + curr_options[marked_ind+1:]

        # deal with negative snr_comps
        curr_options = self._dropdown_contour_colors.options
        can_use_log = self._cnmf_obj_ext.estimates.SNR_comp is not None and np.min(self._cnmf_obj_ext.estimates.SNR_comp) > 0
        if can_use_log and 'snr_comps_log' not in curr_options:
            self._dropdown_contour_colors.options = curr_options + ('snr_comps_log',)
        elif not can_use_log and 'snr_comps_log' in curr_options:
            log_ind = curr_options.index('snr_comps_log')
            self._dropdown_contour_colors.options = curr_options[:log_ind] + curr_options[log_ind+1:]

    def _cache_contour_coms(self, line_collection):
        """
        save centers of mass of contours to use in _euclidean later
        override to treat disconnected components of contour separately
        """
        coms: list[np.ndarray] = []
        rois: list[int] = []  # component for each com; can be multiple coms for a single component

        for roi, contour in enumerate(line_collection.graphics):
            coords = contour.data()
            this_coms = get_coms_of_disconnected_coords(coords)
            coms.extend(this_coms)
            rois.extend([roi] * len(this_coms))
        
        self._contour_coms = np.stack(coms)
        self._contour_rois = np.array(rois)
    
    def _euclidean(self, source, target, event, new_data):
        """maps click events to contour"""
        # euclidean distance to find closest index of com
        indices = np.array(event.pick_info["index"])
        indices = np.append(indices, [0])

        ix = int(np.linalg.norm((self._contour_coms - indices), axis=1).argsort()[0])
        self.set_component_index(self._contour_rois[ix])
    
    def _set_manual(self):
        self._manual_curation_controller.apply_to_cnmf(self._cnmf_obj_ext)
        
        # set the colors
        colors = self._dropdown_contour_colors.value
        assert isinstance(colors, str), 'Dropdown value should be a string'
        self.set_component_colors(colors)

    def _manual_toggle_component(self, ev):
        """Override a/r keyboard shortcut to use my manual curation controller"""
        if not hasattr(ev, "key"):
            return

        if ev.key == "a":
            self._manual_curation_controller._add_cell(b_accept=True)
        elif ev.key == "r":
            self._manual_curation_controller._add_cell(b_accept=False)


    def get_cnmf_data_mapping(self, series: pd.Series, input_movie_kwargs: dict, temporal_kwargs: dict
                              ) -> dict[str, Callable[[], np.ndarray]]:
        mapping = CNMFVizContainer.get_cnmf_data_mapping(series, input_movie_kwargs, temporal_kwargs)
        
        # use wrapper class for CNMF
        get_base_cnmf_fn = mapping['cnmf_obj']
        mapping['cnmf_obj'] = lambda: CNMFExt(copy_from=get_base_cnmf_fn())

        structural_tif_path = self.structural_tif_path
        if structural_tif_path is not None:
            def get_structural_input():
                Y = caiman.load(structural_tif_path)
                if Y.ndim > 3:
                    # flatten planes along X
                    Y = np.reshape(Y, Y.shape[:2] + (np.prod(Y.shape[2:]),), order='F')
                return Y

            mapping['structural_input'] = get_structural_input
            mapping['structural_mean'] = lambda: mapping['structural_input']().mean(axis=0)

        # add correlation image, not sure why not included to begin with
        def nan2zero(x: np.ndarray):
            return np.where(np.isnan(x), 0, x)
        mapping['corr'] = lambda: nan2zero(series.caiman.get_corr_image())

        # add brightness-equalized images
        border = series.params['main']['patch']['border_pix']
        borders = [BorderSpec.equal(border)] * self.n_planes

        def get_equalized_projection(proj_type: str):
            proj = mapping[proj_type]()
            return preprocess_proj_for_seed(proj, borders=borders)

        for proj_type in ['mean', 'max']:
            mapping[proj_type + '_equalized'] = partial(get_equalized_projection, proj_type)

        return mapping

    
    def set_component_colors(self, metric: Union[str, np.ndarray], cmap: Optional[str] = None):
        def remove_nonfinite(classifier) -> np.ndarray:
            cls_array = np.array(classifier)
            arr_finite = cls_array[np.isfinite(cls_array)]
            cls_array[np.isnan(cls_array)] = min(arr_finite)
            cls_array[np.isposinf(cls_array)] = max(arr_finite)
            cls_array[np.isneginf(cls_array)] = min(arr_finite)
            return cls_array

        if metric == 'marked':
            assert self._cnmf_obj_ext.estimates.idx_components_marked is not None, \
                'Marked components should be identifed if "marked" is available'
            marked_inds = self._cnmf_obj_ext.estimates.idx_components_marked
            for subplot in self._image_widget.gridplot:
                contour_plot = cast(fpl.LineCollection, subplot['contours'])
                contour_plot[:].colors = 'g'
                contour_plot[marked_inds].colors = 'r'
                self._set_component_visibility(contour_plot, self._cnmf_obj_ext)
        
        # ensure continuous classifiers are finite
        elif metric == 'snr_comps':
            classifier = self._cnmf_obj.estimates.SNR_comp
            super().set_component_colors(remove_nonfinite(classifier),
                                         cmap)  # type: ignore
        elif metric == 'snr_comps_log':
            classifier = self._cnmf_obj.estimates.SNR_comp
            classifier = remove_nonfinite(classifier)
            classifier[classifier <= 0] = min(classifier[classifier > 0])
            super().set_component_colors(np.log10(classifier),
                                         cmap)  # type: ignore
        elif metric == 'r_values':
            classifier = self._cnmf_obj.estimates.r_values
            super().set_component_colors(remove_nonfinite(classifier),
                                         cmap)  # type: ignore
        elif metric == 'cnn_preds':
            classifier = self._cnmf_obj.estimates.cnn_preds
            super().set_component_colors(remove_nonfinite(classifier),
                                         cmap)  # type: ignore
        else:
            super().set_component_colors(metric, cmap)  # type: ignore


    def metric_histograms(self, n_bins: int = 40) -> tuple[list[MetricHistogram], hv.Layout]:
        """make a layout of DynamicMaps for histograms of metrics parameterized by the current cell and thresholds"""
        hist_objs = []
        hists = []
        for metric_name in self.MetricName.__args__:
            hist_obj = MetricHistogram(metric_name, self, n_bins=n_bins)
            hist_objs.append(hist_obj)
            hists.append(hist_obj.get_histogram(self._metric_stream))
        layout: hv.Layout = hv.Layout(hists).opts(shared_axes=False)  # type: ignore
        return hist_objs, layout


    def make_dff_plot(self) -> hv.DynamicMap:
        def dff_plot(cell_id: int, **_other_params) -> hv.Curve:
            est = self._cnmf_obj_ext.estimates
            if est.F_dff is not None:
                dff = est.F_dff[cell_id]
            else:
                # calc dF/F on the fly for just the current component
                dff = cma.calc_df_over_f(est, use_residuals=True, roi_subset=[cell_id])[0]
            assert isinstance(dff, np.ndarray)
            return hv.Curve((range(len(dff)), dff), kdims='frame', vdims="\u0394F/F").opts(width=900)  # type: ignore
        return hv.DynamicMap(dff_plot, streams=[self._metric_stream]
                             ).opts(framewise=True) # type: ignore

    def set_component_index(self, index):
        super().set_component_index(index)
        if hasattr(index, "pick_info"):
            # came from heatmap component selector
            if index.pick_info["pygfx_event"] is None:
                # this means that the selector was not triggered by the user but that it moved due to another event
                # so then we don't set_component_index because then infinite recursion
                return
            index = index.pick_info["selected_index"]
        self._metric_stream.event(cell_id=index)


@pd.api.extensions.register_dataframe_accessor("cnmf")
class CNMFDataFrameVizWideExtension(CNMFDataFrameVizExtension):
    """Adds viz_wide to the cnmf DataFrame accessor"""
    def viz_wide(self, *args, **kwargs):
        container = CNMFVizWideContainer(
            dataframe=self._dataframe,
            *args, **kwargs
        )
        return container


class RawDataPreviewContainer:
    """Widget that shows a mean projection of n frames of the raw data and allows bidirectional correction"""

    def __init__(self, sbx_files: list[str], frames: Union[int, slice], *,
                 subinds_spatial: Sequence[sbx_utils.DimSubindices]=(), curr_offset: Optional[int],
                 offset_save_callback: Optional[Callable[[int], None]] = None, channel: Optional[int] = 0,
                 title: Optional[str] = None):
        # check whether bidirectional
        is_bidi = [sbx_utils.sbx_meta_data(f)['scanning_mode'] == 'bidirectional' for f in sbx_files]
        if all(is_bidi):
            self.bidi = True
        elif not any(is_bidi):
            self.bidi = False
        else:
            raise RuntimeError('Cannot preview combination of uni- and birectional files')
        
        self.mean_data_3d = average_raw_frames(sbx_files, frames, channel=channel, crop_dead=self.bidi,
                                               subinds_spatial=subinds_spatial, dview=cma.cluster.dview)
        self.offset_data_3d = np.copy(self.mean_data_3d, order='F')  # pre-allocate
        self.curr_offset = 0 if curr_offset is None else curr_offset

        # initialize ImageWidget
        self.image_widget = fpl.ImageWidget(data=[self.get_corrected_image()],
                                            names=[title] if title else None,  # type: ignore
                                            cmap='viridis', grid_plot_kwargs={'size': (1500, 400)})
        self.use_offset = self.curr_offset != 0 or self.bidi  # whether to show offset interface
        self.widget = None
        if offset_save_callback is not None:
            def save_cb(_obj):
                offset_save_callback(self.curr_offset)
            self.save_button = ButtonWithFeedback('Save offset')
            self.save_button.on_click(save_cb)
        else:
            self.save_button = None

    def show(self):
        if not self.use_offset:
            self.widget = self.image_widget.show()
        else:
            # set up interactive offset adjustment
            self.offset_label = Label('Odd row offset: ')
            self.offset_slider = IntSlider(min=-25, max=25, value=self.curr_offset, step=1)
            self.offset_slider.observe(
                lambda change: self.set_offset(change['new']), 'value'
            )

            rows = [self.image_widget.show(), HBox([self.offset_label, self.offset_slider])]
            if self.save_button is not None:
                rows.append(self.save_button)
            self.widget = VBox(rows)
        return self.widget

    def get_corrected_image(self) -> np.ndarray:
        """
        Get image after correcting for current offset.
        Note that offset is the # that the odd rows *are* shifted
        to the right, so to do the correction, we need to shift by the opposite amount.
        Quick and dirty so just change the odd rows.
        """
        correction = -self.curr_offset
        if correction > 0:
            self.offset_data_3d[1::2, correction:] = self.mean_data_3d[1::2, :-correction]
            self.offset_data_3d[1::2, :correction] = 0
        elif correction < 0:
            self.offset_data_3d[1::2, :correction] = self.mean_data_3d[1::2, -correction:]
            self.offset_data_3d[1::2, correction:] = 0
        else:
            self.offset_data_3d[1::2, :] = self.mean_data_3d[1::2, :]
        return np.reshape(self.offset_data_3d, (self.offset_data_3d.shape[0], -1), order='F')

    def set_offset(self, new_offset: int):
        """
        Change image to reflect new offset correction.
        """
        if new_offset == self.curr_offset:
            return
        
        self.curr_offset = new_offset
        self.image_widget.set_data(self.get_corrected_image(), reset_vmin_vmax=False, reset_indices=False)


def mcorr_compare_widget(movie_orig: np.ndarray, movie_mcorr: np.ndarray):
    """Make an ImageWidget that shows the original movie above the motion-corrected one"""
    def flatten(arr: np.ndarray):
        if arr.ndim > 3:
            new_shape = (*movie_orig.shape[:2], movie_orig.shape[2] * movie_orig.shape[3])
            return arr.reshape(new_shape, order='F')
        else:
            return arr

    mcorr_iw = fpl.ImageWidget(
        data=[flatten(arr) for arr in [movie_orig, movie_mcorr]],
        names=['Input', 'Motion corrected'],
        dims_order='tyx',
        grid_shape=(2, 1),
        cmap='gray'
    )
    return mcorr_iw.show()


class patch_shift_mag_and_angle(Operation):
    vdim = hv.param.String(default='shift', doc='vdim containing shifts to operate on')

    def _process(self, element: hv.Dataset, key=None) -> hv.Dataset:
        """Make a new dataset containing magnitude and angles of shifts"""
        kdims_to_keep = [d for d in element.dimensions('key') if d != 'dim']
        shifts_x = cast(hv.Dataset, element.select(dim='x')).reindex(kdims=kdims_to_keep)
        shifts_y = cast(hv.Dataset, element.select(dim='y')).reindex(kdims=kdims_to_keep)
        mag = np.sqrt(shifts_x.data[self.p.vdim] ** 2 + shifts_y.data[self.p.vdim] ** 2)
        angle = np.arctan2(shifts_y.data[self.p.vdim], shifts_x.data[self.p.vdim])
        return shifts_x.add_dimension(
            'shift_mag', 0, mag, vdim=True).add_dimension(
            'shift_angle', 0, angle, vdim=True).clone(
            vdims=['shift_mag', 'shift_angle']
            )


def check_mcorr_nb(movie_orig: np.ndarray, movie_mcorr: np.ndarray, mc_result: MCResult,
                   show_quiver_plot=False):
    # make upper figure with histograms of mcorr shifts
    shifts_rig_ds = mc_result.shifts_rig_hv
    shifts_els_ds = mc_result.shifts_els_hv
    shift_holomap = shifts_rig_ds.to(hv.Curve, 'frame', 'shift').opts(alpha=0.7)
    
    if shifts_els_ds is not None:
        # add min and max piecewise shifts to rigid shift dataset
        min_shifts = cast(hv.Dataset, shifts_els_ds.aggregate(['frame', 'dim', 'plane'], function=np.min))
        max_shifts = cast(hv.Dataset, shifts_els_ds.aggregate(['frame', 'dim', 'plane'], function=np.max))
        shifts_w_range = shifts_rig_ds.add_dimension(
            'minshift', 0, min_shifts.data['shift'], vdim=True).add_dimension(
            'maxshift', 0, max_shifts.data['shift'], vdim=True)
        shifts_w_range = shifts_w_range.transform(
            upper=hv.dim('maxshift') - hv.dim('shift'),
            lower=hv.dim('shift') - hv.dim('minshift')
        )
        range_holomap = shifts_w_range.to(hv.Spread, kdims='frame', vdims=['shift', 'lower', 'upper']).opts(
            fill_alpha=0.45, line_alpha=0
        )
        shift_holomap = shift_holomap * range_holomap

        # make quiver plots of the shifts for each patch
        if show_quiver_plot:
            shifts_els_mag_and_angle = cast(hv.Dataset, patch_shift_mag_and_angle(shifts_els_ds))  # type: ignore
            def quiver(plane: int, frame: int):
                this_mag_and_angle = cast(hv.Dataset, shifts_els_mag_and_angle.select(plane=plane, frame=frame))
                return this_mag_and_angle.reindex().to(
                    hv.VectorField, kdims=['xpatch', 'ypatch'], vdims=['shift_angle', 'shift_mag'],
                    group='Piecewise shifts').opts(invert_yaxis=True, data_aspect=1, responsive=True)

            kdims = ['plane', 'frame']
            quiver_dmap = hv.DynamicMap(quiver, kdims=kdims).redim.values(
                **{kdim: shifts_els_mag_and_angle.data[kdim] for kdim in kdims}
            )
            quiver_plots = quiver_dmap.layout('plane').opts(sizing_mode='stretch_width')  # type: ignore
        else:
            quiver_plots = None
    else:
        quiver_plots = None
    
    shift_holomap.get_dimension('dim').label = ''  # remove title from legend
    shift_plots = shift_holomap.opts(height=300, responsive=True).overlay('dim').layout('plane').opts(
        opts.NdOverlay(legend_cols=2),
        opts.NdLayout(sizing_mode='stretch_width'))

    plot_rows: list = [pn.ipywidget(pn.Row(pn.pane.HoloViews(shift_plots), width_policy='max', max_width=2500))]
    if quiver_plots is not None:
        plot_rows.append(pn.ipywidget(pn.Row(pn.pane.HoloViews(quiver_plots, widget_location='bottom'),
                                             width_policy='max', max_width=2500)))
    plot_rows.append(mcorr_compare_widget(movie_orig, movie_mcorr))
    widget = VBox(plot_rows)
    return widget


def make_rgb_frame_apply(rgb_image: np.ndarray) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Hack to make RGB images work with ImageWidget for fastplotlib <v0.2
    Input should be an image where the last dimension is channels (i.e. of length 3 or 4)
    Outputs are (image_flat, rgb_frame_apply). image_flat should be passed as the image to ImageWidget.
    frame_apply should be used as the frame_apply function for this subplot.
    For example, if the first subplot was grayscale and the second was RGB, you would use
    frame_apply={0: None, 1: rgb_frame_apply}.
    """
    n_chan = rgb_image.shape[-1]
    if n_chan not in (3, 4):
        raise RuntimeError('Number of channels (last dimension) of RGB image must be 3 or 4.')
    
    image_flat = np.reshape(rgb_image, rgb_image.shape[:-2] + (rgb_image.shape[-2] * n_chan,), order='C')
    def rgb_frame_apply(frame: np.ndarray) -> np.ndarray:
        return np.reshape(frame, frame.shape[:-1] + (frame.shape[-1] // n_chan, n_chan), order='C')
    return image_flat, rgb_frame_apply


def my_check_register_ROIs(matched1: list[int],
                           matched2: list[int],
                           unmatched1: list[int],
                           unmatched2: list[int],
                           performance: dict,
                           A1: MaybeSparse,
                           A2: MaybeSparse,
                           A2_orig: MaybeSparse,
                           x_remap: Union[Optional[np.ndarray], NAType],
                           y_remap: Union[Optional[np.ndarray], NAType],
                           dims: tuple[int, int],
                           background1: np.ndarray,
                           background2: np.ndarray,
                           clims: Optional[tuple[float, float]] = None,
                           cmap='viridis',
                           contour1_name: str = 'first contours',
                           contour2_name: str = 'second contours',
                           contour1_color = 'g',
                           contour2_color = 'r',
                           use_contour_colors_for_top_plot=False, 
                           merge_colors: Optional[Sequence[str]] = None,
                           backend: Literal['mpl', 'fpl'] = 'fpl',
                           xrange: Optional[Union[slice, tuple[slice, slice]]] = None,
                           highlight1: Sequence[int] = (),  # cell ids to highlight
                           highlight2: Sequence[int] = (),
                           show_single_plot_contours=True,
                           plot_labels=True,
                           sharexy=True):
    """Plot results from register_ROIs using matplotlib or fastplotlib"""
    pct_1 = performance['recall'] * 100
    pct_2 = performance['precision'] * 100
    titles =[
        f'{contour1_name} ({pct_1:.2f}% matched)',
        f'{contour2_name} ({pct_2:.2f}% matched)',
        'Matched',
        'Unmatched'
    ]

    if isinstance(xrange, slice):
        xrange = (xrange, xrange)

    if xrange is not None:
        # crop background images
        background1 = background1[:, xrange[0]]
        background2 = background2[:, xrange[1]]

        if isinstance(y_remap, np.ndarray):
            y_remap = y_remap[:, xrange[1]]  # resize for bg2
        if isinstance(x_remap, np.ndarray):
            x_remap = x_remap[:, xrange[1]] - xrange[1].start  # resize & remove offset for bg2

    if not isinstance(x_remap, NAType) and not isinstance(y_remap, NAType):
        # make merge image for bottom 2 plots
        bg2_mapped = remap_image(background2, x_remap=x_remap, y_remap=y_remap)
        if merge_colors is None:
            merge_colors = [contour1_color, contour2_color]
        merge = make_merge(background1, bg2_mapped, color1=merge_colors[0], color2=merge_colors[1])
        if backend == 'fpl':
            merge_flat, rgb_frame_apply = make_rgb_frame_apply(merge)
            images = [background1, background2, merge_flat, merge_flat]
            frame_apply = {n: None if n < 2 else rgb_frame_apply for n in range(4)}
        else:
            images = [background1, background2, merge, merge]
            frame_apply = None
    else:
        # use background 1 for bottom plots
        images = [background1, background2, background1, background1]
        frame_apply = None

    # get contours from masks
    contours1 = my_get_contours(A1, dims)
    contours2 = my_get_contours(A2, dims)
    contours2_orig = my_get_contours(A2_orig, dims)

    coordinates1 = [c['coordinates'] for c in contours1]
    coordinates2 = [c['coordinates'] for c in contours2]
    coordinates2_orig = [c['coordinates'] for c in contours2_orig]

    if xrange is not None:
        # filter to only use coordinates in the given plane(s)
        def is_in_plane(c, xr: slice):
            xpos = c['CoM'][1]
            return xpos >= xr.start and xpos < xr.stop

        b_keep_1 = [is_in_plane(c, xrange[0]) for c in contours1]
        b_keep_2 = [is_in_plane(c, xrange[1]) for c in contours2_orig]
        matched1_filtered = [m if b_keep_1[m] else None for m in matched1]
        unmatched1 = [m for m in unmatched1 if b_keep_1[m]]
        matched2_filtered = [m if b_keep_2[m] else None for m in matched2]
        unmatched2 = [m for m in unmatched2 if b_keep_2[m]]
        
        def fix_coords(coord_list: list[np.ndarray], xr: slice) -> list[np.ndarray]:
            return [coords if coords.size == 0 else coords - np.array([xr.start, 0]) for coords in coord_list]

        coordinates1 = fix_coords(coordinates1, xrange[0])
        coordinates2 = fix_coords(coordinates2, xrange[1])
        coordinates2_orig = fix_coords(coordinates2_orig, xrange[1])
    else:
        matched1_filtered = matched1
        matched2_filtered = matched2

    if clims is None:
        # infer vmin/vmax not subsampling or taking absolute minimum into account, to deal with zeros
        def infer_clims_robust(data) -> tuple[float, float]:
            # # old method for fpl:
            # vals = np.unique(data)
            # return (
            #     vals[1] if len(vals) > 1 else vals[0],
            #     vals[-1]
            # )
            pcts = np.percentile(data[~np.isnan(data)], [1, 99.96])
            return float(pcts[0]), float(pcts[1])

        clims1 = infer_clims_robust(background1)
        clims2 = infer_clims_robust(background2)
    else:
        clims1 = clims2 = clims

    plot_fn = make_registration_plot_fpl if backend == 'fpl' else make_registration_plot_mpl
    return plot_fn(images=images, coordinates1=coordinates1, coordinates2=coordinates2, coordinates2_orig=coordinates2_orig,
                   matched1=matched1_filtered, matched2=matched2_filtered, unmatched1=unmatched1, unmatched2=unmatched2,
                   titles=titles, frame_apply=frame_apply, cmap=cmap, clims1=clims1, clims2=clims2,
                   contour1_color=contour1_color, contour2_color=contour2_color,
                   use_contour_colors_for_top_plot=use_contour_colors_for_top_plot,
                   highlight1=highlight1, highlight2=highlight2, show_single_plot_contours=show_single_plot_contours,
                   plot_labels=plot_labels, sharexy=sharexy)
    


def make_registration_plot_fpl(images: list[np.ndarray], coordinates1: list[np.ndarray],
                               coordinates2: list[np.ndarray], coordinates2_orig: list[np.ndarray],
                               matched1: Sequence[Optional[int]], matched2: Sequence[Optional[int]],
                               unmatched1: list[int], unmatched2: list[int],
                               titles: list[str], frame_apply: Optional[dict],
                               cmap: str, clims1: tuple[float, float], clims2: tuple[float, float],
                               contour1_color: str, contour2_color: str, use_contour_colors_for_top_plot: bool,
                               highlight1: Sequence[int] = (), highlight2: Sequence[int] = (), show_single_plot_contours=True,
                               plot_labels=True, sharexy=True):
    if not sharexy:
        logging.warning('Ignoring sharexy=False - not implemented for fastplotlib')

    iw = fpl.ImageWidget(data=images, names=titles, grid_shape=(2, 2), cmap=cmap, frame_apply=frame_apply)

    for plotind, ig, clims in zip([(0, 0), (0, 1)], iw.managed_graphics[:2], [clims1, clims2]):
        subplot = iw.gridplot[plotind]
        if 'histogram_lut' not in subplot.docks['right']:
            ig.vmin = clims[0]
            ig.vmax = clims[1]
        else:
            # use histogram and let the handler update image LUT
            # fails if vmin goes past vmax or vice versa, but if 1st call fails 2nd must succeed
            histogram_lut = subplot.docks['right']['histogram_lut']
            histogram_lut.vmin = clims[0]
            histogram_lut.vmax = clims[1]
            histogram_lut.vmin = clims[0]

    def add_contours(plotind: tuple[int, int], coords: list, selection: Sequence[Optional[int]], thickness,
                     color: str = 'w', plot_labels=False):
        sub_selection = [ind for ind in selection if ind is not None and coords[ind].size > 0]
        iw.gridplot[plotind].add_line_collection([coords[ind] for ind in sub_selection],
            colors=color, thickness=list(thickness[sub_selection]))  # type: ignore

        if plot_labels:  # add numeric labels
            for i, ind in enumerate(selection):  # make sure inds correspond to the original selection
                if ind is None or coords[ind].size <= 0:
                    continue

                # make sure to plot labels on every plane where the component appears
                coms = get_coms_of_disconnected_coords(coords[ind], dist_thresh=50)
                for com in coms:
                    iw.gridplot[plotind].add_text(
                        str(i), tuple(com.astype(int) + (5, 5)) + (0,), face_color=color, anchor='bottom-left')

    
    thicknesses1 = np.full(len(coordinates1), 1.)
    thicknesses1[highlight1] = 2.
    thicknesses2 = np.full(len(coordinates2), 1.)
    thicknesses2[highlight2] = 2.

    if show_single_plot_contours:
        add_contours((0, 0), coordinates1, matched1, thicknesses1,
                     contour1_color if use_contour_colors_for_top_plot else 'w', plot_labels=plot_labels)
        add_contours((0, 0), coordinates1, unmatched1, thicknesses1, 'gray')
        add_contours((0, 1), coordinates2_orig, matched2, thicknesses2,
                     contour2_color if use_contour_colors_for_top_plot else 'w', plot_labels=plot_labels)
        add_contours((0, 1), coordinates2_orig, unmatched2, thicknesses2, 'gray')

    add_contours((1, 0), coordinates1, matched1, thicknesses1, contour1_color, plot_labels=plot_labels)
    add_contours((1, 0), coordinates2, matched2, thicknesses2, contour2_color, plot_labels=plot_labels)
    add_contours((1, 1), coordinates1, unmatched1, thicknesses1, contour1_color)
    add_contours((1, 1), coordinates2, unmatched2, thicknesses2, contour2_color)
    
    return iw.show()


def make_registration_plot_mpl(images: list[np.ndarray], coordinates1: list[np.ndarray],
                               coordinates2: list[np.ndarray], coordinates2_orig: list[np.ndarray],
                               matched1: Sequence[Optional[int]], matched2: Sequence[Optional[int]],
                               unmatched1: list[int], unmatched2: list[int],
                               titles: list[str], frame_apply: Optional[dict],
                               cmap: str, clims1: tuple[float, float], clims2: tuple[float, float],
                               contour1_color: str, contour2_color: str, use_contour_colors_for_top_plot: bool,
                               highlight1: Sequence[int] = (), highlight2: Sequence[int] = (), show_single_plot_contours=True,
                               plot_labels=True, sharexy=True):
    lws1 = np.full(len(coordinates1), 1.)
    lws1[highlight1] = 2.
    lws2 = np.full(len(coordinates2), 1.)
    lws2[highlight2] = 2.

    with mplstyle.context('dark_background'):  # type: ignore
        fig, axs = plt.subplots(2, 2, sharex=sharexy, sharey=sharexy)
        # top plots
        for ax, clims, image, coords, matched, unmatched, contour_color, title, lws in zip(
            axs[0], [clims1, clims2], images[:2], [coordinates1, coordinates2_orig], [matched1, matched2],
            [unmatched1, unmatched2], [contour1_color, contour2_color], titles[:2], [lws1, lws2]
        ):
            ax.imshow(image, interpolation=None, cmap=cmap, vmin=clims[0], vmax=clims[1])
            if show_single_plot_contours:
                for i, cellid in enumerate(matched):
                    if cellid is None or coords[cellid].size == 0:
                        continue
                    color = contour_color if use_contour_colors_for_top_plot else 'w'
                    ax.plot(*coords[cellid].T, lw=lws[cellid], c=color)
                    if plot_labels:
                        # make sure to plot labels on every plane where the component appears
                        coms = get_coms_of_disconnected_coords(coords[cellid], dist_thresh=50)
                        for com in coms:
                            ax.text(*(com + (5, 5)), str(i), c=color, clip_on=True, size='small')
                
                for cellid in unmatched:
                    if coords[cellid].size == 0:
                        continue
                    ax.plot(*coords[cellid].T, lw=lws[cellid], c='gray')
            
            ax.set_axis_off()
            ax.set_title(title)
        
        # bottom plots
        for ax, image, sess1_inds, sess2_inds, title, b_matched in zip(
            axs[1], images[2:], [matched1, unmatched1], [matched2, unmatched2], titles[2:], [True, False]
        ):
            ax.imshow(image, interpolation=None)
            for coords, lws, color, inds in zip(
                [coordinates1, coordinates2], [lws1, lws2], [contour1_color, contour2_color], [sess1_inds, sess2_inds]):
                for i, cellid in enumerate(inds):
                    if cellid is None or coords[cellid].size == 0:
                        continue
                    ax.plot(*coords[cellid].T, lw=lws[cellid], c=color)
                    if b_matched and plot_labels:
                        # make sure to plot labels on every plane where the component appears
                        coms = get_coms_of_disconnected_coords(coords[cellid], dist_thresh=50)
                        for com in coms:
                            ax.text(*(com + (5, 5)), str(i), c=color, clip_on=True, size='small')
            
            ax.set_axis_off()
            ax.set_title(title)
    fig.tight_layout()
    return fig


def check_session_alignment(align_results: dict, first_session_to_view: Union[int, str],
                            other_session: Union[int, str, None] = None,
                            background_params: Optional[dict] = None, show_accepted_only: Union[bool, Literal['either', 'both']] = 'either',
                            k_plane: Union[int, slice, tuple[Union[int, slice], Union[int, slice]], None] = None,
                            cells_to_highlight: Sequence[int] = (), plot_labels=True, use_zproj=False,
                            **check_register_ROIs_kwargs):
    """
    Make figure to check ROI matching between two sessions after running register_ROIs_multisession
    or register_ROIs_multisession_3D.

    align_results (dict): 
        output from register_ROIs_multisession. Should have at least these fields:
        - mouse_id
        - grouptag
        - rec_type
        - sessions
        - all_cells
        - matchings
        - xy_remappings OR xy_remap_file
        - cell_subset_name

    first_session_to_view (Union[int, str]):
        The name of the session whose ROIs will be mapped when comparing between sessions.
        This can either be a number or a string. If it's a number, that session number must be unique within all_sessions.
        If a string, it should be f'{sess_id}_{tag}' or just str(sess_id) if there is no tag.

    other_session (Union[int, str, None]):
        The name of the session whose ROIs will not be mapped. By default this is the session immediately after
        first_session_to_view in the list of sessions used for alignment. Can only be another session if
        all pairwise mappings were computed and saved, e.g. in register_ROIs_multisession_3D (as implemented at the moment)

    saved_mapping_filename_fmt (str):
        Format string for filename containing saved mappings, if loading from a file
            
    background_params (Optional[dict]):
        params for making backgrounds (projections) for visualization, default is just mean.

    show_accepted_only (bool):
        whether to filter out rejected cells for display and statistics
        False -> show all cells
        True or "either" -> show cells accepted by either of the sessions
        "both" -> show cells accepted by both sessions (same as "either" for unmatched cells)

    k_plane (Union[int, slice, tuple[int | slice, int | slice], None]):
        If not None, only plot this plane. If a tuple (i, j), plot plane i from first session against plane j from other session.
        Cannot be used with use_zproj. Slices are also supported to select multiple planes.

    cells_to_highlight (Sequence[int]):
        plot contours for these cells with a thicker line
    
    plot_labels (bool):
        Whether to label matched cells with numeric IDs starting at 0 (not the same as union cell IDs)
    
    use_zproj (bool):
        Whether to collapse projections and contours in Z rather than showing each plane side by side.
        This can be helpful if the matching ROIs are not all in the same plane.
        Cannot be used if k_plane is non-None.
    """
    if k_plane is not None and use_zproj:
        raise RuntimeError('Cannot set k_plane when use_zproj is true')

    mouse_id: int = align_results['mouse_id']
    rec_type: str = align_results['rec_type']
    sessions: pd.DataFrame = align_results['sessions']
    matchings: pd.DataFrame = align_results['matchings']
    cell_subset_name: str = align_results['cell_subset_name']
    grouptag: str = align_results['grouptag']

    if background_params is None:
        background_params = {'type': 'mean'}
    
    # find index of first session
    # here (and below) "1" refers to the second session chronologically and vice versa
    # this is because we are aligning the earlier session to the later one, while check_register_ROIs expects the opposite
    def get_session_index(name_or_id: Union[str, int]) -> int:
        if isinstance(name_or_id, int):
            inds = np.flatnonzero(sessions.sess_id == name_or_id)
        else:
            inds = np.flatnonzero(sessions.sess_name == name_or_id)
        if np.size(inds) == 1:
            return inds.item()
        else:
            raise ValueError(f'Requested session {name_or_id} not found or not unique in list of sessions')
    
    ind2 = get_session_index(first_session_to_view)
    row2 = sessions.iloc[ind2]
    sess2_name = str(row2.at['sess_name'])
    sess2_id = int(row2.at['sess_id'])
    sess2_tag = str(row2.at['tag'])

    if other_session is None:
        ind1 = ind2 + 1
        if ind1 >= sessions.shape[0]:
            raise ValueError('Requested session is the last one; cannot compare to subsequent')
    else:
        ind1 = get_session_index(other_session)
    
    row1 = sessions.iloc[ind1]
    sess1_name = str(row1.at['sess_name'])
    sess1_id = int(row1.at['sess_id'])
    sess1_tag = str(row1.at['tag'])
    
    # find mapping
    if 'xy_remappings' in align_results and ind1 == ind2 + 1:
        xy_remappings: list[tuple[np.ndarray, np.ndarray]] = align_results['xy_remappings']
        xy_mapping = xy_remappings[ind2]
    else:
        try:
            xy_mappings, _ = alignment.load_or_compute_remaps_for_sessions(
                mouse_id, sess_ids=[sess2_id, sess1_id], rec_type=rec_type, tags=[sess2_tag, sess1_tag],
                grouptag=grouptag, use_saved_mappings=True)
        except RuntimeError as e:
            if 'xy_remappings' in align_results:
                raise NotImplementedError('Cannot compare non-consecutive results without saved full grid of mappings') from e
            else:
                raise RuntimeError('xy_remappings not saved in results and not found in a file') from e
        else:
            xy_mapping = tuple(xy_mappings[0, 0])            

    # find indices of each session's cells that are matched/unmatched
    matchings1 = matchings.loc[matchings.sess_name == sess1_name, :].set_index('union_cell_id', drop=False)
    matchings2 = matchings.loc[matchings.sess_name == sess2_name, :].set_index('union_cell_id', drop=False)

    # apply show_accepted_only to each matchings table
    subset_names = [cell_subset_name] if cell_subset_name else []
    if show_accepted_only:
        for sess_matchings, (_, sess) in zip([matchings1, matchings2], sessions.loc[[ind1, ind2], :].iterrows()):
            if 'accepted' not in sess_matchings:
                # populate accepted column manually
                alignment.populate_accepted_column(sess_matchings, sess, rec_type=rec_type)

        if show_accepted_only in [True, 'either']:
            # subsample assignments to cells accepted by either session
            either_accepted = matchings1.accepted | matchings2.accepted  # pandas will combine according to union id
            matchings1 = matchings1.loc[either_accepted, :]  # again pandas will do the right thing
            matchings2 = matchings2.loc[either_accepted, :]
            subset_names.append('accepted by either')
        
        elif show_accepted_only == 'both':
            # subsample matching assignments to cells accepted by both sessions
            matchings1 = matchings1.loc[matchings1.accepted, :]
            matchings2 = matchings2.loc[matchings2.accepted, :]
            subset_names.append('accepted by both')
        else:
            raise ValueError(f'Unexpected value {show_accepted_only} for show_accepted_only')

    if len(subset_names) > 0:
        title_addon = f' ({", ".join(subset_names)})'
    else:
        title_addon = ''

    intersection_ids = np.intersect1d(matchings1.union_cell_id, matchings2.union_cell_id)

    b_matched1 = matchings1.union_cell_id.isin(intersection_ids)
    b_matched2 = matchings2.union_cell_id.isin(intersection_ids)

    matched1 = matchings1.loc[intersection_ids, 'session_cell_id'].to_numpy(dtype=int)
    matched2 = matchings2.loc[intersection_ids, 'session_cell_id'].to_numpy(dtype=int)
    unmatched1 = matchings1.loc[~b_matched1, 'session_cell_id'].to_numpy(dtype=int)
    unmatched2 = matchings2.loc[~b_matched2, 'session_cell_id'].to_numpy(dtype=int)
    n_matched = len(matched1)
    n1 = n_matched + len(unmatched1)
    n2 = n_matched + len(unmatched2)

    # make performance for just these 2 sessions
    perf = compute_matching_performance(n1, n2, n_matched)

    # get templates and ROIs
    info1 = cma.load_latest(mouse_id, sessions.at[ind1, 'sess_id'], tag=sessions.at[ind1, 'tag'], rec_type=rec_type)
    plane_height, plane_width = info1.plane_size
    num_planes = info1.metadata['num_planes']
    background1 = info1.get_projection_for_seed(**background_params)[0]
    assert info1.cnmf_fit is not None and info1.cnmf_fit.estimates.A is not None, 'CNMF should be done'
    A1 = info1.cnmf_fit.estimates.A
    if not isinstance(A1, np.ndarray):
        A1 = sparse.csc_array(A1)

    info2 = cma.load_latest(mouse_id, sessions.at[ind2, 'sess_id'], tag=sessions.at[ind2, 'tag'], rec_type=rec_type)
    background2 = info2.get_projection_for_seed(**background_params)[0]
    assert info2.cnmf_fit is not None and info2.cnmf_fit.estimates.A is not None, 'CNMF should be done'
    A2_orig = info2.cnmf_fit.estimates.A
    if not isinstance(A2_orig, np.ndarray):
        A2_orig = sparse.csc_array(A2_orig)

    xy_mapping_orig = xy_mapping
    if np.all(xy_mapping[0].shape == (plane_height, plane_width)):
        # the mapping is just for X and Y; repeat to make a mapping for the whole image
        plane_offsets = np.arange(num_planes, dtype=np.float32) * plane_width
        x_mapping, y_mapping = xy_mapping
        xy_mapping = (
            (plane_offsets + x_mapping[:, :, np.newaxis]).reshape((x_mapping.shape[0], -1), order='F'),
            np.tile(y_mapping, (1, num_planes))
        )
    A2 = map_footprints(A2_orig, xy_mapping)

    # plane handling
    def make_xrange(kplane: Union[int, slice]) -> slice:
        if isinstance(kplane, int):
            return slice(plane_width * kplane, plane_width * (kplane + 1))
        elif isinstance(kplane, slice):
            if kplane.step is not None and kplane.step != 1:
                raise NotImplementedError('k_plane range with gaps not supported')
            start = kplane.start if kplane.start else 0
            stop = kplane.stop if kplane.stop else num_planes
            return slice(plane_width * start, plane_width * stop)
        else:
            raise ValueError('Unsupported type for k_plane')

    if k_plane is not None:
        if not isinstance(k_plane, (tuple, list)):
            k_plane = (k_plane, k_plane)

        # reverse order for same reason as sess1/sess2 above
        xrange = (make_xrange(k_plane[1]), make_xrange(k_plane[0]))
    else:
        xrange = None

    if use_zproj:
        if np.all(xy_mapping_orig[0].shape == (plane_height, plane_width)):
            # if the mapping was originally for the projection, just use that
            xy_mapping = xy_mapping_orig
        else:
            # if the mapping wasn't originally for the z-projection,
            # do mapping on background2 now before taking projection
            background2 = remap_image(background2, *xy_mapping)
            xy_mapping = (None, None)

        # take max-projections of backgrounds and collapse A matrices
        # (binarizing first to undo weird effects of jointly normalizing across planes)
        background1 = np.max(np.split(background1, num_planes, axis=1), axis=0)
        background2 = np.max(np.split(background2, num_planes, axis=1), axis=0)
        A1 = collapse_footprints_to_xy(A1, num_planes, binarize=True)
        A2 = collapse_footprints_to_xy(A2, num_planes, binarize=True)
        A2_orig = collapse_footprints_to_xy(A2_orig, num_planes, binarize=True)

    # convert cells_to_highlight to session indices
    def get_highlight_sess_ids(matchings: pd.DataFrame) -> list[int]:
        b_take = np.isin(matchings.index, cells_to_highlight)
        if sum(b_take) < len(cells_to_highlight):
            logging.warning('Not all highlighted cells found in session')
        return list(matchings.loc[b_take, 'session_cell_id'])

    highlight1 = get_highlight_sess_ids(matchings1)
    highlight2 = get_highlight_sess_ids(matchings2)

    # show comparison figure
    check_register_ROIs_defaults = {
        'cmap': 'viridis',
        'contour1_color': 'g',
        'contour2_color': 'm',
        'use_contour_colors_for_top_plot': True,
        'merge_colors': ['g', 'm'],
    }
    check_register_ROIs_kwargs = {**check_register_ROIs_defaults, **check_register_ROIs_kwargs}

    return my_check_register_ROIs(
        matched1=list(matched1), matched2=list(matched2),
        unmatched1=list(unmatched1), unmatched2=list(unmatched2), performance=perf,
        A1=A1, A2=A2, A2_orig=A2_orig, dims=(background1.shape[0], background1.shape[1]),
        background1=background1, background2=background2,
        contour1_name=f'Session {sess1_name}' + title_addon,
        contour2_name=f'Session {sess2_name}' + title_addon,
        x_remap=xy_mapping[0], y_remap=xy_mapping[1], xrange=xrange,
        highlight1=highlight1, highlight2=highlight2, plot_labels=plot_labels, **check_register_ROIs_kwargs)


def plot_aligned_ROIs(align_results: dict, border_pix=5, map_to: Optional[Union[str, int]] = None,
                      offset_file: Optional[str] = None, contour_thr: Optional[float] = None,
                      contour_method: Literal['max', 'nrg'] = 'nrg', smoothing_sigma: Union[float, Sequence[float]] = 1.,
                      cmap: BokehPalette = bokeh.palettes.inferno, contour_cmap: BokehPalette = bokeh.palettes.turbo,
                      frac_accepted_threshold=0.5) -> pn.pane.Bokeh:
    """
    Make a plot to visualize 3D alignment results for each cell in the unified set.
    The plot has sliders for cell and Z. For each slider combination,
    the plot displays:
        - (Left background): Max projection of all ROIs mapped to one of the sessions (see below)
        - (Left contours): Overlaid mapped contours of each ROI footprint (labeled with session name)
        - (Right): Interpolated probability map of each pixel being part of the cell at the given Z position
            (to visualize what would be used to match a new session to that cell)
    
    Inputs:
        - align_results: Tabularized multisession alignment results
        - border_pix: How many buffer pixels to plot on each side from the max extent of any contour
        - map_to: Session name or number to map all contours to. By default, if we have a
           rigid offsets file, the session that is closest to the center of all used sessions
           will be used. Otherwise, the first session will be used.
        - offset_file: Can be set to override the offset_file saved in the multisession results (or
            provide if this is missing). Can be a format string with one slot for the mouse ID.
        - contour_thr, contour_method: Parameters for computing contours, see cmcustom.my_get_contours.
        - smoothing_sigma: Sigma in pixels for smoothing binary masks to get probability map
        - cmap: colormap for the max projection and probability maps
        - contour_cmap: palette for coloring contours by session
        - frac_accepted_threshold: Only use cells tha twere accepted in at list this fraction of sessions
        - show_immediately: Whether to call "show" on the returned Bokeh Application
    """
    # Unpack align_results into typed variables
    mouse_id: int = align_results['mouse_id']
    rec_type: str = align_results['rec_type']
    sessions: pd.DataFrame = align_results['sessions']
    all_cells: pd.DataFrame = align_results['all_cells']

    sess_ids: list[int] = list(sessions.loc[:, 'sess_id'])
    tags: list[str] = list(sessions.loc[:, 'tag'])
    sess_names: list[str] = list(sessions.loc[:, 'sess_name'])

    # we read offset_file from align_results unless it was provided
    if offset_file is None and not (offset_file := align_results.get('offset_file')):
        raise RuntimeError('Z offsets required to make per-plane session mapping data')

    # Step 1: Determine which session to map to (session name)
    if isinstance(map_to, str):
        if not any(np.equal(sess_names, map_to)):
            raise ValueError('Given "map_to" session is not part of the multisession results')
    elif isinstance(map_to, int):
        if not any(matches := np.equal(sess_ids, map_to)):
            raise ValueError('Given "map_to" session is not part of the multisession results')
        # just take the first session name if there are multiple
        map_to = str(sessions.iloc[np.flatnonzero(matches)[0]].at['sess_name'])
    else:
        if map_to is not None:
            raise ValueError('Unsupported type for "map_to" session')

        # Find session closest to the middle based on saved offsets
        offsets_um = alignment.load_offsets_for_sessions(
            mouse_id, sess_ids=sess_ids, rec_type=rec_type, tags=tags, filename_fmt=offset_file)

        if 'x' not in offsets_um or 'y' not in offsets_um:
            logging.info('Rigid offsets file does not have X/Y positions')
            map_to = str(sessions.iloc[0].at['sess_name'])
            logging.info(f'Defaulting to mapping to the first session ({map_to})')
        else:
            offsets_xy = offsets_um.loc[:, ['x', 'y']]
            center = (offsets_xy.max() + offsets_xy.min()) / 2
            from_center = (offsets_xy - center).distance()
            map_to_id = from_center.idxmin()
            map_to = str(sessions.loc[np.equal(sess_ids, map_to_id), 'sess_name'][0])

    # Step 2: Make session mapping data and contours, plus some other info
    mapping_data, raw_footprints = alignment.rebuild_session_mapping_data_and_footprints(align_results, offset_file)
    assert len(raw_footprints) > 0, 'No sessions?'
    dims = raw_footprints[0].dims

    # Filter cells initially by fraction of accepting sessions
    b_include = (all_cells.frac_sessions_accepting >= frac_accepted_threshold).to_numpy()

    n_cells = all_cells.shape[0]
    all_contours: list[dict[int, np.ndarray]] = [{} for _ in range(n_cells)]  # for each cell, maps from session indices to X/Y contours
    raw_mapped_footprints: list[FootprintsPerPlane] = []

    min_z, max_z = np.inf, -np.inf  # min, max Z location
    plot_borders = [BorderSpec.maximal(dims) for _ in range(n_cells)]  # X/Y borders to use for plotting

    for sess_ind, (data, fp) in enumerate(zip(mapping_data, raw_footprints)):
        min_z = min(min_z, min(fp.z_positions))
        max_z = max(max_z, max(fp.z_positions))

        # map footprints to the target session
        if map_to != data.sess_name:
            fp.remap(data.xy_mappings_to_others, map_to)
        raw_mapped_footprints.append(deepcopy(fp))

        # compute the contours on each plane
        contours = [
            [c['coordinates'] for c in my_get_contours(A, fp.dims, thr=contour_thr, thr_method=contour_method)]
            for A in fp.data]

        # add contours to the dict for each union cell only if it is nonempty
        for i_cell, (union_id, nonempty_planes) in enumerate(zip(data.matchings, fp.nonempty.T)):
            if not b_include[union_id]:
                continue

            c = np.array([[np.nan, np.nan]])
            for plane_contours, nonempty, plane_bboxes in zip(contours, nonempty_planes, fp.bboxes):
                if nonempty:
                    if plane_contours[i_cell].size > 0:
                        c = np.concatenate([c, plane_contours[i_cell], np.array([[np.nan, np.nan]])], axis=0)   
                    
                    # update borders
                    plot_borders[union_id] = BorderSpec.min(plot_borders[union_id], plane_bboxes[i_cell].decreased(border_pix))
                    plot_borders[union_id] = plot_borders[union_id].enclosing_square(dims)  # make it square
            all_contours[union_id][sess_ind] = c

        # make probability/likelihood maps
        fp.binarize()
        fp.smooth(sigma=smoothing_sigma)
    likelihood_footprints = raw_footprints

    # filter out any empty cells (shouldn't really happen...)
    b_include = np.logical_and(b_include, [border.is_center_nonempty(dims) for border in plot_borders])
    valid_cell_ids = [int(ind) for ind in np.flatnonzero(b_include)]

    min_raw_val = min(
        min(
            fps[:, nonempty].min() 
            for fps, nonempty in zip(sess_fps.data, sess_fps.nonempty))
        for sess_fps in raw_mapped_footprints)
    
    max_raw_val = max(
        max(
            fps[:, nonempty].max() 
            for fps, nonempty in zip(sess_fps.data, sess_fps.nonempty))
        for sess_fps in raw_mapped_footprints)

    # make cached mappings from cell ID to arrays to plot
    matchings = [data.matchings for data in mapping_data]
    get_likelihood_interpolator = footprint_interpolator_per_cell(likelihood_footprints, matchings, plot_borders, cached=True)
    get_maxproj = maxproj_per_cell(raw_mapped_footprints, matchings, plot_borders, cached=True)

    # # Step 3: Make functions for DynamicMaps
    # # declare common dimensions ahead of time
    # cell_dimension = hv.Dimension(
    #     ('cell_id', 'Union cell ID'), values=valid_cell_ids, 
    #     range=(valid_cell_ids[0], valid_cell_ids[-1]), default=valid_cell_ids[0])
    # z_dimension = hv.Dimension(
    #     ('z', 'Z position'), unit='um', step=0.2, range=(min_z, max_z),
    #     default=np.clip(np.round((min_z+max_z) / 2), min_z, max_z))
    # intensity_dimension = hv.Dimension(('intensity', 'Max of raw masks'), range=(min_raw_val, max_raw_val))
    # likelihood_dimension = hv.Dimension(('likelihood', 'Likelihood of cell being here'), range=(0., 1.))
    # session_dimension = hv.Dimension(('sess', 'Session index'), range=(0, len(session_data) - 1), default=0)

    # def max_proj(cell_id: int) -> hv.Overlay:
    #     cell_contours = all_contours[cell_id]
    #     yslice, xslice = plot_borders[cell_id].slices(dims)
    #     maxproj = get_maxproj(cell_id)
    #
    #     imageplot = hv.Image(maxproj, vdims=intensity_dimension,
    #                          bounds=(0, 0, maxproj.shape[1], maxproj.shape[0]), label='Projection'
    #                          ).opts(cmap=cmap, invert_yaxis=True)
    #
    #     # overlay contours
    #     contour_list = [{'x': cell_contours[ind][:, 0] - xslice.start if ind in cell_contours else [],
    #                      'y': cell_contours[ind][:, 1] - yslice.start if ind in cell_contours else [],
    #                      'sess': ind} for ind in range(len(session_data))]
    #     contours = hv.Path(contour_list, vdims=session_dimension, label='Contours'
    #                        ).opts(color='sess', cmap=contour_cmap, colorbar=True)
    #
    #     return hv.Overlay([imageplot, contours], label='Projection')    

    # def interp_rois(cell_id: int, z_pos: float) -> hv.Image:
    #     interpolator = get_likelihood_interpolator(cell_id)
    #     im = interpolator(z_pos)
    #     return hv.Image(interpolator(z_pos), bounds=(0, 0, im.shape[1], im.shape[0]), vdims=likelihood_dimension).opts(
    #         cmap=cmap, invert_yaxis=True)  # type: ignore
        
    # max_proj_plot = hv.DynamicMap(max_proj, kdims=[cell_dimension], label='ProjectionAndContours').opts(framewise=True)
    # interp_plot = hv.DynamicMap(interp_rois, kdims=[cell_dimension, z_dimension], label='Interpolated').opts(framewise=True)
    # return max_proj_plot + interp_plot  # type: ignore

    # raw Bokeh version
    # make sliders
    default_cell_id = valid_cell_ids[0]
    cell_slider = CategoricalSlider(categories=[str(cell) for cell in valid_cell_ids], value=str(default_cell_id), title='Cell ID')
    default_z = np.clip(np.round((min_z+max_z) / 2), min_z, max_z)
    z_slider = Slider(start=min_z, end=max_z, value=default_z, step=0.2, title='Z position', direction='rtl')

    # make plots
    image_tooltips = [("(x,y)", "($x, $y)"), ('value', '@image')]

    maxproj_plot = figure(title='Max-Z-projected raw footprints', frame_width=400, frame_height=400,
                          match_aspect=True, tooltips=image_tooltips)
    maxproj_plot.x_range.update(range_padding=0)
    maxproj_plot.y_range.update(range_padding=0, flipped=True)

    maxproj = get_maxproj(default_cell_id)
    intensity_mapper = LinearColorMapper(low=min_raw_val, high=max_raw_val, palette=cmap(256))
    yslice, xslice = plot_borders[default_cell_id].slices(dims)
    maxproj_im = maxproj_plot.image(image=[maxproj], x=xslice.start, y=yslice.start,
                                    dw=maxproj.shape[1], dh=maxproj.shape[0], color_mapper=intensity_mapper)
    session_mapper = CategoricalColorMapper(factors=sess_names, palette=contour_cmap(len(sess_names)))
    # TODO add contours to plot

    interp_plot = figure(title='Interpolated likelihood', frame_width=400, frame_height=400,
                         match_aspect=True, tooltips=image_tooltips)
    interp_plot.x_range.update(range_padding=0)
    interp_plot.y_range.update(range_padding=0, flipped=True)

    interpolator = get_likelihood_interpolator(default_cell_id)
    lik_im = interpolator(default_z)
    likelihood_mapper = LinearColorMapper(low=0., high=1., palette=cmap(256))
    interp_im = interp_plot.image(image=[lik_im], x=xslice.start, y=yslice.start,
                                  dw=lik_im.shape[1], dh=lik_im.shape[0], color_mapper=likelihood_mapper)

    # callbacks for sliders
    def cell_cb(attr: str, old: str, new: str):
        assert attr == 'value', 'Callback called for unexpected event'
        if new == old:
            return
        
        cell_id = int(new)
        maxproj = get_maxproj(cell_id)
        maxproj_im.data_source.patch({'image': [(0, maxproj)]})  # type: ignore  # replace old image with new one

        interpolator = get_likelihood_interpolator(int(new))
        lik_im = interpolator(z_slider.value)
        interp_im.data_source.patch({'image': [(0, lik_im)]})  # type: ignore

        # update bounds
        yslice, xslice = plot_borders[cell_id].slices(dims)
        maxproj_im.glyph.update(x=xslice.start, y=yslice.start, dh=maxproj.shape[0], dw=maxproj.shape[1])
        interp_im.glyph.update(x=xslice.start, y=yslice.start, dh=lik_im.shape[0], dw=lik_im.shape[1])
        for range in (maxproj_plot.x_range, maxproj_plot.y_range, interp_plot.x_range, interp_plot.y_range):
            range.update(start=np.nan, end=np.nan)  # re-centers the image
    
    cell_slider.on_change('value', cell_cb)

    def z_cb(attr: str, old: float, new: float):
        assert attr == 'value', 'Callback called for unexpected event'
        if new == old:
            return
        
        interpolator = get_likelihood_interpolator(int(cell_slider.value))
        lik_im = interpolator(new)
        interp_im.data_source.patch({'image': [(0, lik_im)]})  # type: ignore

    z_slider.on_change('value', z_cb)

    plots = row(maxproj_plot, interp_plot, column(cell_slider, z_slider))
    curdoc().theme = 'dark_minimal'
    return pn.pane.Bokeh(plots, theme='dark_minimal')


def plot_mean_in_square_region(sessinfo: 'cma.SessionAnalysis', center: tuple[int, int], radius=7, show_accepted=True,
                               bg_img: Optional[np.ndarray] = None, **plot_options):
    """
    Make a figure with a square region highlighted within a contour plot on the left
    and average raw signal within that region on the right
    center: (y, x) pixel to center on (integers)
    radius: use a square with side length 2*radius
    show_accepted: If true, shows blue and red contours for accepted/rejected cells and box in white.
                   If false, shows contours in white and box in red.
    bg_img: background image to plot; if None uses the mean projection
    plot_options: passed on to my_plot_contours (e.g. can override accept/reject colors, vmin, vmax)
    """
    if sessinfo.mmap_file_transposed is None:
        raise RuntimeError('Motion correction or transpose not run')

    if sessinfo.cnmf_fit is None:
        raise RuntimeError('CNMF not run or not selected')

    Yr, dims, T = caiman.load_memmap(sessinfo.mmap_file_transposed)
    Yr_3d = np.reshape(Yr, dims + (T,), order='F')
    if bg_img is None:
        bg_img = sessinfo.get_projection('mean')

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), width_ratios=(1, 2))
    # plot area being focused on
    # override default accepted cell color
    if 'accept_color' not in plot_options:
        plot_options['accept_color'] = 'b'
    my_plot_contours(sessinfo.cnmf_fit.estimates, img=bg_img, display_numbers=True,
                     idx=(sessinfo.cnmf_fit.estimates.idx_components if show_accepted else None),
                     ax=axs[0], **plot_options)

    axs[0].set_ylim([center[0]+radius*7, center[0]-radius*7])
    axs[0].set_xlim([center[1]-radius*7, center[1]+radius*7])
    axs[0].vlines([center[1]-radius, center[1]+radius], center[0]-radius, center[0]+radius, 'w' if show_accepted else 'r')
    axs[0].hlines([center[0]-radius, center[0]+radius], center[1]-radius, center[1]+radius, 'w' if show_accepted else 'r')
    axs[0].set_title('Region of interest')

    mean_activity = np.mean(Yr_3d[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius], axis=(0, 1))
    axs[1].plot(mean_activity)
    axs[1].set_xlabel('samples')
    axs[1].set_ylabel('calcium signal')
    axs[1].set_title('Mean raw activity')

    fig.tight_layout()


# for mosaics of mean images
def make_alignment_plot(mouse_id, rec_type, channel, sessions, n_cols=7):
    images = []
    names = []
    for sess_id in sessions:
        sbx_files = find_sess_sbx_files(mouse_id, sess_id, rec_type=rec_type, remove_ext=True)
        if len(sbx_files) == 0:
            continue
        mean_image = average_raw_frames(sbx_files, channel=channel, frames=slice(0, 30), dview=cma.cluster.dview)
        for kplane in range(mean_image.shape[2]):
            while len(images) <= kplane:
                images.append([])
            images[kplane].append(mean_image[:, :, kplane])
        names.append(f'Session {sess_id}')

    n_sessions = len(names)
    n_rows = math.ceil(n_sessions / n_cols)

    widgets = []
    for k_plane, plane_images in enumerate(images):
        widgets.append(HTML(f'<h1>Plane {k_plane}</h1>'))
        grid = fpl.GridPlot((n_rows, n_cols), size=(320 * n_cols, 255 * n_rows), controller_ids='sync')  # type: ignore
        for image, subplot, name in zip(plane_images, grid, names):
            subplot.name = name
            subplot.add_image(image, cmap='viridis')
        grid_output = grid.show()
        widgets.append(grid_output)
    return VBox(widgets)

def save_plane_image(alignment_plot: VBox, plane: int, name: str, timeout: int = 60):
    """Save a snapshot of one of the plane plots to a .png file"""
    t0 = time.time()
    frame = None
    while frame is None and time.time() < t0 + timeout:
        frame = alignment_plot.children[plane*2 + 1].frame.canvas.get_frame()
        time.sleep(1)
    if frame is None:
        raise TimeoutError(f'Could not capture frame in {timeout} seconds')

    frame_im = Image.fromarray(frame)
    filename = name if '.png' in name else name + '.png'
    frame_im.save(filename)

def make_plots_taller(plots: VBox, new_max_height = 3000):
    """Fix the max_height of each child that is a JupyterOutputContext"""
    max_height = f'{new_max_height}px'
    for child in plots.children:
        if isinstance(child, JupyterOutputContext):
            child.frame.canvas.layout.max_height = max_height