"""
Functions for doing grid searches of parameters using mesmerize-core
"""
from itertools import product
import os
from pathlib import Path
from typing import Any, Optional, Iterable, Union, Mapping, Sequence, cast

import pandas as pd
import matplotlib.pyplot as plt

from mesmerize_core.caiman_extensions.common import Waitable

from cmcode import caiman_analysis as cma, cmcustom
from cmcode.caiman_params import AnalysisStage
from cmcode.util import types


ParamGrid = Mapping[tuple[str, str], Iterable]
def do_cnmf_gridsearch(sessdata: 'cma.SessionAnalysis', params_to_search: Union[ParamGrid, Sequence[ParamGrid]],
                       backend: str, cpus_available: Optional[int] = None, max_cpus_per_job: Optional[int] = None, 
                       partition: Optional[Union[str, list[str]]] = None, make_contour_pdfs: bool = False) -> list[Waitable]:
    """
    Run CNMF on each combination of params_to_search (applied on top of params_base).
    Each key of params_to_search is (group, param_name) (e.g., ('init', 'K')).
    Use backend='slurm' to run on a cluster that uses SLURM.
    """
    # get all combinations of search parameters
    if not isinstance(params_to_search, Sequence):
        params_to_search = [params_to_search]
    
    param_dicts = []
    for param_grid in params_to_search:
        param_keys, param_vals_each = zip(*param_grid.items())
        for valset in product(*param_vals_each):
            param_dicts.append({k: val for k, val in zip(param_keys, valset)})

    # set CPUs in environment variable before starting
    n_procs_var = 'MESMERIZE_N_PROCESSES'
    if backend == 'slurm' and cpus_available is not None:
        cpus_per_job = cpus_available // len(param_dicts)
        if max_cpus_per_job is not None:
            cpus_per_job = min(cpus_per_job, max_cpus_per_job)
        cpus_per_job = max(1, cpus_per_job - 1)
        os.environ[n_procs_var] = str(cpus_per_job)
    elif n_procs_var in os.environ:
        del os.environ[n_procs_var]

    procs: list[Waitable] = []
    uuids = []

    for param_dict in param_dicts:
        param_changes: dict[str, dict[str, Any]] = {}
        for (group, key), val in param_dict.items():
            if group not in param_changes:
                param_changes[group] = {}
            param_changes[group][key] = val
        
        sessdata.update_params(param_changes)

        # get up to the point of doing CNMF for these parameters if necessary
        if sessdata.last_valid_stage < AnalysisStage.CONVERT:
            sessdata.convert_to_tif()
        
        if sessdata.last_valid_stage < AnalysisStage.TRANSPOSE:
            sessdata.do_motion_correction()

        uuid, proc = sessdata.start_cnmf_with_mescore(backend=backend, wait=False, partition=partition)

        uuids.append(uuid)
        procs.append(proc)

    if make_contour_pdfs:
        # wait for all to finish
        for proc in procs:
            proc.wait()

        # make contour plot for each run
        batch = sessdata.get_gridsearch_results()
        for uuid in uuids:
            make_contour_pdf(batch, uuid)
    
    return procs


class GridsearchError(RuntimeError):
    """Error shown after a gridsearch has failed, which displays the traceback of the first failed run if possible."""
    def __init__(self, batch: 'types.MescoreBatch', orig_nruns: int):
        """orig_nruns: How many runs were in the batch before this gridsearch."""
        new_runs = batch.iloc[orig_nruns:, :]
        error_ind = None
        any_not_run = False
        for ind, row in new_runs.iterrows():
            if row.outputs is None:
                any_not_run = True
            elif not row.outputs['success']:
                error_ind = ind
                break

        if error_ind is not None:
            # get backtrace
            err_uuid = new_runs.at[ind, 'uuid']
            err_outputs = cast(dict, new_runs.at[ind, 'outputs'])
            err_bt = err_outputs['traceback']
            super().__init__(f'Error running gridsearch - first error in UUID {err_uuid}. Backtrace:\n' + err_bt)
        elif not any_not_run:
            super().__init__('CNMF succeeded, but another error was raised in run script - see above.')
        else:
            super().__init__('Error running gridsearch, but no error is saved in the dataframe - check log files.')


def make_contour_pdf(batch: pd.DataFrame, uuid: str):
    """Produce PDF of contours for a given batch item, with accepted/rejected ROIs marked."""
    res_series = batch.loc[batch.uuid == uuid, :].iloc[0]
    corr_bg = res_series.caiman.get_corr_image()
    cnmf = res_series.cnmf.get_output()
    output_dir = Path(res_series.cnmf.get_output_path()).parent
    pdf_path = output_dir / f'{uuid}_contours.pdf'

    with plt.ioff():
        fig = cmcustom.my_plot_contours(cnmf.estimates, img=corr_bg, idx=cnmf.estimates.idx_components)
        cma.save_contour_plot_as_pdf(fig, pdf_path)
        plt.close(fig)
