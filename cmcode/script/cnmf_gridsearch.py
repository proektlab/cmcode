import argparse
import os
from socket import gethostname
import subprocess

import mesmerize_core as mc
from cmcode import caiman_analysis as cma, gridsearch_analysis
from cmcode.remote.host_info import get_cpu_limits
from cmcode.util.paths import normalize_path, get_network_hosts

# test whether this computer has slurm
if os.name != 'nt' and subprocess.run(['which', 'sbatch'], stdout=subprocess.DEVNULL).returncode == 0:
    backend = 'slurm'
else:
    backend = 'local'

# cluster info
partition = None
cpus_available, max_cpus_per_job = get_cpu_limits(get_network_hosts(), gethostname(), partition=partition)


if __name__ == '__main__':
    """
    Takes existing pickled SessionAnalysis as argument and does a CNMF grid search
    of params listed in analysis.params_to_search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path')
    parser.add_argument('-w', '--wait', action='store_true', default=False)
    args = parser.parse_args()

    if args.pkl_path is None:
        raise RuntimeError('Must provide path to pickle ')

    pkl_path = normalize_path(args.pkl_path)

    if not isinstance(pkl_path, str) or not os.path.isfile(pkl_path):
        raise RuntimeError('Path to pickle file is invalid')

    sessdata = cma.load(pkl_path)
    if not hasattr(sessdata, 'params_to_search') or sessdata.params_to_search is None:
        raise RuntimeError('Parameters to search not found in SessionAnalysis object')
    
    procs = gridsearch_analysis.do_cnmf_gridsearch(sessdata, sessdata.params_to_search, backend=backend, max_cpus_per_job=max_cpus_per_job,
                                                   cpus_available=cpus_available, partition=partition, make_contour_pdfs=False)
    
    # Block this process until all runs are complete
    if args.wait:
        for proc in procs:
            proc.wait()  # will throw if the run was unsuccessful
