# cmcode: a suite for organized 2P data analysis

[![DOI](https://zenodo.org/badge/1063561641.svg)](https://zenodo.org/badge/latestdoi/1063561641)

This repository contains Python code to manage signal extraction from multi-plane two-photon (2P) calcium imaging recordings using [CNMF](https://www.sciencedirect.com/science/article/pii/S0896627315010843) (as implemented in [CaImAn](https://caiman.readthedocs.io/en/latest/)). It also has various convenient functions for visualizing and interacting with results on a remote server. The `alignment` module has routines to aid matching cells between different sessions with slightly different fields of view (including in Z). Everything is designed to work seamlessly across Windows and Linux workstations, after some initial configuration.

## Installation

First, install [Pixi](https://pixi.prefix.dev/latest/), and ensure it is updated to a recent version (>= 0.71.0). 

Then install the environment using pixi. If you have a GPU and want to use it for pytorch, use the default environment:

`pixi shell`

Otherwise, use the 'cpu' environment:

`pixi shell -e cpu`

**One-time CaImAn setup**: With the environment activated, run `caimanmanager install --inplace`. See [here](https://caiman.readthedocs.io/en/latest/Installation.html#section-2-set-up-demos-with-caimanmanager) for what this does.
* If you want to override the default location for CaImAn demos and data files, first configure the `CAIMAN_DATA` environment variable (or `local_environment.py`) - see "Initial setup" below. Then do the `caimanmanager install --inplace`.

## Usage

### Initial setup
You need to give cmcode some context about your data and computing environment before it can work seamlessly. The easiest way is to set environment variables `CAIMAN_DATA` and `CMCODE_ROOT_DATA_DIR`, and the best way to do this is in the pixi.toml file. There are some commented-out lines that you can uncomment to set these variables. Note that `CAIMAN_DATA` does not need to be set if you are using CaImAn's default location (your home directory), but `CMCODE_ROOT_DATA_DIR` is required.

You can manage more settings by creating and "applying" a `cmcode.util.environment.ComputingEnvironment` object. If you define a variable `computing_environment` in a file called `cmcode/private/local_environment.py` (ignored by git), this will be applied automatically every time cmcode is imported and will override environment variable settings. An example local environment file is at `cmcode/private/example_local_environment.py` (don't forget to save as `local_environment.py` before making changes).

You can set the following fields of `ComputingEnvironment`: 
* `root_data_dir` (corresponds to `CMCODE_ROOT_DATA_DIR`): A parent directory containing your data. It should contain at least the following directory structure:

    ```
    <root_data_dir>/
    ├── raw/
    │   └── <rec_type>/
    │       └── <mouse_id>/
    │           ├── <mouse_id>_000_000.mat
    |           ├── <mouse_id>_000_000.sbx
    |           ├── <mouse_id>_000_001.mat
    |           ├── <mouse_id>_000_001.sbx
    |           ...
    └── processed/
    ```
    * `rec_type` is a string identifier for a particular experiment (or type of recording in an experiment). The default in some places is 'learning_ppc', currently.
    * `mouse_id` can be an integer or a string (all code *should* work with non-numeric ID, but numeric is safer)
    * The second component of each filename is the `sess_id`, and the third is the trial number. These *must* both be integers (matching convention for Scanbox).
    * cmcode works with a single "trial" per session or multiple. If there are multiple trials, they are concatenated together in the first step and processed as a single movie (while keeping track of the trial boundaries).

* `caiman_data_dir` (corresponds to `CAIMAN_DATA`): Used for CaImAn, and I like to use it as a home for notebooks, scripts, etc. See [CaImAn docs](https://caiman.readthedocs.io/en/latest/Installation.html#section-2-set-up-demos-with-caimanmanager). Note, if you change this, you should run `caimanmanager install --inplace`.
* `network_hosts`: A `cmcode.remote.host_info.NetworkInfo` object, essentially a mapping from names to information about various computers you want to be able to run analyses on (e.g. with the `caimanlab` command). See the example file for how to set this up.
* `root_mappings`: A list of `cmcode.util.paths.EquivalentPaths` objects, each of which contains one set of root paths for Windows computers and another for Unix/Linux computers, all of which you want to assume refer to the same location. This allows you to work across computers with different network drive setups and still be able to find your data based on saved paths that may be from a different computer.
*  `ipyprofile_dir`: A shared folder to override the default if using ipyparallel (which is not the default parallel computing library).

### Remote execution

If you want to run code in JupyterLab on a remote server, after installing cmcode on that machine and configuring `network_hosts` as described above and in `example_local_environment.py`, you can just run `caimanlab <host_name>` (where `<host_name>` is the `name` field of the `add_host` call - typically just the actual hostname, but it can be an alias or whatever you want). It may take a minute or two for the environment to start on the remote machine, but eventually the JupyterLab tab should open in your browser automatically.

### Example processing pipeline

A full example notebook is coming soon, including interactive exploration of results and multi-session alignment, but here is example code to process one session of data.

```python
from cmcode import caiman_analysis as cma

rec_type = 'ymaze'  # must match a directory name under <root_data_dir>/raw
mouse_id = 'TM10'   # must match a directory name under <root_data_dir>/raw/<rec_type>
sess_id = 1         # SBX files with this number as 2nd element (padded with 0s) will be used

param_overrides = {  # see CaImAn CNMFParams documentation; there are some added ones defined in caiman_params.py
  'mcorr_extra': {'use_suite2p': True},
  'cnmf_extra': {
    'seed_params': {
      'norm_medw': 25,
      'use_cellpose': True,
      'cellpose_params': {
          'flow_threshold': None,
          'cellprob_threshold': -3.
      }
    }
  }
}

sessinfo = cma.SessionAnalysis(mouse_id, sess_id, rec_type=rec_type, param_overrides=param_overrides)

# convert files from SBX to TIF and concatenate trials
sessinfo.convert_to_tif()

# do motion correction and save C-order movie for CNMF
# (you can also do these in separate steps with do_mcorr_only() and concat_and_transpose())
sessinfo.do_motion_correction()

# run CNMF
sessinfo.do_cnmf()

# see a list of CNMF runs that you've done
sessinfo.get_gridsearch_results()

# see differences in parameters between these runs
sessinfo.get_gridsearch_diffs()

# here you would examine the results, tweak quality criteria and manually accept/reject as necessary
# notebook for this coming soon

# export signals to .mat (or .npz) format, along with a .pkl file containing metadata
# saves to <root_data_dir>/processed/<rec_type>/<mouse_id>/export/<mouse_id>_<sess_id>_<uuid>_{activity.mat,metadata.pkl}
# the UUID is a unique identifier for this CNMF run (shown in the calls to get_gridsearch_results and get_gridsearch_diffs above)
sessinfo.save_estimates(format='.mat')  # or (.npz)
```

## Design

This package started as just "glue" code to run CaImAn on my data, but as the complexity of different processing options and libraries I wanted to try grew, I realized I had to make it more organized to keep track of what settings I had tried, what the currently saved data reflects, and which later processing stages had to be re-run after changing something earlier in the pipeline. To do so, cmcode maintains invariants within the `SessionAnalysis` object every time parameters are changed; all results that may depend on these parameters are removed from the object, and these stages must be re-run to do anything that depends on them. Each of these data fields that may be deleted doubles as a boolean flag that tells the program what stage we are at. For example, let's say `do_motion_correction()` has completed, and you're ready to run CNMF. At this point, the following fields are set (non-None):

* `sessinfo.plane_tifs` (flag for CONVERT stage/`convert_to_tif()`)
* `sessinfo.mc_result` (flag for MCORR stage/`do_mcorr_only()`/first part of `do_motion_correction()`)
* `sessinfo.mmap_file_transposed` (flag for TRANSPOSE stage/`concat_and_transpose()`/second part of `do_motion_correction()`)

The following fields are None:

* `sessinfo.cnmf_fit_filename` (flag for CNMF stage/`do_cnmf()`)

You can see what stage you are at with the property `last_valid_stage`:
```python
sessinfo.last_valid_stage  # prints "AnalysisStage.TRANSPOSE"
```

Now, let's say you realize that you still have motion artifacts, and you need to increase the `max_deviation_rigid` to do more non-rigid correction. You would change that with `update_params`, and then you would see that the last valid stage has automatically updated:

```python
sessinfo.update_params({'motion': {'max_deviation_rigid': 10}})
sessinfo.last_valid_stage  # prints "AnalysisStage.CONVERT"
sessinfo.mc_result is None  # prints "True"
```

For a reasonable workflow, this system of tracking and invalidating stages happens invisibly in the background, but this way you will get an error if you make a mistake that could lead to incorrect results due to using outdated data. This could be subtle: in this case, maybe you remember to run motion correction again using `do_mcorr_only()`, but you forget to do the transpose step. If you try to run `do_cnmf()`, instead of using the old transposed data, you'll get an error.

To handle the same kind of error, but across Python sessions, each result file is saved on disk alongside a JSON file containing parameters for the corresponding stage and earlier, and another params file is saved when the `SessionAnalysis` object itself is saved (which happens at the end of every processing step, or manually with the `save()` method). All the parameters are validated with Pydantic to ensure that they can be reliably serialized/deserialized to/from JSON (that is why we must depend on an un-merged fork of CaImAn (as of 2026) that includes this PR: https://github.com/flatironinstitute/CaImAn/pull/1566). All of the main processing functions will try to load a pre-computed result if possible (you can ensure this by setting `load=True`, or prevent it by setting `load=False`). But if the params saved with the result don't match the current ones, it won't be loaded, and a new result will be computed and saved instead.

## License
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
