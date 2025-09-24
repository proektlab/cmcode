from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from cmcode.remote import host_info
from cmcode.remote.host_info import NetworkInfo
from cmcode.util import paths
from cmcode.util.paths import EquivalentPaths


@dataclass
class ComputingEnvironment:
    """
    Object with information about local computing environment.

    Call apply() to call global variable setter for each field.

    If cmcode/private/local_environment.py exists and contains a variable of this type called
    computing_environment, it will automatically be applied using the appropriate
    global variable setters when cmcode is imported.
    """
    caiman_data_dir: Optional[Union[str, Path]] = None # path to caiman_data, must be set before root_data_dir
    network_hosts: NetworkInfo = NetworkInfo()         # hosts available to do work
    root_mappings: Iterable[EquivalentPaths] = ()      # equivalent root paths
    root_data_dir: Optional[Union[str, Path]] = None   # path with 'processed' and 'raw' folders (required to do anything)
    ipyprofile_dir: Optional[Union[str, Path]] = None  # shared folder for ipyparallel


    def apply(self):
        """Apply ComputingEnvironment by calling global variable setters"""
        # caiman_data_dir must be done first to avoid import order issue
        paths.set_caiman_data_dir(self.caiman_data_dir)
        host_info.set_network_hosts(self.network_hosts)
        paths.set_root_mappings(self.root_mappings)
        paths.set_ipyprofile_dir(self.ipyprofile_dir)
        if self.root_data_dir is not None:
            paths.set_root_data_dir(self.root_data_dir)