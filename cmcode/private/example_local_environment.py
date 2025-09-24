"""
Example file defining a ComputingEnvironment with paths and local network info.
This can be modified and renamed to 'local_environment.py' to be applied by default upon import.

Each of these settings can also be changed individually through function calls, e.g. to use a
different root data path for a given project. To change the root data dir you would do:

```
from cmcode.util import paths
paths.set_root_data_dir(<your path>)
```

See cmcode/util/environment.py to see how to change the other settings.
"""
from pathlib import PurePosixPath, PurePath

from cmcode.remote.host_info import NetworkInfo, HostInfo
from cmcode.util.paths import EquivalentPaths, EquivalentPathsByPlatform
from cmcode.util.environment import ComputingEnvironment

# To work smoothly across computers that access the same drives (e.g., a NAS) but
# have different OS's or differently-organized filesystems, you can define sets of
# equivalent root paths that point to the same location. These will be used to seamlessly
# translate paths when loading and saving data.
# 
# (Most paths will be saved as relative paths with respect to the *root data dir*
# defined below, but this functionality allows even paths elsewhere on shared drives to
# be used if necessary.)
#
# The simplest case is a drive that is mounted in a different location on Windows vs.
# Unix computers. This is handled by EquivalentPathsByPlatform.
# The first entry in windows_paths and posix_paths will
# be used as the canonical path when translating to Windows and POSIX/Unix respectively.

labstuff = EquivalentPathsByPlatform(
    windows_paths=[
        '\\\\our.wonderful.nas\\labstuff',  # canonical Windows path
        'Z:'],
    posix_paths=['/mnt/labstuff']           # canonical POSIX path
)

# If you have a more complicated situation, you can subclass EquivalentPaths and
# define a custom get_path_for_host function that retrieves the appropriate path
# based on the info in the computer's HostInfo: name, platform, even number of cores?
# Probably name is the most useful, though. For example, maybe your shared home directory is
# called something else on one of your computers.
#
# Do make sure to call the superclass constructor with windows_paths and posix_paths,
# so that it correctly parses incoming paths where the originating host is unknown.

class EquivalentHomedirs(EquivalentPaths):
    posix_main = PurePosixPath('/mnt/userhomes')
    posix_sleepy = PurePosixPath('/mnt/u')

    def __init__(self):
        super().__init__(
            windows_paths=['U:'],
            posix_paths=[self.posix_main, self.posix_sleepy]
        )
    
    def get_path_for_host(self, host: HostInfo) -> PurePath:
        if host.name == 'sleepy':
            return self.posix_sleepy
        elif not host.is_pc:
            return self.posix_main
        else:
            return self.windows_paths[0]

labhomes = EquivalentHomedirs()

root_mappings = [labstuff, labhomes]


# Now we can define some other paths necessary for cmcode to work.
# The get_local_path() method of EquivalentPaths is useful to get the appropriate
# paths regardless of which computer we happen to be running on.
labstuff_local = labstuff.get_local_path()

# Root data dir - this is where raw input data and processed results live, as well as
# a few other utility directories. The raw files should be located at:
#   <root_data_dir>/raw/<rec_type>/<mouse_id>/<mouse_id>_<session_number>_<trial_number>.sbx
root_data_dir = labstuff_local / '2p_imaging'

# Caiman data dir - the location of the caiman_data folder. By default this gets put in
# your home directory when you install CaImAn, and this is only necessary if you have
# overridden that default; otherwise it can be set to None.
# caiman_data_dir = labstuff_local / 'caiman_data'
caiman_data_dir = None

# Shared profile directory for ipyparallel - if you want to use ipyparallel to open a cluster
# spanning multiple hosts, this must be located in a directory accessible to all of them.
# Otherwise it can be set to None.
# ipyprofile_dir = labstuff_local / '.ipython' / 'profile_default'
ipyprofile_dir = None

# Define local network architecture - this is optional but necessary if you want to run
# CNMF remotely or on a cluster (including a SLURM cluster).
network_hosts = NetworkInfo()

# Normal PCs for cluster/remote processing
# See definition of HostInfo class in host_info.py for what each field means
network_hosts.add_host(
    name='happy',
    self_name='happy',
    ssh_name='happy.proektlab',
    is_pc=False,
    envname='caiman',
    n_cores=32
)

network_hosts.add_host(
    name='dopey',
    self_name='Dopey',
    ssh_name='dopey.proektlab',
    is_pc=True,
    envname='caiman-test',
    n_cores=32
)

network_hosts.add_host(
    name='sleepy',
    self_name='sleepy.proektlab.local',
    ssh_name='sleepy.proektlab',
    is_pc=False,
    envname='caiman',
    n_cores=56
)

# SLURM-managed cluster
network_hosts.add_host(
    name='node01.abc',
    self_name='node01.local',
    ssh_user='sally',
    ssh_name='node01.abc',
    is_pc=False,
    n_cores=32,
    envname='caiman'
)

# Can add specific SLURM partitions to run jobs on
network_hosts.add_partition(
    parent_host='node01.abc',
    name='main',
    cores_per_node={
        'node01.local': 32,
        'node02.local': 32
    },
    set_default=True
)

network_hosts.add_partition(
    parent_host='node01.abc',
    name='gpu',
    cores_per_node={
        f'mini{n:02d}': 16 for n in range(1, 17)
    } | {
        'gpu01.abc': 64
    }
)

# Add other entry points to same SLURM cluster
network_hosts.add_host_alias(
    base_host='node01.abc',
    name='node02.abc',
    ssh_name='node02.abc',
    self_name='node02.local'
)

# Put this at the end to use info entered above for local host
# if it is one of the network nodes, rather than trying to infer by introspecting
network_hosts.add_localhost()


computing_environment = ComputingEnvironment(
    caiman_data_dir=caiman_data_dir,
    network_hosts=network_hosts,
    root_mappings=root_mappings,
    root_data_dir=root_data_dir,
    ipyprofile_dir=ipyprofile_dir
)
