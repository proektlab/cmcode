"""Routines for setting up multiprocessing clusters"""
from dataclasses import dataclass
import logging
from multiprocessing.pool import Pool
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Union, Callable, Iterable, TypeVar, Generic, Sequence

from caiman.cluster import stop_server, setup_cluster
from ipyparallel import DirectView, Client

from cmcode.remote import remoteops, host_info

RetVal = TypeVar('RetVal')
class FuturesDviewAdapter:
    """Object that holds a concurrent.futures Executor and works like a 'dview'"""
    Client = namedtuple('Client', ['profile'])  # spoof dview.client.profile

    def __init__(self, max_workers: int):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.map = self.map_sync
        self.results = {}  # spoof dview.results.clear()
        self.client = FuturesDviewAdapter.Client(profile='default')

    def map_sync(self, fn: Callable[..., RetVal], args: Iterable) -> Sequence[RetVal]:
        return list(self.executor.map(fn, args))
    
    class AsyncResultAdapter(Generic[RetVal]):
        """Doesn't actually process asynchronously, but pretends to"""
        def __init__(self, dview: 'FuturesDviewAdapter', fn: Callable[..., RetVal], args: Iterable):
            self.result = dview.map_sync(fn, args)
        
        def get(self, timeout: int = 0) -> Sequence[RetVal]:
            return self.result
        
        def __iter__(self):
            return (result for result in self.result)

    def map_async(self, fn: Callable[..., RetVal], args: Iterable) -> AsyncResultAdapter[RetVal]:
        return self.AsyncResultAdapter(self, fn, args)
    
    def __len__(self):
        return self.max_workers
    
    def clear(self):
        pass


@dataclass
class IppClusterInfo:
    """Information about an open ipyparallel cluster"""
    cluster_id: str
    shared_profile_dir: Optional[str]


class ClusterInfo:
    """Information about a parallel (or single) processing environment"""
    def __init__(self, cluster_obj: Union[None, Client, Pool, FuturesDviewAdapter], ncores: int = 1, ipp_info: Optional[IppClusterInfo] = None):
        self._cluster_obj = cluster_obj
        self._ncores = ncores
        self.ipp_info = ipp_info
    
    @property
    def ncores(self) -> int:
        """Get count of current cores"""
        if isinstance(self._cluster_obj, Client):
            return len(self._cluster_obj.ids)
        else:
            return self._ncores
    
    @property
    def dview(self) -> Union[None, DirectView, Pool, FuturesDviewAdapter]:
        """Get dview including all current cores"""
        if isinstance(self._cluster_obj, Client):
            return self._cluster_obj[:]
        else:
            return self._cluster_obj


class Cluster:
    """Object responsible for starting, accessing, and stopping cluster"""
    __slots__ = 'info', 'default_params'

    def __init__(self, **default_params):
        self.info: Optional[ClusterInfo] = None
        self.default_params = default_params

    def __getattribute__(self, name):
        """defer to ClusterInfo object"""
        if hasattr(Cluster, name):
            return object.__getattribute__(self, name)
        
        if self.info is None:
            logging.info('Starting cluster automatically')
            self.start(**self.default_params)
        return object.__getattribute__(self.info, name)

    def start(self, n_cores_to_exclude=1, backend='futures', threaded=False, ipp_cluster_args: Optional[dict] = None,
              host_specs: Optional[list[str]] = None):
        """
        Setup parallel processing environment with the given backend ('single' = no parallel processing).
        If this function has been called already, the current setup should be passed as curr_cluster_info.
        hosts defaults to just the current computer. If hosts is not None, a shared profile dir defined by the project is used
        to manage ipyparallel workers (even if the only entry in hosts is localhost).
        """
        if threaded is None:
            threaded = backend == 'ipyparallel'
        
        if self.info is not None:
            if self.info.dview is not None:
                logging.info(f'Closing previous cluster')
            self.shutdown()
        
        if backend == 'single':
            self.info = ClusterInfo(None)
            return

        logging.info('Setting up new cluster')
        if backend == 'ipyparallel':
            if host_specs is None:
                host_specs = ['localhost']  # just use this computer
                shared_profile_dir = False
            else:
                shared_profile_dir = True

            cluster_args = {'host_specs': host_specs, 'debug': False, 'shared_profile_dir': shared_profile_dir,
                            'n_cores_to_exclude': n_cores_to_exclude}
            if ipp_cluster_args is not None:
                cluster_args.update(ipp_cluster_args)
            if not threaded:
                cluster_args['max_threads_per_worker'] = 1

            client, cluster_id, n_cores, profile_dir = remoteops.start_network_cluster(**cluster_args)
            ipp_cluster_info = IppClusterInfo(cluster_id=cluster_id, shared_profile_dir=profile_dir)

            cluster_info = ClusterInfo(client, ipp_info=ipp_cluster_info)
        else:
            if threaded or host_specs is not None:
                raise Exception('Requested cluster features only supported on ipyparallel')
            n_local_cores = host_info.get_localhost_info().n_cores
            n_procs = n_local_cores - n_cores_to_exclude

            if backend == 'futures':
                dview = FuturesDviewAdapter(max_workers=n_procs)
                cluster_info = ClusterInfo(dview, ncores=n_procs)
                n_cores = n_procs
            else:
                _, dview, n_cores = setup_cluster(backend=backend, n_processes=n_procs, ignore_preexisting=True)
                assert n_cores is not None, 'caiman.setup_cluster should never return None for n_cores in practice'
                cluster_info = ClusterInfo(dview, ncores=n_cores)
        
        logging.info(f'Successfully initialized multicore processing with a pool of {n_cores} CPU cores')
        self.info = cluster_info

    def shutdown(self):
        if self.info is not None:
            if self.info.dview is None:
                pass
            elif self.info.ipp_info is not None:
                # close network cluster
                remoteops.stop_network_cluster(**vars(self.info.ipp_info))
            elif isinstance(self.info._cluster_obj, FuturesDviewAdapter):
                self.info._cluster_obj.executor.shutdown()
            else:
                stop_server(dview=self.info.dview)
            self.info = None
