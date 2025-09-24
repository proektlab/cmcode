"""Functions to do computations on remote machines/clusters"""
import asyncio
import asyncssh
from contextlib import nullcontext
import logging
import math
import os
import shlex
import socket
import stat
import time
from typing import Iterable, Optional, Union, Any, Callable, Coroutine, Awaitable, TypeVar, ParamSpec

import ipyparallel as ipp

from cmcode.util import paths
from cmcode.remote.host_info import HostInfo, PartitionInfo, WorkerContext, get_network_hosts


# ---- asyncio helpers ---- #

Res = TypeVar('Res')
AsyncFnArgs = ParamSpec('AsyncFnArgs')
def make_sync(wrapped_fn: Callable[AsyncFnArgs, Coroutine[Any, Any, Res]]) -> Callable[AsyncFnArgs, Res]:
    """Decorator to make a synchronous version of any async function"""
    def sync_fn(*args: AsyncFnArgs.args, **kwargs: AsyncFnArgs.kwargs):
        cor = wrapped_fn(*args, **kwargs)
        return asyncio.run(cor)
    return sync_fn


def finish_task(wrapped_fn: Callable[AsyncFnArgs, Coroutine[Any, Any, asyncio.Task[Res]]]) -> Callable[AsyncFnArgs, Res]:
    """Decorator to transform a start function that returns a task into a function that creates and then finishes the task"""
    async def wrapper(*args: AsyncFnArgs.args, **kwargs: AsyncFnArgs.kwargs) -> Res:
        proc_task = await wrapped_fn(*args, **kwargs)
        return await proc_task    
    return make_sync(wrapper)

# ------------------------ #

def resolve_host(host_spec: Union[str, WorkerContext], no_slurm: bool = False) -> tuple[WorkerContext, HostInfo]:
    """Take something that may be passed as a a host_spec and return the WorkerContext and HostInfo"""
    if isinstance(host_spec, str):
        host_spec = get_network_hosts().get(host_spec, prefer_default_partition=not no_slurm)
    host = host_spec.host
    return host_spec, host



async def launch_command_on_host(command: str, host_spec: Union[str, WorkerContext] = 'localhost',
                                 use_srun: bool = False, no_slurm: bool = False, slurm_opts: str = '', nohup: bool = False,
                                 wait_for_str: Optional[str] = None, existing_connection: Optional[asyncssh.SSHClientConnection] = None
                                 ) -> asyncssh.SSHClientProcess:
    """
    Start an arbitrary command from within the caiman conda environment on remote (or local) host (must be in host_info.py)
    and return a SSHClientProcess as soon as the command is running.
    Automatically uses sbatch if host is a cluster; set use_srun to use srun instead. Specify as {host}/{partition} to use a specific partition.
    Set no_slurm to True to not use slurm even if the host that a string host_spec resolves to has SLURM partitions.
    slurm_opts is a string of options to append to the sbatch or srun command.
    Set nohup to true to continue running in background after disconnection (by default, this occurs only for Windows hosts)
    If wait_for_str is not None, waits for this string to appear anywhere in the output before returning.
    (If nohup is true for unix host, it actually just waits until nohup has started and thus it is safe to disconnect.)
    If existing_connection is given, the connection will be used and not closed after the function completes.
    """
    host_spec, host = resolve_host(host_spec, no_slurm=no_slurm)
    host_env = host.envname
    if host_env is not None:
        command = f'conda run -n {host_env} --live-stream ' + command

    if not host.is_pc:
        if nohup and wait_for_str is not None:
            # command will be run in background, but echo string we're waiting for first
            command = f'echo {wait_for_str} && ' + command

        # run in interactive shell to ensure we have access to conda
        command = 'bash -ic ' + shlex.quote(command)
        
    if isinstance(host_spec, PartitionInfo):
        if '--partition' not in slurm_opts:
            slurm_opts += ' --partition=' + host_spec.name

        if use_srun:
            command = 'srun --pty ' + slurm_opts + ' ' + command
        else:                
            command = 'sbatch --ntasks=1 --wrap=' + host.lex.quote(command) + ' ' + slurm_opts
    
    if nohup and not host.is_pc:
        command = 'nohup ' + command

    if existing_connection is None:
        connection = asyncssh.connect(host.ssh_name, username=host.ssh_user if host.ssh_user is not None else ())
    else:
        connection = nullcontext(existing_connection)
        
    async with connection as conn:
        proc = await conn.create_process(command, stdin=asyncssh.DEVNULL, stderr=asyncssh.STDOUT)
        if wait_for_str is not None:
            # wait until we have output indicating that process has started
            while True:
                line = await proc.stdout.readline()
                if line == '':
                    raise RuntimeError('EOF received before string being waited on')
                elif wait_for_str in line:
                    break
    return proc


async def monitor_process(proc: asyncssh.SSHClientProcess, conn: Optional[asyncssh.SSHClientConnection],
                          output_file: Optional[str] = None, print_output: bool = False, check: bool = True,
                          timeout: Optional[float] = None) -> asyncssh.SSHCompletedProcess:
    """
    Wait for a given process to complete, optional printing the output and/or writing to a local file
    (using either of these to true prevents the returned object from containing the output).
    If the connection is provided, it is closed once the process exits.
    check: Error if the process errors.
    timeout: Wait up to this many seconds before raising a RuntimeError (default = no timeout)
    """
    try:           
        with open(output_file, 'w') if output_file is not None else nullcontext() as f:
            async def process_output() -> asyncssh.SSHCompletedProcess:
                if (f is not None) or print_output:
                    async for line in proc.stdout:
                        if f is not None:
                            try:
                                print(line, end='', file=f, flush=True)
                            except OSError as err:
                                if err.errno == 22:  # Ignore invalid argument error that seems to happen after a while
                                    print('Failed to write to log file (errno 22): ', line, end='', flush=True)
                                else:
                                    raise
                        if print_output:
                            print(line, end='', flush=True)     
                return await proc.wait(check=check)

            try:
                completed_obj = await asyncio.wait_for(process_output(), timeout=timeout)
            except TimeoutError:
                # ran too long, shut down
                print('Timed out, shutting down process...')
                proc.close()
                try:
                    await asyncio.wait_for(proc.wait_closed(), timeout=30.0)
                    raise RuntimeError('Process timed out, closed')
                except TimeoutError:
                    proc.kill()
                    raise RuntimeError('Process timed out, killed')
        return completed_obj
    finally:
        if conn is not None:
            conn.close()


async def start_command_on_host(command: str, host_spec: Union[str, WorkerContext] = 'localhost', use_srun: bool = False,
                                no_slurm: bool = False, slurm_opts: str = '', timeout: Optional[float] = None, check=False,
                                output_file: Optional[str] = None, print_output: bool = False, connection: Optional[asyncssh.SSHClientConnection] = None
                                ) -> asyncio.Task[asyncssh.SSHCompletedProcess]:
    """
    Run arbitrary command from within the caiman conda environment on remote (or local) host (must be in host_info.py)
    Awaits launch_command_on_host and then returns a Task to run monitor_process. See these functions for arguments.
    """
    host_spec, host = resolve_host(host_spec, no_slurm=no_slurm)
    if isinstance(host_spec, PartitionInfo) and not use_srun and output_file is not None:
        raise NotImplementedError('Redirecting to local file not supported with sbatch (try use_srun=True)')
    
    if connection is None:
        connection = await asyncssh.connect(host.ssh_name, username=host.ssh_user if host.ssh_user is not None else ())
        conn_to_close = connection
    else:
        # don't automatically close passed connection
        conn_to_close = None

    proc = await launch_command_on_host(command, host_spec=host_spec, use_srun=use_srun, no_slurm=no_slurm, slurm_opts=slurm_opts, existing_connection=connection)
    monitor_cor = monitor_process(proc, conn=conn_to_close, output_file=output_file, print_output=print_output, check=check, timeout=timeout)
    return asyncio.create_task(monitor_cor)
    

async def start_script_on_host(script_name: str, script_args: Iterable[str] = (), **kwargs) -> asyncio.Task[asyncssh.SSHCompletedProcess]:
    """Launch a script (in the cmcode.script subpackage) on remote (or local) host (must be in host_info.py)"""
    module_path = f'cmcode.remote.script.{script_name}'
    command = f'python -m {module_path} {" ".join(script_args)}'
    return await start_command_on_host(command, **kwargs)


run_command_on_host = finish_task(start_command_on_host)
run_script_on_host = finish_task(start_script_on_host)


def get_string_output(res: asyncssh.SSHCompletedProcess) -> Optional[str]:
    """Try to interpret the output of a process as a string"""
    if isinstance(res.stdout, bytes):
        return res.stdout.decode('utf-8')
    elif res.stdout is not None:
        return str(res.stdout)
    else:
        return None


async def get_output_from_command(command: str, host_spec: Union[str, WorkerContext], timeout: int = 10,
                                  no_slurm: bool = False, slurm_args: str = '', connection: Optional[asyncssh.SSHClientConnection] = None) -> str:
    """Run a command to completion and return the standard output"""
    task = await start_command_on_host(command, host_spec=host_spec, timeout=timeout, check=True, no_slurm=no_slurm,
                                       use_srun=True, slurm_opts=slurm_args, connection=connection)
    res = await task
    output = get_string_output(res)

    if res.returncode != 0:
        raise RuntimeError(f'Command unsuccessful (code {res.returncode})\n' +
                           ('Output:\n' + output) if output is not None else '')

    if output is None:
        raise RuntimeError('Command returned no output (redirected?)')

    return output


async def forward_remote_port(port: int, host_spec: Union[str, WorkerContext, asyncssh.SSHClientConnection],
                              callback: Callable[[asyncssh.SSHListener], Awaitable]) -> None:
    """
    Set up port forwarding from local to remote, like with ssh -NfL
    If host is an SSHClientConnection, reuses this connection.
    Once connection is established, creates a task by calling callback and awaits it, then closes the connection.
    """
    if isinstance(host_spec, asyncssh.SSHClientConnection):
        connection = nullcontext(host_spec)
    else:
        host_spec, host = resolve_host(host_spec)
        connection = asyncssh.connect(host.ssh_name, username=host.ssh_user if host.ssh_user is not None else ())

    async with connection as conn:
        listener = await conn.forward_local_port('localhost', 0, 'localhost', port)
        await callback(listener)


def get_slurm_partition() -> Optional[str]:
    """Helper for ipyparallel engine startup (needs to be accessible from module scope)"""
    return os.environ.get('SLURM_JOB_PARTITION')


async def start_network_cluster_async(host_specs: Iterable[str], max_threads_per_worker: int = 20, n_cores_to_exclude: int = 1,
                                      debug: bool = False, shared_profile_dir=False, connect_timeout: int = 90
                                      ) -> tuple[ipp.Client, str, int, Optional[str]]:
    """
    Initiate a cluster (dview) using ipyparallel and connect to given hosts and/or partitions (including the local one, typically).
    Partitions should be specified as {hostname}/{partition}, where hostname is the lowercase common name used as a key in the host_info dict.
    Information about each host is read from the file host_info.py. Partitions on slurm hosts can be specified as [host]/[partition].
    If max_threads_per_worker > 1, also calls become_dask() to use dask for multithreading on each worker. This should be 
    chosen according to how heavily the GIL is used for the task at hand, e.g. if it's used 5% of the time, there's little benefit to using >20 threads.
    The total number of threads (workers * threads_per_worker) on each host will be no greater than ncores - n_cores_to_exclude.
    If using more than just localhost, shared_profile_dir will be set to True and project.shared_ipython_profile_dir must not be None.
    connect_timout = number of seconds to wait for all clients to connect before moving on (-1 to wait forever)
    Return values: (client, cluster_id, n_threads, [profile_dir])  (if shared_profile_dir is False, last output is None)
    """
    # get information about each host
    network_hosts = get_network_hosts()
    contexts = [network_hosts.get(name) for name in host_specs]
    hosts = [ctx.host for ctx in contexts]
    is_local = [host.is_local for host in hosts]

    # ensure all hosts are the same platform as the controller and therefore share the same paths to data files
    # caiman requires all workers to use the same path to access memmapped files
    # assumption is that the files we will be working with are on shared drives w/ consistent paths for each platform
    localhost = network_hosts.get_localhost()
    on_pc = localhost.is_pc
    if any(host.is_pc != on_pc for host in hosts):
        raise ValueError('Cluster with a mix of Windows and UNIX computers is not supported')

    # get ipyparallel profile dir (must be accessible to each host)
    if not shared_profile_dir and not all(is_local):
        logging.info('Switching to shared profile dir because one or more non-local hosts were requested')
        shared_profile_dir = True

    ipyprofile_dir = paths.get_ipyprofile_dir()
    if shared_profile_dir:
        # get ssh name for the local host, so workers can find the controller
        if localhost.autogenerated:
            logging.warning('Host info for localhost was auto-generated - SSH for workers to access the controller may be incorrect.\n'
                            'To avoid this, network_hosts should be set to a NetworkInfo object that includes the local host.')
        cluster_args = {
            'profile_dir': ipyprofile_dir,
            'controller_ip': '*',
            'controller_args': ['--location=' + localhost.ssh_name]
        }
    else:  # local cluster
        cluster_args = {}

    # start controller
    cluster = ipp.Cluster(**cluster_args)
    await cluster.start_controller()
    cluster_id = cluster.cluster_id

    if shared_profile_dir:
        # make engine and client files group read and writable (since this is hard-coded in ipp)
        for file_type in ['client', 'engine']:
            path = os.path.join(ipyprofile_dir, 'security', f'ipcontroller-{cluster_id}-{file_type}.json')
            user_grp_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
            timeout_dur = 30
            timeout_time = time.time() + timeout_dur
            while not os.path.exists(path) and time.time() < timeout_time:
                await asyncio.sleep(0.2)

            if not os.path.exists(path):
                raise RuntimeError(f'ipyparallel {file_type} file not created in {timeout_dur}')
            os.chmod(path, user_grp_rw)
    
    # start engines on each host
    engines_per_host = []
    threads_per_engine = []
    threads_per_host = []
    full_name_map: dict[tuple[str, Optional[str]], str] = {}  # map host and partition names to full names

    for context, host_info in zip(contexts, hosts):
        # figure out how many workers and threads to use
        max_cores = context.n_cores - n_cores_to_exclude
        full_name = context.get_full_name()

        if isinstance(context, PartitionInfo):
            # don't have to worry about sharing jobs across nodes for ipyparallel,
            # but let's not use threading because with multiple nodes this could be way too complicated
            n_workers = max_cores
            partition = context.name
            for node in context.nodes:
                full_name_map[(node, context.name)] = full_name
        else:
            n_workers = math.ceil(max_cores / max_threads_per_worker)
            partition = None
            full_name_map[(host_info.self_name, None)] = full_name
            
        threads_per_worker = max_cores // n_workers
        total_threads = n_workers * threads_per_worker

        if host_info.is_local:
            logging.info('Starting local engines...')
            log_level = logging.DEBUG if debug else logging.INFO
            if partition is not None:
                cluster.engine_launcher_class = 'slurm'
                cluster.start_engines_sync(n=n_workers, queue=partition, log_level=log_level)
            else:
                cluster.engine_launcher_class = 'local'
                cluster.start_engines_sync(n=n_workers, log_level=log_level)
            logging.info(f'Started {n_workers} local engines.')
        else:
            profile_dir_quoted: str = host_info.lex.quote(ipyprofile_dir)
            engine_launch_cmd = f"ipengine --profile-dir={profile_dir_quoted} --cluster-id={cluster_id}"
            if debug:
                engine_launch_cmd += ' --debug'

            # settings to avoid getting rejected by SSH server
            max_sessions_per_connection = 10  # SSH server default
            per_connection_wait_time = 0.1

            async def launch_worker(conn: asyncssh.SSHClientConnection, output_file: Optional[str] = None):
                """Start worker over SSH, being sure the process has started before disconnecting"""
                proc = await launch_command_on_host(engine_launch_cmd, host_spec=context, nohup=True, wait_for_str='IPEngine',
                                                    use_srun=True, existing_connection=conn)
                # for debugging
                if output_file is not None:
                    await proc.redirect(stdout=output_file, stderr=asyncssh.STDOUT)
                    await asyncio.sleep(20)

            async def launch_n_workers(n: int, start_time: float, first_worker_output_file: Optional[str] = None):
                """launch workers over SSH in a loop, but not too quickly"""
                await asyncio.sleep(start_time - time.time())
                async with asyncssh.connect(host_info.ssh_name, username=host_info.ssh_user if host_info.ssh_user is not None else ()) as conn:
                    await asyncio.gather(*(launch_worker(conn, output_file=first_worker_output_file if i == 0 else None) for i in range(n)))

            # divide workers evenly between connections
            n_connections_needed = math.ceil(n_workers / max_sessions_per_connection)
            workers_per_conn = [n_workers // n_connections_needed] * n_connections_needed
            for i in range(n_workers % n_connections_needed):
                workers_per_conn[i] += 1

            logging.info(f'Starting {n_workers} engines on {full_name}...')
            t0 = time.time()
            start_times = [t0 + per_connection_wait_time * i for i in range(n_connections_needed)]
            output_files: list[Optional[str]] = [None] * len(workers_per_conn)
            if debug:
                output_filename = context.get_full_name(separator='_') + '_ipengine.log'
                output_files[0] = os.path.join(os.path.expanduser('~'), output_filename)
                logging.info('Saving output of first engine launch to ' + output_files[0])

            await asyncio.gather(*(launch_n_workers(n, t, first_worker_output_file=f) for n, t, f in zip(workers_per_conn, start_times, output_files)))
                        
        engines_per_host.append(n_workers)
        threads_per_engine.append(threads_per_worker)
        threads_per_host.append(total_threads)

    # wait for clients to connect
    total_engines = sum(engines_per_host)
    logging.info(f'Waiting for {total_engines} workers to connect...')
    rc = await cluster.connect_client()
    try:
        rc.wait_for_engines(total_engines, timeout=connect_timeout)
    except TimeoutError:
        logging.warning(f'Not all requested engines connected within {connect_timeout} seconds.')
    
    # report how many connections were actually made
    # more may continue to connect, but just use the number that have connected so far (to avoid race conditions)
    n = len(rc.ids)
    engine_hosts = rc[:n].apply_sync(socket.gethostname)
    engine_partitions = rc[:n].apply_sync(get_slurm_partition)

    # transform to full names
    engine_full_names = [full_name_map[(host, partition)] for host, partition in zip(engine_hosts, engine_partitions)]

    # make a map from host/partition to engine ids
    engine_map: dict[str, list[int]] = {}
    for ind, name in enumerate(engine_full_names):
        if name not in engine_map:
            engine_map[name] = []
        engine_map[name].append(rc.ids[ind])
    
    actual_threads = 0
    logging.info('Connected engines:')
    for host_or_partition, n_thr, n_thr_total in zip(contexts, threads_per_engine, threads_per_host):
        name = host_or_partition.get_full_name()
        engine_ids = engine_map.get(name, [])
        logging.info(f'\t{name}: {len(engine_ids)}')

        # start threads on each engine, if needed
        if n_thr > 1 and len(engine_ids) > 0:
            logging.info(f'\t\tStarting {n_thr_total} threads on {name}')
            rc.become_dask(engine_ids, nthreads=n_thr)  # type: ignore
        actual_threads += n_thr * len(engine_ids)

    logging.info(f'Finished initializing cluster with {actual_threads} total threads.')
    profile_dir = cluster_args.get('profile_dir')
    return rc, cluster_id, actual_threads, profile_dir
    
start_network_cluster = make_sync(start_network_cluster_async)


async def stop_network_cluster_async(cluster_id: str, shared_profile_dir: Optional[str] = None):
    """Stop a cluster started by start_network_cluster. By default, stops all."""
    cluster = ipp.Cluster.from_file(profile_dir=shared_profile_dir, cluster_id=cluster_id)
    await cluster.stop_cluster()

stop_network_cluster = make_sync(stop_network_cluster_async)