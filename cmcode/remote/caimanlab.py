"""Launch notebooks for CaImAn analysis, with backend on remote server"""
import argparse
import asyncio
import asyncssh
import logging
import os
from pathlib import PurePosixPath, PureWindowsPath, PurePath
import sys
import tempfile
import time
from typing import Optional, Union
from urllib.parse import urlparse, ParseResult
import webbrowser

from cmcode import datadir_path_win, datadir_path_unix, setup_logging
from cmcode.remote import remoteops
from cmcode.remote.host_info import WorkerContext, HostInfo
from cmcode.remote.private.local_host_info import network_hosts as network


async def get_path_from_host(host_spec: Union[str, WorkerContext], python_code: str, timeout: int = 45,
                             connection: Optional[asyncssh.SSHClientConnection] = None) -> PurePath:
    """
    Query the remote (or local) host to get a path from a python command.
    python_code: a one-liner that runs in the CaImAn conda environment and prints the path of interest.
    timeout: seconds to wait before giving up
    """
    if isinstance(host_spec, str):
        host_spec = network.get(host_spec)
    host_info = host_spec.host

    command = 'python -c ' + host_info.lex.quote(python_code)
    output = await remoteops.get_output_from_command(command, host_spec, timeout=timeout, no_slurm=True, connection=connection)

    HostPath = PureWindowsPath if host_info.is_pc else PurePosixPath
    # path should be first line of output and ignore any output from SLURM
    for line in reversed(output.splitlines()):
        if line != '' and not line.startswith('srun:'):
            return HostPath(line)
    raise RuntimeError('Command did not return an output')


async def get_jupyterlab_servers(host_spec: Union[str, WorkerContext], notebook_path: PurePath, timeout: int = 45,
                                 no_slurm: bool = True, slurm_args: str = '',
                                 connection: Optional[asyncssh.SSHClientConnection] = None) -> set[ParseResult]:
    """Get a set of urls for existing jupyter-lab servers with given notebook path"""
    if isinstance(host_spec, str):
        host_spec = network.get(host_spec)

    output = await remoteops.get_output_from_command('jupyter-lab list', host_spec, timeout=timeout, no_slurm=no_slurm,
                                                     slurm_args=slurm_args, connection=connection)
    logging.info(f'jupyter-lab list output:\n' + output)

    server_entries = [l.split(' :: ') for l in output.splitlines() if ' :: ' in l]
    server_entries = [p for p in server_entries if len(p) == 2]  # filter out invalid lines
    server_urls = [url for (url, path) in server_entries if type(notebook_path)(path) == notebook_path]
    return {urlparse(url) for url in server_urls}


async def check_for_exit(jupyter_task: asyncio.Task[asyncssh.SSHCompletedProcess]) -> bool:
    """Check whether the Jupyter process is dead"""
    try:
        res = await asyncio.wait_for(asyncio.shield(jupyter_task), timeout=1.0)
        if res.returncode == 0:
            return True

        # exited abnormally
        output = remoteops.get_string_output(res)
        raise RuntimeError(f'Jupyter exited early (code {res.returncode})\n' +
                           ('Output:\n' + output) if output is not None else '')
    except asyncio.exceptions.TimeoutError:
        return False


async def launch_new_server(host_spec: WorkerContext, host_info: HostInfo, notebook_path: PurePath, launch_timeout: int = 60,
                            idle_timeout: Optional[int] = None, no_slurm: bool = True, slurm_args: str = '', show_output: bool = False,
                            connection: Optional[asyncssh.SSHClientConnection] = None) -> tuple[ParseResult, asyncio.Task[asyncssh.SSHCompletedProcess]]:
    """
    Start a new JupyterLab instance on given host. See launch for more info.
    Returns server_url, server_connection
    no_slurm: force running without SLURM even if the remote host is a cluster.
    """
    jupyter_cmd =  'jupyter-lab --no-browser '\
                   '--notebook-dir=' + host_info.lex.quote(str(notebook_path))
    
    if idle_timeout is not None:
        idle_timeout = int(idle_timeout)  # for safety
        jupyter_cmd += (f' --ServerApp.shutdown_no_activity_timeout={idle_timeout}')

    # save Jupyter log to temp file
    file, log_path = tempfile.mkstemp(suffix='.log', prefix='jupyterlab_', text=True)
    os.close(file)
    print(f'Jupyter output will be logged to {log_path}.')

    # launch jupyter-lab and let it run in the background because the timeout should kill it if idle
    proc_task = None
    server = None
    ntries = 0
    max_tries = 3
    try:
        while server is None and ntries < max_tries:
            if ntries > 0:
                logging.warning('Trying again...')
            ntries += 1
            existing_servers = await get_jupyterlab_servers(host_spec, notebook_path, no_slurm=no_slurm, slurm_args=slurm_args,
                                                            connection=connection)
            proc_task = await remoteops.start_command_on_host(jupyter_cmd, host_spec=host_spec, no_slurm=no_slurm, use_srun=True,
                                                              output_file=log_path, print_output=show_output, connection=connection)

            #  find the URL of newly-created jupyter lab server
            waitstart = time.time()
            new_servers: set[ParseResult] = set()
            while len(new_servers) == 0 and time.time() - waitstart < launch_timeout:
                if await check_for_exit(proc_task):
                    raise RuntimeError('Jupyter server exited before connection could be established')
                curr_servers = await get_jupyterlab_servers(host_spec, notebook_path, no_slurm=no_slurm, slurm_args=slurm_args,
                                                            connection=connection)
                new_servers = curr_servers - existing_servers  # should be indifferent to another server closing
                existing_servers = curr_servers
                await asyncio.sleep(5)

            if len(new_servers) == 1:
                server = new_servers.pop()
            elif len(new_servers) == 0:
                proc_task.cancel()
                logging.warning(f'Timed out while waiting for server to launch ({launch_timeout} seconds)')
            else:
                proc_task.cancel()
                logging.warning('Could not determine which server was launched')
        
        if server is None:
            raise RuntimeError(f'Could not start Jupyter server after {max_tries} tries')
    except:
        if proc_task is not None:
            proc_task.cancel()        
        raise
    
    assert proc_task is not None, 'Jupyter process should exist at this point'
    return server, proc_task


async def launch(host_spec: Union[str, WorkerContext], force_new: bool = False, launch_timeout: int = 60,
                 idle_timeout: Optional[int] = None, no_slurm: bool = True, slurm_args: Optional[str] = None,
                 show_output: bool = False, root_path: Optional[str] = None):
    """
    Start JupyterLab on the given host, or find existing instance, and launch a web browser locally to connect to it.
    The following options only take effect for new servers. If they are required (for example, partition), it may
    be safer to use force_new to force the creation of a new server.
        server_launch_timeout: Time in seconds to wait for the server to start up
        server_idle_timeout: Time in seconds before server shuts down if nothing is running on it.
        slurm_args: arguments to sbatch (ignored if host is not a cluster)
    """
    if slurm_args is None:
        slurm_args = ''

    if isinstance(host_spec, str):
        host_spec = network.get(host_spec)
    host_info = host_spec.host

    if root_path is not None:
        root_purepath = PureWindowsPath(root_path) if host_info.is_pc else PurePosixPath(root_path)
    else:
        root_purepath = datadir_path_win if host_info.is_pc else datadir_path_unix

    # look for an existing server
    async with asyncssh.connect(host_info.ssh_name, username=host_info.ssh_user if host_info.ssh_user is not None else ()) as conn:
        existing_servers = await get_jupyterlab_servers(host_spec, root_purepath, no_slurm=no_slurm, slurm_args=slurm_args, connection=conn)
        if not force_new and len(existing_servers) > 0:
            server_url = existing_servers.pop()
            logging.info(f'Found existing server with remote URL {server_url.geturl()}')
            jupyter_task = None
        else:
            server_url, jupyter_task = await launch_new_server(host_spec, host_info, root_purepath, launch_timeout=launch_timeout, idle_timeout=idle_timeout,
                                                               no_slurm=no_slurm, slurm_args=slurm_args, show_output=show_output, connection=conn)
            if await check_for_exit(jupyter_task):
                raise RuntimeError('Jupyter exited before connection could be established')
            logging.info(f'Created new server with remote URL {server_url.geturl()}')
            
        remote_port = server_url.port
        assert remote_port is not None, 'Jupyter URL should have a port'

        async def run_jupyterlab_client(listener: Optional[asyncssh.SSHListener]) -> None:
            """Open browser to jupyterlab, then wait for server to close"""
            if listener is None:
                # server is running locally
                local_url = server_url
            else:
                remote_host = server_url.hostname
                assert remote_host is not None, 'Jupyter URL should have a hostname'
                
                # Get the local URL by substituting in the local port number
                local_port = listener.get_port()
                new_netloc = server_url.netloc.replace(f':{remote_port}', f':{local_port}')
                local_url = server_url._replace(netloc=new_netloc)
                print(f'Access locally at {local_url.geturl()}')
                
            webbrowser.open_new_tab(local_url.geturl())
            
            try:
                # loop forever until killed
                while True:                    
                    if jupyter_task is not None:
                        if await check_for_exit(jupyter_task):
                            break
                    else:
                        await asyncio.sleep(1)
            finally:
                # terminate jupyter
                if jupyter_task is not None:
                    print('Shutting down JupyterLab')
                    jupyter_task.cancel()
                
        if host_info.is_local and isinstance(host_spec, HostInfo):  # not running on SLURM
            # no forwarding needed
            await run_jupyterlab_client(None)
        else:
            await remoteops.forward_remote_port(remote_port, conn, run_jupyterlab_client)


def handle_args(args: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description='Tool to manage JupyterLab for CaImAn')
    parser.add_argument('host_spec', default='localhost')
    parser.add_argument('-f', '--force-new', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--launch-timeout')
    parser.add_argument('--idle-timeout')
    parser.add_argument('-s', '--slurm-args')
    parser.add_argument('-l', '--log-level', default='WARNING')
    parser.add_argument('--slurm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--show-output', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-p', '--root-path')
    parse_res = parser.parse_args(args)
    return {k: v for k, v in vars(parse_res).items() if v is not None}


def main(args: Optional[list[str]] = None):
    opts = handle_args(args)
    try:
        levelstr: str = opts['log_level']
        log_level = getattr(logging, levelstr.upper())
    except AttributeError:
        print(f'Invalid logging level {levelstr}', file=sys.stderr)
        sys.exit(1)
    del opts['log_level']

    opts['no_slurm'] = not opts['slurm']
    del opts['slurm']

    setup_logging(log_level)
    asyncio.run(launch(**opts))
