"""General setup for Ethan's Proekt Lab code"""
import builtins
import logging
from typing import Union
import nest_asyncio

from cmcode.util.environment import ComputingEnvironment

try:
    from cmcode.private.local_environment import computing_environment
except ImportError:
    pass
else:
    assert isinstance(computing_environment, ComputingEnvironment), \
        'computing_environment should be an instance of ComputingEnvironment'
    computing_environment.apply()


def setup_logging(log_level: Union[int, str], force: bool = True):
    """
    Enable basic logging to console
    If force is true (default), overrides any previous call to basicConfig
    log_level can be DEBUG, INFO, WARNING, ERROR, or CRITICAL
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    logging.basicConfig(
        format="{asctime} - {levelname} - [{filename} {funcName}() {lineno}] - pid {process} - {message}",
        filename=None, force=force,
        level=log_level, style="{")
    
    # add specific filter for noisy motion correction log messages
    def filter_mc_logs(record: logging.LogRecord):
        if record.funcName == 'tile_and_correct' and record.getMessage() in ['Extracting patches', 'extracting shifts for each patch']:
            return False
        return True

    caiman_logger = logging.getLogger('caiman')
    caiman_logger.addFilter(filter_mc_logs)
    
    # override asyncssh logging level - maybe todo do this more idiomatically
    ssh_logger = logging.getLogger('asyncssh')
    ssh_logger.setLevel(logging.WARNING)

setup_logging('INFO')

def in_jupyter() -> bool:
    return getattr(builtins, "__IPYTHON__", False) != False

if in_jupyter():
    # allow asyncio.run() to work in notebooks
    nest_asyncio.apply()
