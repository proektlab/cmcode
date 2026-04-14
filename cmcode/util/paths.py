"""Path utilities"""
from contextlib import contextmanager
from datetime import datetime
import logging
import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import re
import sys
from string import Template
from typing import (Optional, Protocol, Union, Literal, Sequence, Callable, Generic, Generator,
                    Iterable, TypeVar, ParamSpec, Concatenate, runtime_checkable, overload)

import numpy as np

from cmcode.remote.host_info import  HostInfo, get_network_hosts

# Global variables, getters and setters
_root_data_dir: Optional[Path] = None
_ipyprofile_dir: str = ''
_root_mappings: Iterable['EquivalentPaths'] = ()
_caiman_data_dir: Optional[Path] = None


def set_root_data_dir(path: Union[str, Path]):
    """Set to the directory above 'raw' and 'processed'"""
    global _root_data_dir
    path = Path(path)
    if path == _root_data_dir:
        return

    _root_data_dir = path
    
    # also update root dir for mesmerize-core
    import mesmerize_core as mc  # importing locally for now to avoid import order issue with caiman
    mc.set_parent_raw_data_path(path / 'processed')

def get_root_data_dir() -> Path:
    """Get the directory containing all data (above 'raw' and 'processed')"""
    if _root_data_dir is None:
        raise RuntimeError('Root data path has not been set; set using set_root_data_path')
    return _root_data_dir

def set_ipyprofile_dir(path: Union[str, Path, None]):
    """Set directory to be used for ipyparallel communication"""
    global _ipyprofile_dir
    if path is None:
        path = ''
    _ipyprofile_dir = str(path)

def get_ipyprofile_dir() -> str:
    return _ipyprofile_dir

def set_root_mappings(mappings: Iterable['EquivalentPaths']):
    global _root_mappings
    _root_mappings = mappings

def get_root_mappings() -> Iterable['EquivalentPaths']:
    return _root_mappings

def set_caiman_data_dir(path: Union[str, Path, None]):
    global _caiman_data_dir
    path = path if path is None else Path(path) 
    if path == _caiman_data_dir:
        return

    if 'caiman' in sys.modules:
        # for now there is an import order issue
        raise RuntimeError('Cannot set caiman_data_dir; caiman already loaded')
    
    _caiman_data_dir = path
    if path is None:
        if 'CAIMAN_DATA' in os.environ:
            del os.environ['CAIMAN_DATA']
    else:
        os.environ['CAIMAN_DATA'] = str(path)


# ------ equivalent path handling -------#


Self = TypeVar('Self')
class EquivalentPaths:
    """
    Represents a set of absolute paths that should be treated as equivalent,
    along with an arbitrary function from HostInfo to path to use for that host.
    This is an abstract class; use EquivalentPathsByPlatform for the most common case.
    """
    def __init__(self, windows_paths: Iterable[Union[str, PureWindowsPath]] = (),
                 posix_paths: Iterable[Union[str, PurePosixPath]] = ()):
        self.windows_paths: list[PureWindowsPath] = []
        self.posix_paths: list[PurePosixPath] = []

        for path_or_str in windows_paths:
            path = PureWindowsPath(path_or_str)
            if not path.is_absolute():
                raise ValueError('EquivalentPaths takes only absolute paths')
            self.windows_paths.append(path)
        
        for path_or_str in posix_paths:
            path = PurePosixPath(path_or_str)
            if not path.is_absolute():
                raise ValueError('EquivalentPaths takes only absolute paths')
            self.posix_paths.append(path)

    @property
    def paths(self) -> Sequence[PurePath]:
        return list(self.windows_paths) + list(self.posix_paths)

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(repr(p) for p in self.paths) + ')'


    def get_path_for_host(self, host: HostInfo) -> PurePath:
        """Override to implement the host-to-path mapping function"""
        ...

    def get_local_path(self) -> Path:
        return Path(self.get_path_for_host(get_network_hosts().get_localhost()))

    def try_split(self, path: Union[str, PurePath], posix_relpath=True) -> Optional[tuple[PurePath, PurePath]]:
        """
        See if the input path is an absolute path that is relative to any of our equivalent base paths.
        If so, return (base_path, relative_to_base); if not, return None.
        If posix_relpath is true (the default), the relative path is always returned as a PurePosixPath.
        """
        if isinstance(path, str):
            # try interpreting as either a Windows or a Posix path
            for PathClass in (PurePosixPath, PureWindowsPath):
                if (res := self.try_split(PathClass(path), posix_relpath=posix_relpath)) is not None:
                    return res
            return None

        paths_to_try = self.windows_paths if isinstance(path, PureWindowsPath) else self.posix_paths
        for root_path in paths_to_try:
            if path.is_relative_to(root_path):
                relpath = path.relative_to(root_path)
                if posix_relpath:
                    relpath = PurePosixPath(relpath)
                return root_path, relpath
        return None
    
    def try_map_path_to_host(self, path: Union[str, PurePath], host: Optional[HostInfo]) -> Optional[PurePath]:
        """
        See whether the given path is an absolute path relative to one of these equivalent paths,
        and if so, map it to the canonical path for the given host.
        If not, return None.
        If host is none, use localhost.
        """
        if (parts := self.try_split(path, posix_relpath=False)) is not None:
            relpath = parts[1]
            if host is None:
                return self.get_local_path() / relpath
            else:
                return self.get_path_for_host(host) / relpath
        return None


class EquivalentPathsByPlatform(EquivalentPaths):
    """
    Represents a set of absolute paths that should be treated as equivalent,
    where the path to use for a given host is determined by its platform (windows vs. posix).
    The first Windows and first POSIX paths input to the constructor are taken as the canonical ones.
    """
    def get_path_for_platform(self, platform: Literal['windows', 'posix']) -> PurePath:
        if platform == 'windows':
            if len(self.windows_paths) == 0:
                raise RuntimeError(f'No Windows equivalent for {self}')
            return self.windows_paths[0]
        elif platform == 'posix':
            if len(self.posix_paths) == 0:
                raise RuntimeError(f'No POSIX equivalent for {self}')
            return self.posix_paths[0]
        else:
            raise ValueError('platform must be "windows" or "posix"')

    def get_path_for_host(self, host: HostInfo) -> PurePath:
        return self.get_path_for_platform('windows' if host.is_pc else 'posix')
    
    def try_map_path_to_platform(self, path: Union[str, PurePath], platform: Literal['windows', 'posix']) -> Optional[PurePath]:
        """
        See whether the given path is an absolute path relative to one of these equivalent paths,
        and if so, map it to the canonical path for the given platform.
        If not, return None.
        """
        if (parts := self.try_split(path, posix_relpath=False)) is not None:
            relpath = parts[1]
            return self.get_path_for_platform(platform) / relpath
        return None


# the below is some code that allows applying a couple functions robustly to paths or objects that contain paths 
PathMappable = Union[None, str, PurePath, np.bytes_, 'CustomPathMappable', list['PathMappable']]
P = ParamSpec('P')
class PathMapper(Generic[P], Protocol):
    """Function that can normalize a path or list of paths"""
    @overload
    def __call__(self, obj: None, *args: P.args, **kwargs: P.kwargs) -> None: ...

    @overload
    def __call__(self, obj: Union[str, np.bytes_], *args: P.args, **kwargs: P.kwargs) -> str: ...

    @overload
    def __call__(self, obj: PurePath, *args: P.args, **kwargs: P.kwargs) -> PurePath: ...

    PM = TypeVar('PM', bound=PathMappable)
    @overload
    def __call__(self, obj: list[PM], *args: P.args, **kwargs: P.kwargs) -> list[PM]: ...

    N = TypeVar('N', bound='CustomPathMappable')
    @overload
    def __call__(self, obj: N, *args: P.args, **kwargs: P.kwargs) -> N: ...

    def __call__(self, obj: PathMappable, *args: P.args, **kwargs: P.kwargs) -> PathMappable: ...


@runtime_checkable
class CustomPathMappable(Protocol):
    """Object whose paths can be normalized"""
    def apply_path_mapper(self: Self, path_mapper: PathMapper[P], *args: P.args, **kwargs: P.kwargs) -> Self:
        """Fix paths within object for current host"""
        ...


def accept_any_path_mappable(wrapped_fn: Callable[Concatenate[Union[str, PurePath], P], PurePath]) -> PathMapper[P]:
    """
    Decorator that converts a function that takes a string and retuns
    another string to one that also handles None and nested lists of items
    that may themselves be None or lists.
    """
    class MapperWrapper(PathMapper[P]):
        def __call__(self, obj: PathMappable, *args: P.args, **kwargs: P.kwargs) -> PathMappable:
            if obj is None:
                return None
            
            if isinstance(obj, list):
                return [self(p, *args, **kwargs) for p in obj]
            
            if isinstance(obj, np.bytes_):
                # convert to str first
                obj = obj.decode('utf-8')
            
            if isinstance(obj, str):
                return str(wrapped_fn(obj, *args, **kwargs))

            if isinstance(obj, PurePath):
                return wrapped_fn(obj, *args, **kwargs)
            
            else:  # CustomNormalizable
                return obj.apply_path_mapper(self, *args, **kwargs)
    return MapperWrapper()


@accept_any_path_mappable
def normalize_path(path: Union[str, PurePath], for_host: Optional[HostInfo] = None) -> PurePath:
    """Convert given path to work on given host, or localhost if for_host is None"""
    # if absolute, try mapping according to root_mappings
    if isinstance(path, str) or path.is_absolute():
        root_mappings = get_root_mappings()
        for root_mapping in root_mappings:
            if (mapped_path := root_mapping.try_map_path_to_host(path, for_host)) is not None:
                return mapped_path
    
    #  otherwise just convert to platform and make relative paths branch from the root data dir
    if for_host is None:
        platform_path = Path(path)    
    elif for_host.is_pc:
        platform_path = PureWindowsPath(path)
    else:
        platform_path = PurePosixPath(path)
    
    return get_root_data_dir() / platform_path


@accept_any_path_mappable
def relativize_path(path: Union[str, PurePath]) -> PurePath:
    """
    Given a relative path or absolute path that is relative to the root data dir,
    convert to a Posix-style relative path.
    Emits a warning if the path absolute and is not relative to the root data dir.
    """
    path_here = Path(path)
    if path_here.is_absolute():
        root_path = get_root_data_dir()
        if not path_here.is_relative_to(root_path):
            logging.warning(f'Cannot relativize {path} - not relative to root path. Keeping as absolute.')
            return path_here
        else:
            path_here = path_here.relative_to(root_path) 
    return PurePosixPath(path_here)


def get_raw_dir(mouse_id: Union[int, str], rec_type: str = 'learning_ppc') -> str:
    """Get dir with raw recordings for this mouse/rec type"""
    return str(get_root_data_dir() / 'raw' / rec_type / str(mouse_id))


def get_processed_dir(mouse_id: Union[int, str], rec_type: str = 'learning_ppc',
                      create_if_not_found=False) -> str:
    """Get dir (and optionally create) with processed 2P data for this mouse/rec type"""
    processed_dir = get_root_data_dir() / 'processed' / rec_type / str(mouse_id)
    if create_if_not_found:
        os.makedirs(processed_dir, exist_ok=True)
    return str(processed_dir)


class PctTemplate(Template):
    """String template that uses the % symbol instead of $, so that re escaping doesn't affect it"""
    delimiter = '%'
    idpattern = r'(?a:[a-z][a-z0-9]*)'  # don't allow underscores as part of identifiers


def make_timestamped_filename(file_pattern: str) -> str:
    """
    Given a filename with a format like the following, fill the slot in with the current timestamp:
    'example_file_%dt.npy' -> 'example_file_2025-07-18_17-14-00.npy'
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return PctTemplate(file_pattern).substitute({'dt': timestamp})


def get_all_timestamped_files(parent_dir: Union[str, Path], file_pattern: str) -> list[str]:
    """
    Find files within parent_dir matching a pattern with timestamps
    included in the filename and return the full paths,
    sorted from newest to oldest timestamp.
        - file_pattern: string with "%dt" or "%{dt}" where the date should go.
    """
    file_pattern_escaped = re.escape(file_pattern)
    dt_re = r'(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)'
    file_re = PctTemplate(file_pattern_escaped).substitute({'dt': dt_re}) + '$'

    # find files in the dir that match and sort by date
    try:
        all_files = os.listdir(parent_dir)
    except FileNotFoundError:
        return []
    match_objs = [re.match(file_re, f) for f in all_files]
    matches = sorted(filter(None, match_objs), key=lambda m: m[1], reverse=True)
    return [os.path.join(parent_dir, match[0]) for match in matches]


def get_latest_timestamped_file(parent_dir: Union[str, Path], file_pattern: str) -> Optional[str]:
    """
    Find the file within parent_dir matching a pattern with the latest timestamp
    (where the timestamp is part of the filename, not the modified time)
    and return the full path (or None if none were found).
        - file_pattern: string with "%dt" or "%{dt}" where the date should go.
    """
    all_files = get_all_timestamped_files(parent_dir, file_pattern)
    return all_files[0] if len(all_files) > 0 else None


def params_file_for_result(result_file: Union[str, Path]) -> str:
    return os.path.splitext(result_file)[0] + '_params.json'


def add_timestamp_to_path(path: str) -> str:
    """Add the current timestamp to an existing path right before the extension"""
    start, ext = os.path.splitext(path)
    path_template = start + '_%dt' + ext
    dir, name_template = os.path.split(path_template)
    link_path = os.path.join(dir, make_timestamped_filename(name_template))
    return link_path


@contextmanager
def linked_timestamped_path(path: str) -> Generator[str, None, None]:
    """
    Hard-link a version of the path with the timestamp added and delete the link
    when the context manager exits. This is useful for some caiman function(s)
    (motion correction) that determine the output path based on the input.
    """
    link_path = add_timestamp_to_path(path)
    os.link(src=path, dst=link_path)
    yield link_path
    os.unlink(link_path)
