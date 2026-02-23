import re
from typing import Optional, Sequence, Union, Iterable

import numpy as np


def make_sess_name(sess_id: Union[int, np.integer], tag: Optional[str], underscore=True, zero_padded=False) -> str:
    """Create session name in given format"""
    sess_str = f'{sess_id:03d}' if zero_padded else str(sess_id)
    tag_str = ('_' if underscore else '') + tag if tag else ''
    return sess_str + tag_str

def make_sess_names(
    sess_ids: Iterable[Union[str, int, np.integer]], tags: Optional[Sequence[Optional[str]]] = None,
    underscore=True, zero_padded=False) -> list[str]:

    if tags is None:
        tags = [None for _ in sess_ids]

    sess_names: list[str] = []
    for sess_id, tag in zip(sess_ids, tags):
        if isinstance(sess_id, (int, np.integer)):
            sess_names.append(make_sess_name(sess_id, tag, underscore=underscore, zero_padded=zero_padded))
        else:
            if tag is not None:
                raise ValueError('Cannot pass both string sess_id and tag')
            sess_names.append(sess_id)
    return sess_names


def split_sess_name(sess_name: str) -> tuple[int, Optional[str]]:
    """Split a sessions name (whether or not it contains an underscore)"""
    # regex matches the sess_id and optionally a tag (which must not start with a digit or underscore)
    match = re.match(r'(\d+)_?([^\d_].*)?', sess_name)
    if match is None:
        raise ValueError(f'{sess_name} is not a a valid session name')
    return int(match[1]), match[2]

def split_sess_names(sess_names: Sequence[Union[int, str]]) -> tuple[list[int], list[Optional[str]]]:
    """Split a sequence that may contain both numerical IDs and strings"""
    sess_ids: list[int] = []
    tags: list[Optional[str]] = []

    for name in sess_names:
        if isinstance(name, int):
            sess_ids.append(name)
            tags.append(None)
        else:
            sess_id, tag = split_sess_name(name)
            sess_ids.append(sess_id)
            tags.append(tag)
    
    return sess_ids, tags

def format_sess_name(sess_name: str, underscore=True, zero_padded=False) -> str:
    """Convert session name to given format"""
    sess_id, tag = split_sess_name(sess_name)
    return make_sess_name(sess_id, tag, underscore=underscore, zero_padded=zero_padded)