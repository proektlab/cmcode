from typing import Optional, Sequence

def make_sess_name(sess_id: int, tag: Optional[str]) -> str:
    return f'{sess_id}{"_" + tag if tag else ""}'

def make_sess_names(sess_ids: Sequence[int], tags: Sequence[Optional[str]]) -> list[str]:
    return [make_sess_name(sess_id, tag) for sess_id, tag in zip(sess_ids, tags)]