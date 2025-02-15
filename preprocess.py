from nibabel import load as _load
from os import listdir as _listdir, removedirs as _removedirs
from shutil import copyfile as _copyfile
from functools import reduce as _reduce
from operator import mul as _mul


def sort(src: str) -> list[str]:
    r = _listdir(src)
    r.sort(key=lambda x: _reduce(_mul, _load(f"{src}/{x}").shape))
    return r


def select_samples(src: str, dst: str) -> None:
    samples = sort(src)[:10]
    for file in samples:
        _copyfile(f"{src}/{file}", f"{dst}/{file}")


def clear_cache(src: str) -> None:
    _removedirs(src)
