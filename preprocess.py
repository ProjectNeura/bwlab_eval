from nibabel import load as _load
from os import listdir as _listdir, mkdir as _mkdir
from shutil import copyfile as _copyfile, rmtree as _rmtree
from functools import reduce as _reduce
from operator import mul as _mul


def sort(src: str) -> list[str]:
    r = _listdir(src)
    r.sort(key=lambda x: _reduce(_mul, _load(f"{src}/{x}").shape))
    return r


def select_samples(src: str, dst: str) -> dict[str, str]:
    samples = sort(src)[:10]
    r = {}
    i = 0
    for file in samples:
        mapped_name = f"case{str(i).zfill(4)}_0000.nii.gz"
        r[mapped_name] = file
        _copyfile(f"{src}/{file}", f"{dst}/{mapped_name}")
        i += 1
    return r


def clear_cache(src: str) -> None:
    _rmtree(src)
    _mkdir(src)
