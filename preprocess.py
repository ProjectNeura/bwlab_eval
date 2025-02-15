from nibabel import load as _load
from os import listdir as _listdir
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


if __name__ == "__main__":
    select_samples("/workspace/data/nnUNet_raw/Dataset702_AbdomenMR-20250215T202943Z-001/Dataset702_AbdomenMR/imagesTs",
                   "/workspace/data/nnUNet_eval_output")
