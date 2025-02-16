from argparse import ArgumentParser as _ArgumentParser
from os import listdir as _listdir

from numpy import ndarray as _ndarray, NaN as _NaN


def compute_dice_coefficient(pred: _ndarray[bool], gt: _ndarray[bool]) -> float | _NaN:
    volume_sum = gt.sum() + pred.sum()
    if volume_sum == 0:
        return _NaN
    volume_intersect = (gt & pred).sum()
    return 2 * volume_intersect / volume_sum


def __entry__() -> None:
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("pred_path")
    parser.add_argument("gt_path")
    parser.add_argument("save_path")
    args = parser.parse_args()
    filenames = _listdir(args.pred_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()


if __name__ == "__main__":
    __entry__()
