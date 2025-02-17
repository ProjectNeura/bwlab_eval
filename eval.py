from argparse import ArgumentParser as _ArgumentParser
from os import listdir as _listdir
from nibabel import load as _load
from collections import OrderedDict as _OrderedDict

from numpy import ndarray as _ndarray, NaN as _NaN, signedinteger as _signedinteger, uint8 as _uint8, sum as _sum, \
    max as _max, where as _where, min as _min
from pandas import DataFrame as _DataFrame


def compute_dice_coefficient(pred: _ndarray[bool], gt: _ndarray[bool]) -> float:
    volume_sum = gt.sum() + pred.sum()
    if volume_sum == 0:
        return _NaN
    volume_intersect = (gt & pred).sum()
    return 2 * volume_intersect / volume_sum


def find_lower_upper_zbound(organ_mask: _ndarray[bool]) -> tuple[_signedinteger, _signedinteger]:
    organ_mask = _uint8(organ_mask)
    assert _max(organ_mask) == 1, print('mask label error!')
    z_index = _where(organ_mask > 0)[2]
    return _min(z_index), _max(z_index)


LABELS: list[str] = ["liver", "right-kidney", "spleen", "pancreas", "aorta", "ivc", "rag", "lag", "gallbladder",
                     "esophagus", "stomach", "duodenum", "left-kidney"]


def __entry__() -> None:
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("--seg_path", default="/workspace/data/nnUNet_eval_output")
    parser.add_argument("--gt_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()
    filenames = [x for x in _listdir(args.seg_path) if x.endswith('.nii.gz')]
    seg_metrics = _OrderedDict({label: [] for label in LABELS})
    seg_metrics["mean"] = []
    for filename in filenames:
        pred = _uint8(_load(f"{args.seg_path}/{filename}").get_fdata())
        gt = _uint8(_load(f"{args.gt_path}/{filename}").get_fdata())
        for i in range(1, 14):
            if _sum(gt == i) == 0 and _sum(pred == i) == 0:
                dsc = 1
            elif _sum(gt == i) == 0 and _sum(pred == i) > 0:
                dsc = 0
            else:
                if i == 5 or i == 6 or i ==10:
                    z_lower, z_upper = find_lower_upper_zbound(gt == i)
                    organ_i_gt, organ_i_pred = gt[:, :, z_lower:z_upper] == i, pred[:, :, z_lower:z_upper] == i
                else:
                    organ_i_gt, organ_i_pred = gt == i, pred == i
                dsc = compute_dice_coefficient(organ_i_gt, organ_i_pred)
            seg_metrics[LABELS[i - 1]].append(dsc := round(dsc, 4))
            seg_metrics["mean"].append(dsc)
        df = _DataFrame({k: [sum(v) / len(v)] for k, v in seg_metrics.items()})
        df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    __entry__()
