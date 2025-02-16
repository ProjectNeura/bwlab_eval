from argparse import ArgumentParser as _ArgumentParser
from typing import Callable as _Callable, Any as _Any
from os import environ as _environ
from subprocess import run as _run
from nibabel import load as _load
from numpy import ndarray as _ndarray, sum as _sum
from time import time as _time

from preprocess import *


def timeit(runnable: _Callable[[...], _Any], *args) -> float:
    start = _time()
    runnable(*args)
    return _time() - start


def dice_coefficient(pred: _ndarray, gt: _ndarray) -> float:
    return float(2 * _sum(pred * gt) / (_sum(pred) + _sum(gt) + 1e-7))


def normalize(mask: _ndarray, keep_label: int) -> _ndarray:
    mask = mask.round()
    mask[mask == keep_label] = 1
    mask[mask != keep_label] = 0
    return mask


def evaluate(pred_dir: str, gt_dir: str, name_mapping: dict[str, str], label: int) -> list[float]:
    r = []
    for pred_name, gt_name in name_mapping.items():
        pred = normalize(_load(f"{pred_dir}/{pred_name.replace('_0000', '')}").get_fdata(), label)
        gt = normalize(_load(f"{gt_dir}/{gt_name.replace('_0000', '')}").get_fdata(), label)
        r.append(dice_coefficient(pred, gt))
    return r


def __entry__() -> None:
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("-d", "--dataset", type=int)
    parser.add_argument("-dn", "--dataset_name")
    parser.add_argument("-c", "--configuration", default="3d")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer")
    parser.add_argument("-p", "--plan", default="nnUNetPlans")
    parser.add_argument("-n", "--num_classes", type=int, default=13)
    parser.add_argument("--eval_input", default="/workspace/data/nnUNet_eval_input")
    parser.add_argument("--eval_output", default="/workspace/data/nnUNet_eval_output")
    args = parser.parse_args()
    clear_cache(args.eval_input)
    clear_cache(args.eval_output)
    name_mapping = select_samples(f"/workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/imagesTs",
                                  "/workspace/data/nnUNet_eval_input")
    _environ["nnUNet_raw"] = "/workspace/data/nnUNet_raw"
    _environ["nnUNet_preprocessed"] = "/workspace/data/nnUNet_preprocessed"
    _environ["nnUNet_results"] = "/workspace/data/nnUNet_weights"
    cmd = f"nnUNetv2_predict -i {args.eval_input} -o {args.eval_output} -d {args.dataset} -c {args.configuration} -tr {args.trainer} -p {args.plan} -f all"
    print(cmd)
    print(f"Inference time: {timeit(_run, cmd.split())}")
    dice_scores = []
    for label in range(1, args.num_classes + 1):
        label_dice_scores = evaluate(args.eval_output, f"/workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/labelsTs", name_mapping, label)
        print(f"Dice score for label {label}: {sum(label_dice_scores) / len(label_dice_scores)}")
        dice_scores += label_dice_scores
    print(f"Overall dice score: {sum(dice_scores) / len(dice_scores)}")


if __name__ == "__main__":
    __entry__()
