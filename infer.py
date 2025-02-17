from argparse import ArgumentParser as _ArgumentParser
from os import environ as _environ
from subprocess import run as _run
from time import time as _time
from typing import Callable as _Callable, Any as _Any

from preprocess import *


def timeit(runnable: _Callable[[...], _Any], *args) -> float:
    start = _time()
    runnable(*args)
    return _time() - start


def __entry__() -> None:
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("-d", "--dataset", type=int)
    parser.add_argument("-dn", "--dataset_name")
    parser.add_argument("-c", "--configuration", default="3d")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer")
    parser.add_argument("-p", "--plan", default="nnUNetPlans")
    parser.add_argument("-ps", "--plans_stage", type=int, default=-1)
    parser.add_argument("--eval_input", default="/workspace/data/nnUNet_eval_input")
    parser.add_argument("--eval_output", default="/workspace/data/nnUNet_eval_output")
    args = parser.parse_args()
    clear_cache(args.eval_input)
    clear_cache(args.eval_output)
    select_samples(f"/workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/imagesTs",
                   args.eval_input)
    _environ["nnUNet_raw"] = "/workspace/data/nnUNet_raw"
    _environ["nnUNet_preprocessed"] = "/workspace/data/nnUNet_preprocessed"
    _environ["nnUNet_results"] = "/workspace/data/nnUNet_weights"
    stage = "" if args.plans_stage < 0 else f" --plans_stage {args.plans_stage}"
    cmd = f"nnUNetv2_predict -i {args.eval_input} -o {args.eval_output} -d {args.dataset} -c {args.configuration} -tr {args.trainer} -p {args.plan}{stage} -f all"
    print(cmd)
    print(f"Inference time: {timeit(_run, cmd.split())}")


if __name__ == "__main__":
    __entry__()
