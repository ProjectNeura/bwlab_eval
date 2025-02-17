from argparse import ArgumentParser as _ArgumentParser, BooleanOptionalAction as _BooleanOptionalAction
from subprocess import run as _run


if __name__ == "__main__":
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("-d", "--dataset", type=int)
    parser.add_argument("-dn", "--dataset_name")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer")
    parser.add_argument("-p", "--plan", default="nnUNetPlans")
    parser.add_argument("--eval_only", action=_BooleanOptionalAction, default=False,)
    args = parser.parse_args()
    _run(f"python /workspace/code/infer.py -d {args.dataset} -dn {args.dataset_name} -c 3d_fullres -tr {args.trainer} -p {args.plan}".split())
    _run(f"python /workspace/code/eval.py --gt_path /workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/labelsTs --save_path /workspace/data/bwlab_eval_results.csv".split())
