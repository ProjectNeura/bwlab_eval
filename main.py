from argparse import ArgumentParser as _ArgumentParser, BooleanOptionalAction as _BooleanOptionalAction
from subprocess import run as _run


if __name__ == "__main__":
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("-d", "--dataset", type=int)
    parser.add_argument("-dn", "--dataset_name")
    parser.add_argument("-c", "--configuration", default="3d")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer")
    parser.add_argument("-p", "--plan", default="nnUNetPlans")
    parser.add_argument("--eval_only", action=_BooleanOptionalAction, default=False,)
    parser.add_argument("-mp", "--model_path", default="")
    parser.add_argument("-cp", "--checkpoint", default="checkpoint_final.pth")
    args = parser.parse_args()
    if not args.eval_only:
        _run(f"python /workspace/code/infer.py -d {args.dataset} -dn {args.dataset_name} -c {args.configuration} -tr {args.trainer} -p {args.plan} -mp {args.model_path} -cp {args.checkpoint}".split())
    _run(f"python /workspace/code/eval.py --gt_path /workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/labelsTs --save_path /workspace/data/bwlab_eval_results.csv".split())
