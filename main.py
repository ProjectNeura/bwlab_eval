from argparse import ArgumentParser as _ArgumentParser
from subprocess import run as _run


if __name__ == "__main__":
    parser = _ArgumentParser(prog="bwlab_eval")
    parser.add_argument("-d", "--dataset", type=int)
    parser.add_argument("-dn", "--dataset_name")
    parser.add_argument("-tr", "--trainer", default="nnUNetTrainer")
    parser.add_argument("-p", "--plan", default="nnUNetPlans")
    args = parser.parse_args()
    _run(f"python infer.py -d {args.dataset} -dn {args.dataset_name} -c 3d_fullres -tr {args.trainer} -p {args.plan} -n 13")
    _run(f"python eval.py /workspace/data/nnUNet_eval_output /workspace/data/nnUNet_raw/Dataset{args.dataset}_{args.dataset_name}/imagesTs ./results.csv")
