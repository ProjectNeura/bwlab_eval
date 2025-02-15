```shell
docker build ./ -t bwlab_eval:latest
docker run --rm -v "S:/SharedDatasets/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/eval.py -d 701 -dn AbdomenCT -tr nnUNetTrainerMICCAI
python /workspace/code/eval.py -d 701 -dn AbdomenCT -tr nnUNetTrainerMICCAI -p nnUNetResEncUNetMPlans
python /workspace/code/eval.py -d 702 -dn AbdomenMR -tr nnUNetTrainerMICCAI
python /workspace/code/eval.py -d 702 -dn AbdomenMR -tr nnUNetTrainerMICCAI -p nnUNetResEncUNetMPlans
```