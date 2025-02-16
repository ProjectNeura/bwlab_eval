```shell
docker build --no-cache ./ -t bwlab_eval:latest
docker run --ipc=host --rm -v "S:/SharedDatasets:/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/eval.py -d 701 -tr nnUNetTrainerMICCAI -dn AbdomenCT -c 3d_fullres
python /workspace/code/eval.py -d 701 -tr nnUNetTrainerMICCAI -dn AbdomenCT -c 3d_fullres -p nnUNetResEncUNetMPlans
python /workspace/code/eval.py -d 702 -tr nnUNetTrainerMICCAI -dn AbdomenMR -c 3d_fullres
python /workspace/code/eval.py -d 702 -tr nnUNetTrainerMICCAI -dn AbdomenMR -c 3d_fullres -p nnUNetResEncUNetMPlans
```