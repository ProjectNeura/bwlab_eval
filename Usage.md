```shell
docker build --no-cache ./ -t bwlab_eval:latest
docker run --ipc=host --rm -v "S:/SharedDatasets:/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainer__nnUNetPlans__3d_fullres_S5
```