```shell
docker build --no-cache ./ -t bwlab_eval:latest
docker run --ipc=host --rm -v "S:/SharedDatasets:/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -dn AbdomenCT -d 701 -c 3d_fullres_S5
python /workspace/code/main.py -dn AbdomenCT -d 701 -c 3d_fullres_S4 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -dn AbdomenCT -d 701 -c 3d_fullres_S5 -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -dn AbdomenMR -d 702 -c 3d_fullres_S5
python /workspace/code/main.py -dn AbdomenMR -d 702 -c 3d_fullres_S4 -p nnUNetResEncUNetMPlans
```