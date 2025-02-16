```shell
docker build --no-cache ./ -t bwlab_eval:latest
docker run --ipc=host --rm -v "S:/SharedDatasets:/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/main.py -d 701 -tr nnUNetTrainerMICCAI -dn AbdomenCT
python /workspace/code/main.py -d 701 -tr nnUNetTrainerMICCAI -dn AbdomenCT -p nnUNetResEncUNetMPlans
python /workspace/code/main.py -d 702 -tr nnUNetTrainerMICCAI -dn AbdomenMR 
python /workspace/code/main.py -d 702 -tr nnUNetTrainerMICCAI -dn AbdomenMR -p nnUNetResEncUNetMPlans
```