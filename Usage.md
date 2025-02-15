```shell
docker build ./ -t bwlab_eval:latest
docker run --rm -v "S:/SharedDatasets/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/eval.py -d 701 -dn AbdomenCT
python /workspace/code/eval.py -d 701 -dn AbdomenCT -p nnUNetResEncUNetMPlans
python /workspace/code/eval.py -d 702 -dn AbdomenMR
python /workspace/code/eval.py -d 702 -dn AbdomenMR -p nnUNetResEncUNetMPlans
```