```shell
docker build ./ -t bwlab_eval:latest
docker run --rm -v "S:/SharedDatasets/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```