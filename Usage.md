```shell
docker build ./ -t bwlab_eval:latest
docker run --rm -v "S:/SharedDatasets/workspace/data" --gpus="device=0" -it bwlab_eval:latest
```

```shell
python /workspace/code/preprocess.py
export nnUNet_raw=/workspace/data/nnUNet_raw
export nnUNet_preprocessed=/workspace/data/nnUNet_preprocessed
export nnUNet_results=/workspace/data/nnUNet_weights
nnUNetv2_predict -i /workspace/data/nnUNet_eval_input -o /workspace/data/nnUNet_eval_output -d 701 -c 2d -p nnUNetResEncUNetMPlans -f all --save_probabilities
```