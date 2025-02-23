#!/bin/bash

python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702 --name Default --save_path /workspace/data/bwlab_eval_results_2.csv --eval_input /workspace/data/nnUNet_eval_input_2 --eval_output /workspace/data/nnUNet_eval_output_2
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenMR -d 702 -p nnUNetResEncUNetMPlans --name ResEncUNetM --save_path /workspace/data/bwlab_eval_results_2.csv --eval_input /workspace/data/nnUNet_eval_input_2 --eval_output /workspace/data/nnUNet_eval_output_2
