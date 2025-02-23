#!/bin/bash

python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 -c 3d_fullres --name Default
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 -p nnUNetResEncUNetMPlans -c 3d_fullres --name ResEncUNetM