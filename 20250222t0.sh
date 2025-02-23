#!/bin/bash

python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 --name Default
python /workspace/code/main.py -tr nnUNetTrainerMICCAI -dn AbdomenCT -d 701 -p nnUNetResEncUNetMPlans --name ResEncUNetM