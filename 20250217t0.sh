#!/bin/bash

python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_100.pth --name repvgg100
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_200.pth --name repvgg200
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_300.pth --name repvgg300
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_400.pth --name repvgg400
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_500.pth --name repvgg500
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_600.pth --name repvgg600
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_700.pth --name repvgg700
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_800.pth --name repvgg800
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_900.pth --name repvgg900
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_1000.pth --name repvgg1000
python /workspace/code/main.py -dn AbdomenCT -d 701 -mp /workspace/data/nnUNet_weights/Dataset701_AbdomenCT/nnUNetTrainerMICCAI_repvgg__nnUNetPlans__3d_fullres -cp checkpoint_best.pth --name repvggbest