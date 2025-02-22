FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
LABEL authors="Project Neura"

RUN apt update
RUN apt install -y git
RUN git clone https://github.com/ProjectNeura/bwlab_eval /workspace/code
RUN pip install nibabel pandas cupy
RUN pip install git+https://github.com/ProjectNeura/nnUNet