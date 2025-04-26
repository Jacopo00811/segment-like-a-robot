#!/bin/sh
#BSUB -q gpua100
#BSUB -J ft_ptv3_scannet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
##BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o exp/scannet/ft/ptv3/ft_ptv3_scannet%J.out
#BSUB -e exp/scannet/ft/ptv3/ft_ptv3_scannet%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/scannet/ft_ptv3.py
