#!/bin/sh
#BSUB -q gpua100
#BSUB -J ft_ptv3_scannet
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
##BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -o exp/scannet/ft/ptv3/ft_ptv3_scannet%J.out
#BSUB -e exp/scannet/ft/ptv3/ft_ptv3_scannet%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/scannet/ft_ptv3.py
