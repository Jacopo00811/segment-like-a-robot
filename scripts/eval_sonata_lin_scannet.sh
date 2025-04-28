#!/bin/sh
#BSUB -q gpua100
#BSUB -J eval_sonata_lin_scannet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -u s215158@dtu.dk
#BSUB -w ft_sonata_lin_scannet
#BSUB -B
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o exp/scannet/eval/sonata-lin/eval_sonata_lin_scannet%J.out
#BSUB -e exp/scannet/eval/sonata-lin/eval_sonata_lin_scannet%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/scannet/eval_lin_sonata.py
