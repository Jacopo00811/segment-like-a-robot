#!/bin/sh
#BSUB -q gpua100
#BSUB -J ft_sonata_lin_scannet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
##BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:30
#BSUB -o exp/scannet/ft/sonata-lin/ft_sonata_lin_scannet%J.out
#BSUB -e exp/scannet/ft/sonata-lin/ft_sonata_lin_scannet%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/scannet/ft_lin_sonata.py