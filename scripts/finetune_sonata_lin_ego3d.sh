#!/bin/sh
#BSUB -q gpua100
#BSUB -J ft_sonata_lin_ego3d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
##BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o exp/ego/ft/sonata-lin/ft_sonata_lin_ego3d%J.out
#BSUB -e exp/ego/ft/sonata-lin/ft_sonata_lin_ego3d%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/ego_3d/ft_lin_sonata.py
