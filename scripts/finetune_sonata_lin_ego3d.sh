#!/bin/sh
#BSUB -q gpua100
#BSUB -J ft_sonata_lin_ego3d
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
##BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -o exp/ego/ft/sonata-lin/ft_sonata_lin_ego3d%J.out
#BSUB -e exp/ego/ft/sonata-lin/ft_sonata_lin_ego3d%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/ego_3d/ft_lin_sonata.py
