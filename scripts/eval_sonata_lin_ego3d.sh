#!/bin/sh
#BSUB -q gpua100
#BSUB -J eval_sonata_lin_ego3d
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -u s215158@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -o exp/ego/eval/sonata-lin/eval_sonata_lin_ego3d%J.out
#BSUB -e exp/ego/eval/sonata-lin/eval_sonata_lin_ego3d%J.err

source /zhome/f9/0/168881/Desktop/segment-like-a-robot/.venv/bin/activate

python src/ego_3d/eval_lin_sonata.py
