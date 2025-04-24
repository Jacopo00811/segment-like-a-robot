#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J ft_sonata_1gpu
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s215170@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --


module load cuda/12.4

# Activate the conda environment
source /zhome/c8/c/169006/Repos/segment-like-a-robot/.venv/bin/activate

# link the dataset to the correct location
ln -s /dtu/blackhole/0e/169006/ScanNet/preprocessed /zhome/c8/c/169006/Repos/segment-like-a-robot/data/scannet

# run the training script

sh /zhome/c8/c/169006/Repos/segment-like-a-robot/Pointcept/scripts/train.sh -m 1 -g 2 -d sonata -c semseg-sonata-v1m1-0c-scannet-ft -n semseg-sonata-v1m1-0-base-0c-scannet-ft -w /zhome/c8/c/169006/Repos/segment-like-a-robot/models/sonata/pretrain-sonata-v1m1-0-base.pth
