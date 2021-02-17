#!/usr/bin/env bash
#!/bin/bash
# Configure the resources required
#SBATCH -M volta
#SBATCH -n 1 # number of cores (here 2 cores requested)
#SBATCH -c 8
#SBATCH --time=23:59:59 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:4# generic resource required (here requires 1 GPU)
#SBATCH --mem=256GB # memory pool for all cores (here set to 8 GB)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=b.zhang@adelaide.edu.au


source /hpcfs/users/a1652385/apps/anaconda3/bin/activate
conda activate torch1.1

bash ./tools/dist_train.sh  ./configs/padnet/padnet_r50_512x1024_40k_cityscapes_sem_loss_on.py  ./work_dirs/padnet_city
