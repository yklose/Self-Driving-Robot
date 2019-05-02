#!/bin/bash -l
#SBATCH --workdir /home/klose
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos gpu_free
#SBATCH --account civil-459
#SBATCH --reservation civil-459-project
#SBATCH --time 12:00:00

module load gcc python cuda

source ~/venv/pytorch/bin/activate 

python train.py \
  --lr=1e-3 \
  --momentum=0.95 \
  --epochs=75 \
  --lr-decay 60 70 \
  --batch-size=8 \
  --basenet=resnet50block5 \
  --head-quad=1 \
  --headnets pif91 \
  --square-edge=401 \
  --regression-loss=laplace \
  --lambdas 30 2 2 \
  --crop-fraction=0.5 \
  --freeze-base=1
  
  
  