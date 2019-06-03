module load gcc python cuda

source ~/venv/pytorch/bin/activate 

CUDA_VISIBLE_DEVICES=3 python ./PR_train.py \
  --lr=0.3e-3 \
  --momentum=0.95 \
  --epochs=100 \
  --lr-decay 60 70 \
  --batch-size=2 \
  --basenet=resnet50block5 \
  --head-quad=1 \
  --headnets pif1 \
  --square-edge=401 \
  --regression-loss=laplace \
  --lambdas 30 2 2 \
  --crop-fraction=0.5 \
  --freeze-base=1