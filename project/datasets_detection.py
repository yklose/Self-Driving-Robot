import copy
import logging
import os
import torch.utils.data
import torchvision
from PIL import Image

from openpifpaf import transforms
from openpifpaf import utils
from openpifpaf.datasets import collate_images_targets_meta

import numpy as np

ANNOTATIONS_TRAIN = ''
ANNOTATIONS_VAL = ''
IMAGE_DIR_TRAIN = ''
IMAGE_DIR_VAL = ''

################################################################################
# TODO:                                                                        #
# - Create dataset class modeled after CocoKeypoints in the official           #
#   OpenPifPaf repo                                                            #
# - Modify to take all categories of COCO (CocoKeypoints uses only the human   #
#   category)                                                                  #
# - Using the bounding box and class labels, create a new ground-truth         #
#   annotation that can be used for detection                                  #
#   (using a single keypoint per class, being the center of the bounding box)  #
#                                                                              #
# Hint: Use the OpenPifPaf repo for reference                                  #
#                                                                              #
################################################################################
pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sampe')
    group.add_argument('--loader-workers', default=2, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')


def train_factory(args, preprocess, target_transforms):
    ################################################################################
    # TODO:                                                                        #
    # Implement the dataset loaders and datasets                                   #
    #                                                                              #
    # Hint: Use the OpenPifPaf repo for reference (especially datasets.py)         #
    ################################################################################
    pass
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    return train_loader, val_loader, pre_train_loader
