import copy
import logging
import os
import torch.utils.data
import torchvision
from PIL import Image

from openpifpaf import transforms
from openpifpaf import utils
from openpifpaf.datasets import collate_images_targets_meta

import PR_pillow_testing
from skimage import measure                        
from shapely.geometry import Polygon, MultiPolygon 

import numpy as np
import random

ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'

class CocoKeypoints(torch.utils.data.Dataset):
    """CocoKeypoints is a subclass of torch.utils.data.Dataset
    see https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    override the following methods:

    __len__ so that len(dataset) returns the size of the dataset.
    __getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
            Also, we don't read images until __getitem__ because it's more memory efficient.
    
    dataset will be a dict dataset[i] = {'image': image, 'landmarks': landmarks}
    Our dataset will take an optional argument transform so that any required processing can be applied on the sample. 
    
    input example:
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
    """

def train_cli(parser):
    """A function I don't know.
    """
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
    """This function prepares the data and directly give it back to train.py
    input:
    1. args,
    2. preprocess
    3. target_transforms
    output: train_loader, val_loader, pre_train_loader
    utilities:
        CocoKeypoints, which is a sub class of torch.utils.data.Dataset,
        which generates dataset for training and val.

    """
    train_data =  CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,  
    )

    np.random.seed(100)
    # use random number to use only 20k but not 118k training images.
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, np.random.choice(len(train_data),20000)),
        batch_size=args.batch_size, shuffle=not args.debug, pin_memory=args.pin_memory, 
        num_workers=args.loader_workers, drop_last=True, collate_fn=collate_images_targets_meta)

    val_data = CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)
    
    pre_train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        
    )
    pre_train_loader = torch.utils.data.DataLoader(
        pre_train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return train_loader, val_loader, pre_train_loader