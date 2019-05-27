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
        root=args.train_image_dir, ## or args.val_image_dir
        annFile=args.train_annotations, ## or .val_
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
    """
    def __init__(self, root, annFile, image_transform=None, target_transforms=None, preprocess=None, horzontalflip=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        
        # get all images - not filter
        
        self.cat_ids = self.coco.getCatIds()
        self.ids = self.coco.getImgIds()
        self.filter_for_box_annotations()
        #self.ids = self.ids[:5]
        
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        self.log = logging.getLogger(self.__class__.__name__)
    
    def __getitem__(self,index):
        """"Important variables:
        image_info: created by coco.loadImgs(), It has 'file_name' dict to load our file
        """
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)

        # 50% no object algorithm
        threshold = 50
        rand_num = random.randint(0, 100)
        if rand_num > threshold:
            paste = True
        else:
            paste = False
        anns, overlay_image = self.modify_keypoints(anns, image_info['file_name'], paste)
        
        image = overlay_image.convert('RGB')
        #with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
        #    image = Image.open(f).convert('RGB')
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }
        return image, targets, meta
        
    def __len__(self):
        return len(self.ids)

    def modify_keypoints(self, anns, filename, paste):
        # in the end we just want to have one keypoint
        # this keypoints is the center of our chosen tracking object
        keypoint_array = [0.,0.,0.]
       
        #ann = anns[0]                   # image ID is the same all annotations of one image
        
        #background_path = IMAGE_DIR_TRAIN + str(filename)
        object_path = "test_images/model.png"
        image, center_x, center_y, x_pos, y_pos, length, height = PR_pillow_testing.overlay(background_path, object_path, paste)
        
        # set keypoint array
        keypoint_array[0] = center_x
        keypoint_array[1] = center_y
        if (paste):
            keypoint_array[2] = 2       # we always set the keypoint to visible 
        else:
            keypoint_array[2] = 0       # if paste is not true, no image is inserted
            
        # extract important information out of json file
        image_id = ann['image_id']
        annotation_id = ann['id']       # take unique annotation ID (this is unique over all images?)
        is_crowd = 0                    # single object
        annotations = []
     
        # create annotations
        bbox = [x_pos, y_pos, length, height]
        annotation_object = {
            'segmentation': [],
            'iscrowd': 0,
            'image_id': 1,
            'id': 1,
            'bbox': bbox
        }
        annotation_object['keypoints'] = keypoint_array
        annotations.append(annotation_object)
        
        return annotations, image


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