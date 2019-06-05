#########################################################################
#                                                                       #
#    Authors: Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre    #
#    Year 2019                                                          #
#                                                                       #
#    Modified: Yannick Paul Klose                                       #
#    (Object detection instead of Human Pose estimation)                #
#########################################################################


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


# no need for training and validation set since the images are pasted randomly
# images used in annotations train and val are just background images
ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_train2017.json'
IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
IMAGE_DIR_VAL = 'data-mscoco/images/train2017/'

class CocoKeypoints(torch.utils.data.Dataset):
    
    def __init__(self, root, annFile, image_transform=None, target_transforms=None, preprocess=None, horzontalflip=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        
        # get all images - not filter
        
        self.cat_ids = self.coco.getCatIds()
        self.ids = self.coco.getImgIds()
        self.filter_for_box_annotations()

        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        self.log = logging.getLogger(self.__class__.__name__)
            

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        
        # set percentage for pasting
        threshold = 50
        rand_num = random.randint(0, 100)
        if rand_num > threshold:
            paste = True
        else:
            paste = False
            
            
        # just for after training on special dataset 
        after_training = False
        anns, overlay_image = self.modify_keypoints(anns, image_info['file_name'], paste, after_training)
       
        image = overlay_image.convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, preprocess_meta = self.preprocess(image, anns)

        meta.update(preprocess_meta)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_image(image, valid_area)

        # if there are not target transforms, done here
        self.log.debug(meta)
        if self.target_transforms is None:
            return image, anns, meta

        # transform targets
        targets = [t(anns, original_size) for t in self.target_transforms]
        return image, targets, meta
    
    def __len__(self):
        return len(self.ids)
    

    def modify_keypoints(self, anns, filename, paste, after_training):
        # in the end we just want to have one keypoint
        # this keypoints is the center of our chosen tracking object
        keypoint_array = [0]*(3)

       
        # image ID is the same all annotations of one image
        ann = anns[0]                   
        
        background_path = IMAGE_DIR_TRAIN + str(filename)
        object_path = "tracked_pattern/model2.png"
        
        if after_training:
            # background path will be overwritten!
            image, center_x, center_y, x_pos, y_pos, length, height = PR_pillow_testing.overlay(background_path, object_path, paste, True)
        else: 
            # take coco dataset for training!
            image, center_x, center_y, x_pos, y_pos, length, height = PR_pillow_testing.overlay(background_path, object_path, paste, False)
        
        # set keypoint array
        keypoint_array[0] = center_x
        keypoint_array[1] = center_y
        if (paste):
            # we always set the keypoint to visible 
            keypoint_array[2] = 2       
        else:
            # if paste is not true, no image is inserted
            keypoint_array[2] = 0       
            
        # extract important information out of json file
        image_id = ann['image_id']
        annotation_id = ann['id']      
        is_crowd = 0                    
        annotations = []
     
        # create annotations
        annotation_object = self.create_annotation(x_pos, y_pos, length, height, image_id, annotation_id, is_crowd)
        annotation_object['keypoints'] = keypoint_array
        annotations.append(annotation_object)
        
        return annotations, image
    
    def create_annotation(self, x_pos, y_pos, length, height, image_id, annotation_id, is_crowd):
            
        segmentations = []
        bbox = [x_pos, y_pos, length, height]
        
        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'id': annotation_id,
            'bbox': bbox
        }

        return annotation
    
    def filter_for_box_annotations(self):

        def has_keypoint_annotation(image_id):

            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                if 'bbox' not in ann:
                    continue
                # create bounding box
                if any(v > 0.0 for v in ann['bbox'][2::3]):
                    return True
            return False

        # select the images that have a keypoint!
        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]

        print('... done.')
    


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
    
    train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        
    )
    
    np.random.seed(100)
    
    
    # select if model should be finetuned or not!
    after_training = False
    if after_training:
        num_train_images = 1000
        num_val_images = 100
        num_pretrain_images = 100
    else:
        num_train_images = 40000
        num_val_images = 1000
        num_pretrain_images = 1000
        

    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, np.random.choice(len(train_data),num_train_images)), batch_size=args.batch_size, shuffle=not args.debug, pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True, collate_fn=collate_images_targets_meta)
    
    val_data = CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        
        
    )
    
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_data, np.random.choice(len(val_data),num_val_images)), batch_size=args.batch_size, shuffle=not args.debug, pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True, collate_fn=collate_images_targets_meta)
    

    pre_train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        
        
    )
    
    pre_train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, np.random.choice(len(train_data),num_pretrain_images)), batch_size=args.batch_size, shuffle=not args.debug, pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True, collate_fn=collate_images_targets_meta)
    

    return train_loader, val_loader, pre_train_loader
