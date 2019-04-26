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

ANNOTATIONS_TRAIN = 'data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = 'data-mscoco/images/train2017/'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'

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
class CocoKeypoints(torch.utils.data.Dataset):
    
    def __init__(self, root, annFile, image_transform=None, target_transforms=None, preprocess=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        
        # get all images - not filter
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
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

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
    
    def filter_for_box_annotations(self):
        
        # single keypoint per box in the center!
        
        print('filter for box annotations ...')
        def has_box_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                # create bounding box
                if any(v > 0.0 for v in ann['bbox'][2::3]):
                    bb = ann['bbox']
                    # dont create area, but keypoints for bounding box
                    x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                    center_x = (x2 + x1)/2
                    center_y = (y2 + y1)/2
                    classobject = ann['category_id']
                    keypoint_array = np.zeros(91*3)
                    print(" ")
                    print(image_id)
                    print(classobject)
                    keypoint_array[classobject*3] = center_x
                    keypoint_array[classobject*3+1] = center_y
                    keypoint_array[classobject*3+2] = 2 # 2 means visible keypoint
                    # create 1D array --> 80x3 (x,y,visible=2)
                    # create array that can have multiple entries
                    ann['keypoints'] = [keypoint_array]
                    
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_box_annotation(image_id)]
        print('... done.')

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
    train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

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
    
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    return train_loader, val_loader, pre_train_loader
