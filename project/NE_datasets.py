import copy
import logging
import os
import torch.utils.data
import torchvision
from PIL import Image
import torchvision.transforms as transforms_pytorch
import math

from openpifpaf import transforms
from openpifpaf import utils
from openpifpaf.datasets import collate_images_targets_meta

import numpy as np

ANNOTATIONS_TRAIN = '/home/zyi/data-mscoco/annotations/instances_train2017.json'
ANNOTATIONS_VAL = '/home/zyi/data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_TRAIN = '/home/zyi/data-mscoco/images/train2017/'
IMAGE_DIR_VAL = '/home/zyi/data-mscoco/images/val2017/'

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
        #self.filter_for_box_annotations()
        #self.ids = self.ids[:5]

        self.target_img = Image.open('patch.png', 'r').convert("RGB")
        
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = transforms.Normalize()
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

        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }
        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)


        image, anns = paste_img(image, self.target_img, image_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns)
        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)

        return self.single_image_processing(image, anns, meta, meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        # transform targets
        if self.target_transforms is not None:
            anns = [t(anns, original_size) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.ids)


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

def paste_img(background, target_img, image_id):
    transform_train = transforms_pytorch.Compose([
        transforms_pytorch.RandomHorizontalFlip(), ## Modify: You can remove
        transforms_pytorch.RandomAffine(math.pi/6, translate=None, shear=None, resample=False),# ## Modify + A fillcolor='white' can be added as argument
        transforms_pytorch.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5) ## Modify: different values
    ])

    img_w, img_h = target_img.size
    bg_w, bg_h = background.size
    scale = img_w / img_h
    img_size_w = bg_w
    img_size_h = bg_h
    min_object_size = bg_h//16 ## Modify: different scale
    max_object_size = bg_h//6  ## Modify: different scale



    # paste target image
    image_annotations = []
    is_bbox = np.random.rand() > 0.3 ## Modify: different probabilities
    if (is_bbox == False):
        x,y,h,w,x2,y2,h2,w2=0, 0, 0, 0, 0, 0, 0, 0
    else:
        h = np.random.randint(min_object_size, max_object_size)
        w = int(h*scale)
        target_img_i = target_img.copy()
        target_img_i = transform_train(target_img_i)
        target_img_i = target_img_i.resize((w,h))
        x = np.random.randint(0, (img_size_w - w) if (img_size_w - w)>0 else 0)
        y = np.random.randint(0, (img_size_h - h) if (img_size_h - h)>0 else 0)

        background.paste(target_img_i, (x,y))

    image_annotations.append({
                'image_id': image_id,
                'category_id': 0,
                'keypoints': [x+w//2, y+h//2,2 if is_bbox else 0],
                'num_keypoints' : 1 if is_bbox else 0,
                'bbox': [x, y, w, h],
                'iscrowd': 0,
                'segmentation': 0,
            })

    return background, image_annotations
