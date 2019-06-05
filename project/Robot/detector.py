import argparse
import glob
import json
import os

import numpy as np
import torch

from openpifpaf.network import nets
from openpifpaf import decoder, transforms
import torchvision
from data import COCO_LABELS
def cli(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.05)
    # parser.add_argument('images', nargs='*',
    #                     help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json'],
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    args = parser.parse_args(arguments)

    # # glob
    # if args.glob:
    #     args.images += glob.glob(args.glob)
    # if not args.images:
    #     raise Exception("no image files given")

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args

class Detector(object):
    """docstring for Detector"""
    def __init__(self, arguments, input_size = 320):
        super(Detector, self).__init__()

        self.args = cli(arguments)
        # load model
        self.model, _ = nets.factory_from_args(self.args)
        self.model = self.model.to(self.args.device)
        self.processor = decoder.factory_from_args(self.args, self.model)

        self.preprocess = transforms.SquareRescale(input_size, black_bars=False, random_hflip=False,horizontal_swap=None)

        self.image_transform = transforms.image_transform
        INV_COCO_LABELS = {v: k for k, v in COCO_LABELS.items()}
        self.chosen_label = INV_COCO_LABELS["person"]-1
        self.input_size = input_size

    def forward(self, image):
        w, h = image.size
        if self.preprocess is not None:
            image = self.preprocess(image,[])[0]
        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)

        ##Add a dimension
        image_tensors = original_image.unsqueeze(0)
        processed_images_cpu = image.unsqueeze(0)

        ##
        images = image_tensors.permute(0, 2, 3, 1)
        processed_images = processed_images_cpu.to(self.args.device, non_blocking=True)
        fields_batch = self.processor.fields(processed_images)

        self.processor.set_cpu_image(images[0], processed_images[0])
        keypoint_sets, scores = self.processor.keypoint_sets(fields_batch[0])

        pred_y_label = [np.argmax(kps[:,2])+1 for kps in keypoint_sets]
        pred_bboxes = [kps[np.argmax(kps[:,2])] for kps in keypoint_sets]
        if len(pred_bboxes) == 0:
            return [w/2,h/2,0,0], [0]

        pred_bboxes = pred_bboxes[pred_y_label == self.chosen_label]
        pred_bboxes[0] = pred_bboxes[0]*(w/self.input_size)
        pred_bboxes[1] = pred_bboxes[1]*(h/self.input_size)

        #pdb.set_trace()
        return pred_bboxes, [pred_y_label[pred_y_label == self.chosen_label]]
