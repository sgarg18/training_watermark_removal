from __future__ import print_function, absolute_import

import argparse
import torch
import os
from math import log10
import cv2
import numpy as np
import src.networks as nets

torch.backends.cudnn.benchmark = True

import datasets as datasets
import src.models as models
from torchsummary import summary
from options import Options
import torch.nn.functional as F
import pytorch_ssim
from evaluation import compute_IoU, FScore, AverageMeter, compute_RMSE, normPRED
# from skimage.measure import compare_ssim as ssim
from pytorch_msssim import ssim, SSIM
# from skimage.metrics import structural_similarity as ssim
import time


pretrained_model_path = 'pre-trained_model/model_best.pth.tar'

model = nets.__dict__['slbr'](args=args)
# model = models.__dict__['slbr']
model.model.eval()

current_checkpoint = torch.load(pretrained_model_path)
if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
    print("this ran")
    current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
    current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

print(model)
model.model.load_state_dict(current_checkpoint['state_dict'], strict=True)
print(model)
print(summary(model, (3,256,256)))



# main function 
def main(args):





if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    main(parser.parse_args())