import argparse
import json
import torch
import os
import cv2
import numpy as np
from argparse import Namespace
torch.backends.cudnn.benchmark = True

# import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F

class Watermarkremover_model:

    def __init__(self,args):


        with open('arguments.json', 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args_1 = parser.parse_args(namespace=t_args)
        args_1 = vars(args_1)
        print(args_1["crop_size"])

        # Model 
        self.args = args
        self.Machine = models.__dict__[args.models](datasets=(None, None), args=args)
        self.model = self.Machine
        self.model.model.eval()
        print("==> Model loaded with Weights")

        

    def tensor2np(self,x, isMask=False):
        if isMask:
            if x.shape[1] == 1:
                x = x.repeat(1,3,1,1)
            x = ((x.cpu().detach()))*255
        else:
            x = x.cpu().detach()
            mean = 0
            std = 1
            x = (x * std + mean)*255
            
        return x.numpy().transpose(0,2,3,1).astype(np.uint8)


    def preprocess(self,file_path, img_size):
        img_J = cv2.imread(file_path)
        assert img_J is not None, "NoneType"
        h,w,_ = img_J.shape
        img_J = cv2.cvtColor(img_J, cv2.COLOR_BGR2RGB).astype(np.float)/255.
        img_J = torch.from_numpy(img_J.transpose(2,0,1)[np.newaxis,...]) #[1,C,H,W]
        org_size = [h,w]
        img_J = F.interpolate(img_J, size=(img_size, img_size), mode='bilinear')
    
        return img_J , org_size

    def get_inferene(self,file_path):
        img_size = self.args.crop_size
        print(img_size)
        image , org_size = self.preprocess(file_path,img_size)
        with torch.no_grad():
            image = image.to(self.model.device).float()
            outputs = self.model.model(image)
            imoutput,immask_all,imwatermark = outputs
            imoutput = imoutput[0]  
            immask = immask_all[0]
            imfinal =imoutput*immask + self.model.norm(image)*(1-immask) # output

            # resizing to orginal size 
            final_pred_resized = F.interpolate(imfinal, size=(org_size[0], org_size[1]), mode='bilinear')
            output = cv2.cvtColor(self.tensor2np(final_pred_resized)[0], cv2.COLOR_RGB2BGR)

            # save or return
            cv2.imwrite('output1.jpeg', output)
        return output
            

if __name__ == '__main__':
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    wmr = Watermarkremover_model(parser.parse_args())
    img=wmr.get_inferene('test_infer/Images/1-4k.jpeg')

    
   
