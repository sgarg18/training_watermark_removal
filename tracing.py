import argparse
import json
import torch
import os
import cv2
import numpy as np
from argparse import Namespace
torch.backends.cudnn.benchmark = True
from torch.cuda import amp
from torchvision import utils

# import datasets as datasets
import src.models as models
from options import Options
import torch.nn.functional as F

print(torch.__version__)

image = cv2.imread("test_infer/Images/1-4k.jpeg")
data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data=cv2.resize(data,(768,768))
data=torch.from_numpy(data[None]).to('cuda')

class WrappedModel(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        # Model 
        self.args = args
        self.Machine = models.__dict__[args.models](datasets=(None, None), args=args)
        self.model = self.Machine.model
        self.model.eval()
        print("==> Model loaded with Weights")

    @torch.inference_mode()
    def forward(self, data, fp16=False):    # [B,H,W,C]
        with amp.autocast(enabled=fp16):
            data = data.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]
            image = data.div(255)
            outputs = self.model(image)
            imoutput,immask_all,imwatermark = outputs
            imoutput = imoutput[0]  
            immask = immask_all[0]
            imfinal =imoutput*immask + self.Machine.norm(image)*(1-immask) # output
            imfinal = imfinal.mul_(255)
            imfinal = imfinal.to(torch.uint8)
            imfinal = imfinal.permute(0, 2, 3, 1)
            return imfinal.contiguous()


parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
wrp_model = WrappedModel(parser.parse_args())
with torch.no_grad():
    svd_out = wrp_model(data)
img_output = cv2.cvtColor(svd_out[0].cpu().numpy(), cv2.COLOR_BGR2RGB)
cv2.imwrite("w1.png",img_output)
print(svd_out.shape)

OUT_PATH = "out"
os.makedirs(OUT_PATH, exist_ok=True)


with torch.inference_mode(), torch.jit.optimized_execution(True):
    traced_script_module = torch.jit.trace(wrp_model, data)
    # traced_script_module = torch.jit.optimize_for_inference(traced_script_module)


# print(traced_script_module.code)

print(f"{OUT_PATH}/model.pt")
traced_script_module.save(f"{OUT_PATH}/model.pt")

# """
traced_script_module = torch.jit.load(f"{OUT_PATH}/model.pt")
# pdb.set_trace()

torch.cuda.synchronize()
with torch.no_grad():
    o = traced_script_module(data)
torch.cuda.synchronize()
print(o.shape)
img_output_2 = cv2.cvtColor(o[0].cpu().numpy(), cv2.COLOR_BGR2RGB)
print("probbbbkleeem")
cv2.imwrite("w2.png",img_output_2)

print(o.shape, o.dtype)

np.testing.assert_allclose(
    o.cpu().numpy(), svd_out.cpu().numpy(), rtol=1e-02, atol=1)
