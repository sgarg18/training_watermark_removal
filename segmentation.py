
import numpy as np
import cv2
from rbg720.postprocessing import PostprocessingTransparent
from triton.triton_api import InferModel
from Utils.gen_utils import download_image
from SOC_segmentation.logoplace import NumberPlateReplace
import PIL
from PIL import Image


class SOC_Segmentation:
    def __init__(self, model_name="SOC_segmentation", input_size=(768, 768), model_version='1'):
        self.input_size = input_size
        self.model_name = model_name
        self.model_version = model_version
        self.triton_model = InferModel(url="localhost:8001", verbose=False)
        self.logo = NumberPlateReplace()
        self.pp = PostprocessingTransparent(rem_out=True)

    def get_inference(self, img_url):
        if isinstance(img_url, str):
            img_url = download_image(img_url)
        if isinstance(img_url, PIL.Image.Image):
            img = np.array(img_url.convert("RGB"), dtype=np.uint8)
        else:
            img = img_url

        img1 = cv2.resize(img, self.input_size)
        args = (img1[None],)
        output = self.triton_model(args, model_name=self.model_name, model_version=self.model_version)
        output = output[0].copy()    # it's read only for some reason.
        return output, img

    def interior_removebg(self, img_url):
        mask, img = self.get_inference(img_url)
        mask = 255 - cv2.resize(mask, img.shape[-2::-1])
        self.pp.rem_out = False
        output = self.pp.mask_process(mask, img)
        return output

    def get_plate(self, img_url, logo_url='1'):
        mask, img = self.get_inference(img_url)
        ori_size = img.shape
        h, w, _ = mask.shape

        mask[:h//3, :] = 255
        self.pp.rem_out = True
        mask = 255 - cv2.resize(mask, ori_size[-2::-1])
        output = self.pp.mask_process(mask, img)

        mask=output[...,3]

        if not self.logo.plate_check(mask):
            return img

        if logo_url == '1':
            white_img = Image.new("RGB", ori_size[-2::-1], (255, 255, 255))
            output = self.logo.paste_logo(img, white_img, mask)
            return output

        elif logo_url == '2':
            return mask

        else:
            logo_url = download_image(logo_url)
            return self.logo.main(img, logo_url, mask)
