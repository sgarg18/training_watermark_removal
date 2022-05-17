import cv2
import numpy as np
from PIL import Image
from triton.triton_api import InferModel
from Utils.gen_utils import download_image
import torch


class Watermarkremover:
    def __init__(self,
                 model_name="Watermarkremover",
                 model_version="1",
                 input_size=(768, 768, 3)):
        self.input_size = input_size
        self.model_name = model_name
        self.model_version = model_version
        self.triton_model = InferModel(url="localhost:8001", verbose=False)

    def prepare_infer_images(self, image_url):
        if isinstance(image_url, str):
            img = download_image(image_url)
        elif isinstance(image_url, np.ndarray):
            img = Image.fromarray(image_url)
        else:
            img = image_url
        img_arr = np.array(img.convert("RGB"))
        # original Size
        h,w,_ = img_arr.shape
        org_size = [h,w]
        h, w, _ = self.input_size
        img_arr=cv2.resize(img_arr,(h,w))
        return img_arr , org_size

    def process_image(self, image_url):
        input_raw_image,org_size = self.prepare_infer_images(image_url)
        args = (input_raw_image[None], )
        output = self.triton_model(args, model_name=self.model_name, model_version=self.model_version)
        output = output[0]
        output = cv2.resize(output,org_size)
        return output, image_url
