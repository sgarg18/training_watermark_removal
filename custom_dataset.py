from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from tqdm import tqdm



class CustomDatasetDataLoader(torch.utils.data.Dataset):
    def __init__(self ,is_train,dataset_dir,img_size):
        super().__init__()
        self.img_size = img_size 

        if is_train == True:
            self.root = dataset_dir + '/train/'
        elif is_train == False:
            self.root = dataset_dir + '/test/' 

        mark='train'

        root_dataset='train2017'
        root_train='custom_dataset2'

        self.img_path=osp.join(root_dataset) # coco dataset



        self.img_source_path=osp.join(root_train,mark,'Watermarked_image','%s.jpg') 

        self.img_target_path=osp.join(root_train,mark,'Watermark_free_image','%s.jpg')

        self.mask_path=osp.join(root_train,mark,'Mask','%s.png')

        self.W_path=osp.join(root_train,mark,'Watermark','%s.png')

        self.base_img_path=osp.join(root_dataset,'%s.jpg') # watermark free image

        # print("base ",self.base_img_path)
        self.watermarks_path='watermarks' # watermark mask


        # Transformations
        self.base_img_transformations = A.Compose([
        A.Flip(),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(), 
            A.ColorJitter(brightness=(0.5,1.1), contrast=0.4, saturation=0.4, hue=0.05),           
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        # A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
        ])

        self.watermark_transformation = A.Compose([
            A.GridDistortion(p=0.2),
            A.Flip(p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=150, approximate=True, same_dxdy=True, p=0.2),
            A.Sharpen(p=0.4),          
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Affine(scale=(0.8,1.), translate_percent=(-0.10, 0.10), rotate=(-30, 30), shear=(-10,10), interpolation=cv2.INTER_LINEAR),
            # ! check here once size
            
            A.RandomResizedCrop(512, 512, scale=(0.5,1.0), ratio=(0.5, 2.),p=0.1),
            # A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_LINEAR),
        ])

        self.ids = list()
        for file in os.listdir(root_dataset):
            self.ids.append(file.strip('.jpg'))

    def __getitem__(self, index):

        img_id = self.ids[index]
        base_img = cv2.imread(self.base_img_path%img_id)
        
        # print("base Img read :",self.base_img_path%img_id)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        # display(Image.fromarray(base_img))
        base_img_transformed = self.base_img_transformations(image=base_img)
        # display(      base_img_transformed['image']))

        files=os.listdir(self.watermarks_path)
        watermark_img_name=random.choice(files)
        watermark_img_path = os.path.join(self.watermarks_path,watermark_img_name)
        watermark = Image.open(watermark_img_path)

        watermark = watermark.resize((base_img.shape[1],base_img.shape[0]))

        watermark_transformed = self.watermark_transformation(image =np.array(watermark))

        watermarked_image = Image.fromarray(base_img_transformed['image'])

        watermark_transformed = Image.fromarray(watermark_transformed['image'])

        # mask part _______________
        wmnp = np.array(watermark_transformed)
        mask = np.sum(wmnp,axis=2)>0
        final_mask = np.uint8(mask*255)
        
        factor =random.uniform(0.00, 1.00)
        
        enhancer= ImageEnhance.Brightness(watermark_transformed)
        enhanced_wm = enhancer.enhance(factor)
        
        watermarked_image.paste(enhanced_wm,(0,0), enhanced_wm)
        # display(watermarked_image)

        
        watermarked_image = np.array(watermarked_image)
        watermark = np.array(watermark_transformed)


        aug_base_img = base_img_transformed['image']

        # Saving all Images
        pil_aug = Image.fromarray(aug_base_img)
        pil_watermark =Image.fromarray(watermark)
        pil_watermarked_image = Image.fromarray(watermarked_image)
        pil_final_mask =Image.fromarray(final_mask)

        pil_aug.save(self.img_target_path%img_id)

        pil_watermark.save(self.W_path%img_id)

        pil_watermarked_image.save(self.img_source_path%img_id)

        pil_final_mask.save(self.mask_path%img_id)
        
        return aug_base_img , watermark , watermarked_image , final_mask

    def __len__(self):
        return len(self.ids)


dataset = CustomDatasetDataLoader(True,'datasets/custom_dataset2', 1080)

for i in tqdm(range(100000,118000)):
    aug_base_img , watermark , watermarked_image , final_mask =dataset[i]
