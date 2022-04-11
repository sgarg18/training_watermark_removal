from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
from .base_dataset import get_transform
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2



class SpyneDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args, root_dataset = 'train2017'):
        
        args.is_train = is_train == 'train'       
        if args.is_train == True:
            self.root = args.dataset_dir + '/train/'
            # self.keep_background_prob = 0.01
            self.keep_background_prob = -1
        elif args.is_train == False:
            self.root = args.dataset_dir + '/test/' #'/test/'
            self.keep_background_prob = -1
            args.preprocess = 'resize'
            args.no_flip = True

        self.img_size = (args.crop_size,args.crop_size)
        # Paths 
        self.watermarks_path='watermarks'
        self.base_img_path=osp.join(root_dataset,'%s.jpg') #update this path - jpg part
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
        A.RandomResizedCrop(self.img_size[0],self.img_size[1],scale=(0.5,1.0), ratio=(0.5, 2.),p=0.8),

        ])

        self.watermark_transformation = A.Compose([
            A.GridDistortion(p=0.2),
            A.Flip(p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=150, approximate=True, same_dxdy=True, p=0.2),
            A.Sharpen(p=0.5),          
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.Affine(scale=(0.8,1.), translate_percent=(-0.10, 0.10), rotate=(-30, 30), shear=(-10,10), interpolation=cv2.INTER_LINEAR),
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50),
            A.RandomResizedCrop(self.img_size[0],self.img_size[1],scale=(0.5,1.0), ratio=(0.5, 2.),p=0.8),
        ])

        self.args = args

        # Augmentataion?
        self.transform_norm=transforms.Compose([
            transforms.ToTensor()])

        self.augment_transform = get_transform(args, 
            additional_targets={'J':'image', 'I':'image', 'watermark':'image', 'mask':'mask'}) #,
            # additional_targets={'J':'image', 'I':'image', 'watermark':'image', 'mask':'mask', 'alpha':'mask' }) #,

        self.transform_tensor = transforms.ToTensor()
		
        self.ids = list()
        # Itterate in base Img path and get ids 
        for file in os.listdir(root_dataset):
            self.ids.append(file.strip('.jpg'))
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # For random choice of watermarks making it a list
        self.watermark_folder_list = []
        for file in os.listdir(self.watermarks_path):
            self.watermark_folder_list.append(file)
    
       
    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, index):
        img_id = self.ids[index]
        base_img = cv2.imread(self.base_img_path%img_id)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        # Selecting random watermark
        idx = random.randint(0, len(self.watermark_folder_list)-1)
        watermark_img_name = self.watermark_folder_list[idx]
        watermark_img_path = os.path.join(self.watermarks_path,watermark_img_name)
        watermark = Image.open(watermark_img_path)
        # Resizing to base size
        # self.base_img_size = base_img.shape
        # watermark = watermark.resize((base_img.shape[1],base_img.shape[0]))


        # Transformations :
        base_img_transformed = self.base_img_transformations(image=base_img)
        watermark_transformed = self.watermark_transformation(image =np.array(watermark))
        
        watermarked_image = Image.fromarray(base_img_transformed['image'])
        watermark_transformed = Image.fromarray(watermark_transformed['image'])
        watermark_transformed = watermark.resize(watermarked_image.size)
        
        # mask part
        wmnp = np.array(watermark_transformed)
        mask = np.sum(wmnp,axis=2)>0
        final_mask = np.uint8(mask*255)
        watermarked_image.paste(watermark_transformed,(0,0), watermark_transformed)
        watermarked_image = np.array(watermarked_image)

        aug_base_img = base_img_transformed['image']

        watermark = np.array(watermark_transformed)

        # Resizing to input_size
        final_mask = cv2.resize(final_mask, self.img_size)
        watermarked_image = cv2.resize(watermarked_image, self.img_size)
        base_img = cv2.resize(base_img, self.img_size)
        watermark = cv2.resize(watermark, self.img_size)

        # print("Shapes : ", final_mask.shape,watermarked_image.shape,base_img.shape, watermark.shape)
        return {'J': watermarked_image, 'I': aug_base_img, 'watermark': watermark, 'mask':final_mask}

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        J = self.transform_norm(sample['J'])
        I = self.transform_norm(sample['I'])
        w = self.transform_norm(sample['watermark'])
        
        mask = sample['mask'][np.newaxis, ...].astype(np.float32)
        mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)
        # alpha = sample['alpha'][np.newaxis, ...].astype(np.float32)

        data = {
            'image': J,
            'target': I,
            'wm': w,
            'mask': mask
        }
        #  data = {
        #     'image': J,
        #     'target': I,
        #     'wm': w,
        #     'mask': mask,
        #     'alpha':alpha,
        #     'img_path':sample['img_path']
        # }
        return data

    def check_sample_types(self, sample):
        assert sample['J'].dtype == 'uint8'
        assert sample['I'].dtype == 'uint8'
        assert sample['watermark'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.augment_transform is None:
            return sample
        #print(self.transform.additional_targets.keys())
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.augment_transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augment_transform(image=sample['I'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            #print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 100



