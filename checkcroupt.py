from os import listdir
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd

filedeleted = []
count = 0
path = 'custom_dataset2/train/Watermarked_image'
# for i in tqdm(range(28548,57097)):
#     filename = (listdir(path)[i])
for filename in tqdm(listdir(path)):
    try:
        img = Image.open('custom_dataset2/train/Watermarked_image/'+filename) # open the image file
        img.verify() # verify that it is, in fact aEn image
    except (IOError, SyntaxError) as e:
        count+=1
        # os.remove('custom_dataset/train/Watermarked_image/'+filename) # one file at a time
        print('Bad file:', filename) 

print(count)
# Bad file: 000000505649.png