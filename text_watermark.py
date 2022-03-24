import cv2 
import numpy as np
import random
from PIL import Image

def rotate(src, angle):
    rows,cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    dst = cv2.warpAffine(src, M, (cols,rows))
    return dst
shape = (1080,1920,4)
x = np.zeros(shape, dtype=np.uint8)
text_list = ['@atharvjairath','@Photograph','@randotmextret','STOCK','Zomato','Adobe']
cv2.putText(x, random.choice(text_list), (1200,200), random.randint(0,7), random.randint(1,2), (255,255,255,255), 2)
angle_list = [0,45,90,-90,-45]
x = rotate(x, 0)
img = Image.fromarray(x)
img.save('imadethis24.png')


