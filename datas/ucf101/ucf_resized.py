
from __future__ import print_function
import csv
import glob
import os

from PIL import Image
#/data/UCF-101/
path_to_images = '../ucf_data/testsplit1_128/'

all_file = glob.glob(path_to_images + '*'+'/'+'*')


#返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。

# Resize images

for i, image_file in enumerate(all_file):
    temp = [os.path.join(image_file,x) for x in os.listdir(image_file)]
    for sample in temp:
        im = Image.open(sample)
        im = im.resize((128, 128), resample=Image.LANCZOS)
        im.save(sample)
    finish_word=image_file+" is finished..."
    print(finish_word)



