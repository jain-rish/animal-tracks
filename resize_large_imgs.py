import pandas as pd
import numpy as np
import cv2, os

file_directory = '/Users/rmillin/Documents/Insight/animal-tracks/downloads/images/all_others'

from os import listdir
from os.path import isfile, join

def shrink_images(file_directory, keyword, shrink_size):

    image_files = [f for f in listdir(file_directory) if isfile(join(file_directory, f))]
    for image_file in image_files:
        if keyword in image_file:
            try:
                img = cv2.imread(os.path.join(file_directory,image_file))
                orig_imsize = list(img.shape)
                if np.max(orig_imsize)>shrink_size: # then shrink it
                    sfactor = shrink_size/np.max(orig_imsize)
                    dim = (int(orig_imsize[1] * sfactor), int(orig_imsize[0] * sfactor)) 
                    # perform the actual resizing of the image and show it
                    new_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(file_directory,image_file.replace('.','_shrink.')), new_img)
                else:
                    continue
            except:
                print(image_file)

shrink_size = 500
shrink_images(file_directory,".",shrink_size)
shrink_images(file_directory,".",shrink_size)
