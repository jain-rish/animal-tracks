# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import pygame, sys
from PIL import Image
from crop_images import *
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import cv2, os

file_directory = '/Users/rmillin/Documents/Insight/animal-tracks/downloads/images/all_others/ready_to_crop'
file_names = [f for f in listdir(file_directory) if isfile(join(file_directory, f))]

# load and crop images

for file_name in file_names:

    try:
    
        input_loc = os.path.join(file_directory,file_name)
        output_loc = os.path.join(file_directory,'cropped',file_name.replace('.','_cropped.'))
        screen, px = setup(input_loc)
        left, upper, right, lower = mainLoop(screen, px)

        # ensure output rect always has positive width, height
        if right < left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower
        im = Image.open(input_loc)
        im = im.crop(( left, upper, right, lower))
        pygame.display.quit()
        im.save(output_loc)
    except:
        print('failure')
