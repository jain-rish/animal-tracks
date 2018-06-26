from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from shutil import copyfile
import cv2, os
import numpy as np

basepath = '/Users/rmillin/Documents/Insight/image_reorg'

animals = ['bear','canine','feline','hooved','others']
n_images = 51

n_testing = 8
n_training = n_images-n_testing

n_folds = 6

for fold in range(n_folds):
    test_inds = np.arange(fold*n_testing,(fold+1)*n_testing)
    training_inds = np.arange((fold+1)*n_testing,(fold+1)*n_testing+n_training)
    training_inds[training_inds>(n_images-1)] = training_inds[training_inds>(n_images-1)] - n_images
    print(training_inds)
    print(test_inds)
    directory = join(basepath, 'fold'+str(fold))
    # make directories if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    train_directory = join(directory, 'train')
    test_directory = join(directory, 'test')
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    for animal in animals:
        animal_files = [f for f in listdir(join(basepath, animal)) if isfile(join(basepath, animal, f))]
        # make directories if they don't exist
        if not os.path.exists(join(train_directory, animal)):
            os.makedirs(join(train_directory, animal))
        if not os.path.exists(join(test_directory, animal)):
            os.makedirs(join(test_directory, animal))
       
        # copy the files to the directories
        for f in training_inds:
            copyfile(join(basepath, animal, animal_files[f]), join(train_directory, animal, animal_files[f])) 
        for f in test_inds:
            copyfile(join(basepath, animal, animal_files[f]), join(test_directory, animal, animal_files[f])) 
    

