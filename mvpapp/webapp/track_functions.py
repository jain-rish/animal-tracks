#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def image_preprocessing(img):
    import cv2
    import numpy as np
    img_size = [50, 50]
    orig_imsize = list(img.shape)
    if orig_imsize[0] > orig_imsize[1]:
        sfactor = img_size[1] / orig_imsize[1]
    else:
        sfactor = img_size[0] / orig_imsize[0]
    # shrink/expand to have the larger size match the desired image size
    dim = (int(orig_imsize[1] * sfactor), int(orig_imsize[0] * sfactor))
    # perform the actual resizing of the image and show it
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # get the crop                
    margin = np.array([new_img.shape[0]-img_size[0], new_img.shape[1]-img_size[1]])/2
    cropped_img = new_img[int(round(margin[0])):int(round(margin[0]))+img_size[0], 
                          int(round(margin[1])):int(round(margin[1]))+img_size[1], :]
    print('completed preprocessing')
    return cropped_img

def image_feature_extraction(cropped_img):
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.applications.vgg19 import VGG19
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.vgg19 import preprocess_input, decode_predictions
    from keras.preprocessing import image
    from keras.models import Model
    import cv2
    # do feature generation
    cropped_img = np.reshape(cropped_img, (1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
    test_datagen = image.ImageDataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        vertical_flip=True,
        horizontal_flip=False
    )
    batch_size = 1
    test_generator = test_datagen.flow(
    cropped_img,
    batch_size = batch_size)
    # load the model

    vgg_conv = VGG19(weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224,3))

#    test_data = vgg_conv.predict(test_generator)
    test_data = vgg_conv.predict(cropped_img)
    test_data = np.reshape(test_data, (test_data.shape[2], test_data.shape[3]))
    print('generated features')
    return test_data

def image_classification(test_data):
    ##### NEED TO CHANGE TO LOAD MODEL INSTEAD WHEN MORE TIME ####
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import numpy as np
    import os
    
    train_data = np.squeeze(np.load('/Users/rmillin/Documents/Insight/animal-tracks/bottleneck_features_train.npy'))

    n_dogs = 22
    n_cougars = 22
    train_labels = np.array([0] * n_cougars + [1] * n_dogs)
    class_labels = ['cougar','dog']

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    
    clf = LogisticRegression(penalty='l1',C=0.01).fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print('classification complete')
    return class_labels[int(np.round(pred))]


def full_pipeline(img):
    '''
    Given a path to an img (img_path), performs the full processing pipeline
    '''
    cropped_img = image_preprocessing(img)
    test_data = image_feature_extraction(cropped_img)
    predicted_class = image_classification(test_data)
    return predicted_class
    
