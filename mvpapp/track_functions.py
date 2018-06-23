#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def image_preprocessing(img):
<<<<<<< HEAD
  import cv2
  import numpy as np

  # take a center crop of the image
  img_size = [224, 224]
  orig_imsize = list(img.shape)
  if orig_imsize[0]>orig_imsize[1]:
      sfactor=img_size[1]/orig_imsize[1]
  else:
      sfactor=img_size[0]/orig_imsize[0] 
  # shrink/expand to have the larger size match the desired image size
  dim = (int(orig_imsize[1] * sfactor), int(orig_imsize[0] * sfactor)) 
  # perform the actual resizing of the image and show it
  new_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  # get the crop        
  margin = np.array([new_img.shape[0]-img_size[0], new_img.shape[1]-img_size[1]])/2
  cropped_img = new_img[int(round(margin[0])):int(round(margin[0]))+img_size[0],int(round(margin[1])):int(round(margin[1]))+img_size[1],:]

  # make the image grayscale
  gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
  # median filter
  blur_img = cv2.medianBlur(gray_img, 13);
  
  print('completed preprocessing')
  return blur_img

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
  cropped_img = np.repeat(np.reshape(cropped_img, (1, cropped_img.shape[0], cropped_img.shape[1], 1)),3,3)
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
  batch_size=batch_size)
  # load the model

  vgg_conv = VGG19(weights='imagenet',
                  include_top=False)

#  test_data = vgg_conv.predict(test_generator)
  test_data = vgg_conv.predict(cropped_img)
  test_data = np.reshape(test_data, (1, np.prod(test_data.shape)))
  print('generated features')
  return test_data

def image_classification(test_data):
  ##### NEED TO CHANGE TO LOAD MODEL INSTEAD WHEN MORE TIME ####
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  import numpy as np
  import os
  
##  train_data = np.squeeze(np.load('/Users/rmillin/Documents/Insight/animal-tracks/mvpapp/webapp/static/data/gray_filt_multi_bottleneck_features_train.npy'))
  train_data = np.load('/Users/rmillin/Documents/Insight/animal-tracks/mvpapp/webapp/static/data/gray_filt_multi_bottleneck_features_train.npy')
  train_data = np.reshape(train_data, (train_data.shape[0], np.prod(train_data.shape[1:])))
  n_total = train_data.shape[0]
  n_classes = 4
  n_per_class = int(n_total/n_classes)
  # labels
  train_labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class + [3] * n_per_class)
  class_labels = ['a bear', 'a canine', 'a feline', 'an animal with hooves', 'a small animal']
=======
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
>>>>>>> c0779d4f5bfca422b17fea119967a4343adf196f

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    
    clf = LogisticRegression(penalty='l1',C=0.01).fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print('classification complete')
    return class_labels[int(np.round(pred))]


def full_pipeline(img):
<<<<<<< HEAD
  '''
  Given a path to an img (img_path), performs the full processing pipeline
  '''
  from os import listdir
  from os.path import isfile, join
  import random

  cropped_img = image_preprocessing(img)
  test_data = image_feature_extraction(cropped_img)
  predicted_class = image_classification(test_data)
  print(predicted_class)
  #### WILL HAVE TO FIGURE OUT HOW TO REWRITE THIS WHEN ON AWS #####
  file_directory = './webapp/static/images'  
  if predicted_class=='a bear':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('blackbear' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('grizzly' in f))]
    image_files = [random.choice(files1), random.choice(files2)]
    creature_names = ['black bear','grizzly bear']
#    track_file1 = [f for f in listdir(join(file_directory ,tracks) if (isfile(join(file_directory, tracks, f)) and ('blackbear' in f))]
#    track_file1 = [f for f in listdir(join(file_directory ,tracks) if (isfile(join(file_directory, tracks, f)) and ('blackbear' in f))]
  elif predicted_class=='a canine':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('dog' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('wolf' in f))]
    files3 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('fox' in f))]
    image_files = [random.choice(files1), random.choice(files2), random.choice(files3)]    
    creature_names = ['dog','wolf','fox']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('dog' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('wolf' in f))]
    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('fox' in f))]
    creature_tracks = [files1[0], files2[0], files3[0]]
  elif predicted_class=='a feline':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('cougar' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('bobcat' in f))]
    files3 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('lynx' in f))]
    image_files = [random.choice(files1), random.choice(files2), random.choice(files3)]    
    creature_names = ['cougar','bobcat','lynx']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('cougar' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('bobcat' in f))]
    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('lynx' in f))]
    creature_tracks = [files1[0], files2[0], files3[0]]
  elif predicted_class=='an animal with hooves':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('deer' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('goat' in f))]
    files3 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('elk' in f))]
    image_files = [random.choice(files1), random.choice(files2), random.choice(files3)]    
    creature_names = ['deer','goat','elk']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('deer' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('goat' in f))]
    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('elk' in f))]
    creature_tracks = [files1[0], files2[0], files3[0]]
  else:
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('bird' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('raccoon' in f))]
    files3 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('otter' in f))]
    image_files = [random.choice(files1), random.choice(files2), random.choice(files3)]
    creature_names = ['bird','raccoon','otter']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('bird' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('raccoon' in f))]
    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('otter' in f))]
    creature_tracks = [files1[0], files2[0], files3[0]]

  image_paths = [join(file_directory, image_file).replace('./webapp','..') for image_file in image_files]
  track_paths = [join(file_directory, 'tracks', creature_track).replace('./webapp','..') for creature_track in creature_tracks]
 
  print(image_paths)
  print(random.choice(files1))
  print(track_paths)
    
  return predicted_class, image_paths, creature_names, track_paths #, track_file
=======
    '''
    Given a path to an img (img_path), performs the full processing pipeline
    '''
    cropped_img = image_preprocessing(img)
    test_data = image_feature_extraction(cropped_img)
    predicted_class = image_classification(test_data)
    return predicted_class
>>>>>>> c0779d4f5bfca422b17fea119967a4343adf196f
    