#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def image_preprocessing(img):
  import cv2
  import numpy as np
  from os.path import join

  img_size = [224, 224]
  filt_size = 13
  orig_imsize = list(img.shape)
  file_directory = '/home/ubuntu/animal-tracks/mvpapp/webapp/uploads/preprocessed'  
  blur_img_fname = join(file_directory, 'blurred.jpg')

  # take a center crop of the image
  if orig_imsize[0]>orig_imsize[1]:
      sfactor=img_size[1]/orig_imsize[1]
  else:
      sfactor=img_size[0]/orig_imsize[0] 
  # shrink/expand to have the larger size match the desired image size
  dim = (int(orig_imsize[1] * sfactor), int(orig_imsize[0] * sfactor))
  print(orig_imsize)
  print(dim)
  # perform the actual resizing of the image and show it
  new_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  # get the crop        
  margin = np.array([new_img.shape[0]-img_size[0], new_img.shape[1]-img_size[1]])/2
  cropped_img = new_img[int(round(margin[0])):int(round(margin[0]))+img_size[0],int(round(margin[1])):int(round(margin[1]))+img_size[1],:]
                     
  # make the image grayscale
  gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

  # median filter
  blur_img = cv2.medianBlur(gray_img, filt_size);
  print(blur_img.shape)
  # save the blurred grayscale image
  cv2.imwrite(blur_img_fname, blur_img)
 
  return str.replace(file_directory,'preprocessed','')


def image_classification(test_data):
  
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import StandardScaler
  import numpy as np
  import os
  import pickle
  
  class_labels = ['a bear', 'a canine', 'a feline', 'an animal with hooves', 'unknown - sorry!']
  # load the trained logistic regression classifier
  filename = './webapp/static/data/finalized_model.sav'
  with open(filename, 'rb') as pickle_file:
    clf = pickle.load(pickle_file)
  # load the trained scaling function
  filename = './webapp/static/data/scaler.sav'
  with open(filename, 'rb') as pickle_file:
    scaler = pickle.load(pickle_file)
  # scale the features for the input image
  test_data = scaler.transform(test_data)
  # predict the class
  pred = clf.predict(test_data)
  print(pred)
  return class_labels[int(np.round(pred))]


def get_outputs(predicted_class):
  # based on prediction, figure out which images to display
  
  import random
  from os import listdir
  from os.path import isfile, join
  
  print(predicted_class)
  file_directory = './webapp/static/images'  
  if predicted_class=='a bear':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('blackbear' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('grizzly' in f))]
    image_files = [random.choice(files1), random.choice(files2)]
    creature_names = ['black bear','grizzly bear']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('blackbear' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('grizzly' in f))]
#    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('fox' in f))]
    creature_tracks = [files1[0], files2[0]]
  elif predicted_class=='a canine':
    files1 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('dog' in f))]
    files2 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('wolf' in f))]
    files3 = [f for f in listdir(file_directory) if (isfile(join(file_directory, f)) and ('fox' in f))]
    image_files = [random.choice(files1), random.choice(files2), random.choice(files3)]    
    creature_names = ['dog','wolf','fox']
    files1 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('dog' in f))]
    files2 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('wolf' in f))]
#    files3 = [f for f in listdir(join(file_directory, 'tracks')) if (isfile(join(file_directory, 'tracks', f)) and ('fox' in f))]
    creature_tracks = [files1[0], files2[0]]
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
 
  return image_paths, creature_names, track_paths

