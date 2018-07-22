#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:56:53 2018

"""
import glob, os, io, flask
import cv2
import urllib
import tensorflow
import numpy as np
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from webapp import app
from webapp import track_functions as tf

app.secret_key = 'rachel'
graph = tensorflow.get_default_graph()
cnn = VGG16(weights='imagenet',
                  include_top=False)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/',  methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])


def index():
    
    # Get method type
    method = flask.request.method
    print(method)


    if method == 'GET':
        return flask.render_template('index.html')
    
    if method == 'POST':
        # No file found in the POST submission
        if 'file' not in flask.request.files:
            print("FAIL")
            return flask.redirect(flask.request.url)

        # File was found
        file = flask.request.files['file']
        if file and allowed_file(file.filename):
            print('SUCCESS')

            # Image info
            img_shape = (224, 224)
            img_file = flask.request.files.get('file')
            img_name = secure_filename(img_file.filename)
            
            # Write image to static directory
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            print(imgurl)
            img_file.save(imgurl)
            img = cv2.imread(imgurl)
            print(img)
            
            # preprocess
            preproc_img_dir = tf.image_preprocessing(img)
            print(preproc_img_dir)
            print(os.listdir(preproc_img_dir))
            
            # get cnn features
            test_datagen = image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=0,
                width_shift_range = 0,
                height_shift_range = 0,
                shear_range=0,
                zoom_range=0,
                fill_mode = 'nearest'
            )

            batch_size = 50
            test_generator = test_datagen.flow_from_directory(
                preproc_img_dir,
                target_size = img_shape,
                batch_size = batch_size,
                class_mode = None,
                shuffle = False
            )

            global graph
            with graph.as_default():
                test_data = cnn.predict_generator(test_generator)
            print(test_data.shape)
            test_data = np.reshape(test_data, (1, np.prod(test_data.shape)))
            
            # identify
            predicted_class = tf.image_classification(test_data)
            image_files, creature_names, track_paths = tf.get_outputs(predicted_class)
            
            return flask.render_template('output.html', the_result = predicted_class, image_files = image_files, creature_names = creature_names, track_paths = track_paths, input_photo = imgurl)

        flask.flash('Upload only image files')
        
        return flask.redirect(flask.request.url)
    
