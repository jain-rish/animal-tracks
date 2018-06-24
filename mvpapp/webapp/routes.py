#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:56:53 2018

@author: adam
"""
import glob, os, io, flask
import cv2
import urllib
import numpy as np
from werkzeug.utils import secure_filename
from webapp import app
from webapp import track_functions as tf

app.secret_key = 'rachel'

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
            img_file = flask.request.files.get('file')
#            img_name = img_file.filename
            img_name = secure_filename(img_file.filename)
            # Write image to static directory and do the hot dog check
            imgurl=os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            print(imgurl)
            img_file.save(imgurl)
#            img = kimage.load_img(imgurl, target_size=(224, 224))
            img = cv2.imread(imgurl)
            print(img)
#            req = urllib.request.urlopen(imgurl)
#            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#            img = cv2.imdecode(arr, -1) # 'Load it as it is'           print(type(img))
#            cropped_img = tf.image_preprocessing(img)
#            test_data = tf.image_feature_extraction(cropped_img)
#            predicted_class = tf.image_classification(test_data)
#            the_result = tf.full_pipeline(img)
            predicted_class, image_files, creature_names, track_paths = tf.full_pipeline(img)
##
##            predicted_class = 'bullshit!'
            
            return flask.render_template('output.html', the_result = predicted_class, image_files = image_files, creature_names = creature_names, track_paths = track_paths, input_photo = imgurl)
        flask.flash('Upload only image files')

        
        return flask.redirect(flask.request.url)
    
