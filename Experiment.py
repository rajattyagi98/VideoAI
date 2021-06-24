# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:10:46 2021

@author: RAJAT
"""

# example of using a pre-trained model as a classifier
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from glob import glob
import matplotlib.pyplot as plt
import os
# Load HAAR face classifier

from keras.models import load_model
model = load_model('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//facefeatures_new_model1.h5') 
predictions = []
for file in os.listdir('C://Users//RAJAT//.spyder-py3//photo'):
    img_path = 'C://Users//RAJAT//.spyder-py3//photo//' + file
    img = load_img(img_path, target_size=(400, 400))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    frame = cv2.imread(img_path)
    if(pred[0][0]>0.65):
        cv2.putText(frame,'tallied', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(50)
    else:
        cv2.putText(frame,'Not tallied', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Video', frame)
        cv2.waitKey(50)
    if cv2.waitKey(1) & ord('q'):
        break
    