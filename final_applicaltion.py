# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:37:15 2021

@author: RAJAT
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import load_img, img_to_array
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import mysql.connector
import pandas as pd
import sqlalchemy
from PIL import Image
import cv2
import numpy as np
import os
import shutil
#import model1
# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('E://haarcascades//haarcascade_frontalface_alt.xml')
# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.1, 4)
    if faces is ():
        return None
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]
    return cropped_face
mysql_ = mysql.connector.connect(host = 'localhost', user='root', passwd='root')
mycursor = mysql_.cursor()
appno = int(input("Enter your application number: "))
mycursor.execute("use new")
engine = sqlalchemy.create_engine('mysql+pymysql://root:root@localhost:3306/new')
dataframe = pd.read_sql_table("kk",engine)
results = []
results.append(dataframe.loc[dataframe.App_no == appno,:])
print(results)
'''
if appno in dataframe[dataframe.App_no]:
    answer = 'y'
else:
    answer = 'n'
'''
answer = input("Are you registered cndidate(y/n)?")
answer_1 = input("Is this you (y/n)?")
if(answer == 'y' and answer_1 == 'y'):
        #New part starts from here--->
        candidate = input("What is your name?")
        #os.remove('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//facefeatures_new_model_19.h5')
        #shutil.rmtree('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train//'+candidate)
        #shutil.rmtree('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test//'+ candidate)
        file_name_path = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train' + '//' + candidate
        os.mkdir(file_name_path)
        file_name_path1 = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test' + '//' + candidate                    
        #path_new = os.path.join(file_name_path1,candidate)
        os.mkdir(file_name_path1)
        print("Capturing your faces")
        # Initialize Webcam
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (224, 224))
                #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                # Save file in specified directory with unique name
                if count <= 80:
                    new_path = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train//' + candidate + '//' + str(count) + '.jpg'
                else:
                    new_path = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test//' + candidate + '//' + str(count) + '.jpg'
                # Put count on images and display live count
                cv2.imwrite(new_path, face)
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not found")
                pass
            if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()
        print("FACE CAPTURED")
        
        IMAGE_SIZE = [224, 224]
        train_path = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train'
        valid_path = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test'
        # add preprocessing layer to the front of VGG
        vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        # don't train existing weights
        for layer in vgg.layers:
          layer.trainable = False
        # useful for getting number of classes
        folders = glob('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train//*')
        # our layers - you can add more if you want
        x = Flatten()(vgg.output)
        # x = Dense(1000, activation='relu')(x)
        prediction = Dense(len(folders), activation='softmax')(x)
        # create a model object
        model = Model(inputs=vgg.input, outputs=prediction)
        # view the structure of the model
        model.summary()
        # tell the model what cost and optimization method to use
        model.compile(
          loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy']
        )
        #from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train',
                                                         target_size = (224, 224),
                                                         batch_size = 32,
                                                         class_mode = 'categorical')
        test_set = test_datagen.flow_from_directory('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test',
                                                    target_size = (224, 224),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')
        # fit the model
        r = model.fit(
          training_set,
          validation_data=test_set,
          epochs=8,
          steps_per_epoch=len(training_set),
          validation_steps=len(test_set)
        )
        # loss
        plt.plot(r.history['loss'], label='train loss')
        plt.plot(r.history['val_loss'], label='val loss')
        plt.legend()
        plt.show()
        plt.savefig('LossVal_loss')
        # accuracies
        plt.plot(r.history['accuracy'], label='train acc')
        plt.plot(r.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.show()
        plt.savefig('AccVal_acc')
        print("MODEL CREATED")
        import tensorflow as tf
        from keras.models import load_model
        model.save('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//facefeatures_new_model.h5')
        video_capture = cv2.VideoCapture(0)
        while True:
            _, frame = video_capture.read()
            #canvas = detect(gray, frame)
            #image, face =face_detector(frame)
            model = load_model('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//facefeatures_new_model.h5') 
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
                    #cv2.imwrite(path, frame) #
                    cv2.waitKey(50)
                    break
            if cv2.waitKey(1) & ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        os.remove('C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//facefeatures_new_model.h5')
        real_0 = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Train//'+ candidate
        real_1 = 'C://Users//RAJAT//.spyder-py3//FACE_RECOGNITION//dataset//Test//'+ candidate
        shutil.rmtree(real_0,ignore_errors=True)
        shutil.rmtree(real_1,ignore_errors=True)
else:
    print("You're not that guy")
