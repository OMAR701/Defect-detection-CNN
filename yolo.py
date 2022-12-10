import imutils
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from tensorflow.tools.compatibility import ipynb

##import model
##from model import LeNet
from tensorflow.keras.preprocessing import image

class LeNet:
    def build(width, height, depth, classes):
        model = Sequential()  # initialize the model
        inputshape = (height, width, depth)  # 128*28*3
        # first set of conv => relu => max pooling
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputshape))  # 124*124*20
        model.add(Activation('relu'))  # 124*124*20
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 62*62*20

        # second set of conv => relu => max pooling
        model.add(Conv2D(50, (5, 5), padding='same'))  # 58*58*50
        model.add(Activation('relu'))  # 58*58*50
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 29*29*50
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))  # creates nodes equal to  output number of classes
        model.add(Activation('softmax'))  # converts the value into probability coressponding to each class
        return model

img = image.load_img('C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train/def_front/010.jpg')
img


from imutils import paths
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
data = []
labels = []
args=dict({'dataset':"C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train",
           'model':r'C:/Users/dell/PycharmProjects/Graphical_interface_project/'})
imagePaths=sorted(list(paths.list_images(args['dataset'])))



random.seed(20)
random.shuffle(imagePaths)
##imagePaths
imagePaths



for i in imagePaths:
    image = cv2.imread(i)
    image=cv2.resize(image,(28,28))
    image_matrix=img_to_array(image)
    data.append(image_matrix)
    label= i.split(os.path.sep)[-2]
    if label == 'ok_front':
        label=1
    else: label=0
    labels.append(label)

data =np.array(data,dtype='float')/255
labels=np.array(labels)

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.25,random_state=40)
y_train= to_categorical(y_train,num_classes=2)
y_test= to_categorical(y_test,num_classes=2)



#  Using Data Aungmentaion to create more training images
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")
import cv2
# initilaizing the model :
print("Compiling model")
model= LeNet.build(width=28,height=28,depth=3,classes=2)
opt=  Adam(learning_rate=INIT_LR,beta_1=0.9,beta_2=0.999,decay=INIT_LR /25)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
print("training netwrok")

H=model.fit(aug.flow(x_train,y_train,batch_size=BS),validation_data=(x_test, y_test),
                    epochs=60,verbose=1,steps_per_epoch=len(x_train) // BS)

model.save('model_2/yolo.h5')

# FONT_HERSHEY_COMPLEX
# FONT_HERSHEY_SIMPLEX
# image = cv2.imread(testimagePaths_rand[0])
image = cv2.imread('C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test/def_front/Copy of 010.jpg')
org = image.copy()
image = cv2.resize(image, (28, 28))
image_matrix = img_to_array(image)
image_matrix = np.array(image_matrix, dtype='float') / 255
image = np.expand_dims(image_matrix, axis=0)
def_front, ok_front = model.predict(image)[0]

if ok_front > def_front:
    label = 'ok'
else:
    label = 'defect'

if ok_front > def_front:
    prob = ok_front
else:
    prob = def_front
label = "{}: {:.2f}%".format(label, prob * 100)
output = imutils.resize(org, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("out",output)
cv2.waitKey(0)