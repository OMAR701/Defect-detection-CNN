import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import time

t1 = time.time()



def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

from tensorflow import keras
model = keras.models.load_model('C:/Users/dell/PycharmProjects/Graphical_interface_project/model/casting_model.h5')


import pathlib
import numpy as np
def view_random_image(image_path):
  # Setup target directory (we'll view images from here)


  # Get a random image path
  #random_image = random.sample(os.listdir(image_path), 1)

  # Read in the image and plot it using matplotlib

  img = mpimg.imread(image_path)

  #print(f"Image shape: {img.shape}") # show the shape of the image

  return img



import random
def_img = load_and_prep_image("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test/def_front/Copy of Copy of 006.jpg")
def_img = tf.expand_dims(def_img, axis=0)
from datetime import datetime

# Getting the current date and time
dt = datetime.now()




import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# def view_random_image(target_dir, target_class):
#   # Setup target directory (we'll view images from here)
#   target_folder = target_dir+target_class
#
#   # Get a random image path
#   random_image = random.sample(os.listdir(target_folder), 1)
#
#   # Read in the image and plot it using matplotlib
#   full_path=target_folder + "/" + random_image[0]
#   img = mpimg.imread(full_path)
#
#   #print(f"Image shape: {img.shape}") # show the shape of the image
#
#   return img,full_path


import os
data_dir = pathlib.Path("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)
plt.figure(figsize=(10, 10))
for i in range(1):
    ax = plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    target_class = "def_front"
    img = view_random_image("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test/def_front/Copy of Copy of 006.jpg")

    pred = model.predict(def_img)

    pred_class = class_names[int(tf.round(pred)[0][0])]
    print(pred_class)

    ax2 = plt.imshow(img, cmap=plt.cm.binary)
    if pred_class == 'def_front':
        for l in ax.spines:
            ax.spines[l].set_color('red')
    else:
        for l in ax.spines:
            ax.spines[l].set_color('green')

    plt.xlabel(f" predicted:{pred_class}")

ax = plt.show()
