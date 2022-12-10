import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


##Generating Image Dataset


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_dir="C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train"
test_dir="C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test"


train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32, # number of images to process at a time
                                               target_size=(224, 224), # convert all images to be 224 x 224
                                               class_mode="binary", # type of problem we're working on
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)




##Creating CNN Model (Tinyvgg)


#CNN Model (Tinyvgg)
model_1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10,
                         kernel_size=3, # can also be (3, 3)
                         activation="relu",
                         input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
])

# Compile the model
model_1.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=20,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))


model_1.save('C:/Users/dell/PycharmProjects/Graphical_interface_project/model/casting_model.h5')


##Prep image function¶

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




import pathlib
import numpy as np
data_dir = pathlib.Path("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)




def_img=load_and_prep_image("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train/def_front/005.jpg")
def_img=tf.expand_dims(def_img, axis=0)



pred=model_1.predict(def_img)
pred_class = class_names[int(tf.round(pred)[0][0])]
pred_class


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  full_path=target_folder + "/" + random_image[0]
  img = mpimg.imread(full_path)

  #print(f"Image shape: {img.shape}") # show the shape of the image

  return img,full_path


import os

plt.figure(figsize=(10, 10))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    target_class = random.choice(["def_front", "ok_front"])
    img, img_path = view_random_image(
        target_dir="C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train/",
        target_class=target_class)

    # uncomment below and comment above to get always 2 ok and 2 def.
    #    if i%2==0:
    #        target_class="ok_front"
    #        img,img_path = view_random_image(target_dir="../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/",
    #                 target_class=target_class)
    #    else:
    #        target_class="def_front"
    #        img,img_path = view_random_image(target_dir="../input/real-life-industrial-dataset-of-casting-product/casting_data/casting_data/train/",
    #                 target_class=target_class)
    def_img = load_and_prep_image(img_path)
    def_img = tf.expand_dims(def_img, axis=0)

    pred = model_1.predict(def_img)
    pred_class = class_names[int(tf.round(pred)[0][0])]

    ax2 = plt.imshow(img, cmap=plt.cm.binary)
    if pred_class == 'def_front':
        for l in ax.spines:
            ax.spines[l].set_color('red')
    else:
        for l in ax.spines:
            ax.spines[l].set_color('green')

    plt.xlabel(f"target class:{target_class}, predicted:{pred_class}")

ax = plt.show()




