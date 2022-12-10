from PIL import Image
import os
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

####================Main Preprocessing
# we classify defective items as positive, as they have the "signal" pixels
data_dir = "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/train/"
neg_dir = "ok_front/"
pos_dir = "def_front/"
output_dir = ""
# output dir split in same way
dir_list = [pos_dir, neg_dir]

pos_imgs, neg_imgs = [], []
for pos_file in os.listdir(data_dir + pos_dir):
    img = Image.open(data_dir + pos_dir + pos_file)
    img = np.array(img, dtype=np.uint8)
    pos_imgs.append(img)
for neg_file in os.listdir(data_dir + neg_dir):
    img = Image.open(data_dir + neg_dir + neg_file)
    img = np.array(img, dtype=np.uint8)
    neg_imgs.append(img)

x_data = np.concatenate([pos_imgs, neg_imgs], axis=0)
y_data = np.concatenate([np.ones((len(pos_imgs), 1)),
                         np.zeros((len(neg_imgs), 1))], axis=0)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, Dropout, Reshape, Lambda, Flatten
from tensorflow.keras.layers import GaussianDropout, GaussianNoise
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

import matplotlib as mplot
import matplotlib.pyplot as plt

import numpy as np

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


####================Main Model Function
def build_model(opt, lr, out_type):
    policy = tf.keras.mixed_precision.experimental.Policy(
        'mixed_float16')  # sets values to be float16 for nvidia 2000,3000 series GPUs, plus others im sure

    input_img = Input(shape=(679, 550, 1))
    x = Lambda(lambda x: K.cast_to_floatx(x))(input_img)
    x = Lambda(lambda x: x / 255.)(x)

    x = BatchNorm()(x)
    x = Conv2D(8, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = Conv2D(8, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = MaxPooling2D(2)(x)
    x = BatchNorm()(x)

    x = Conv2D(16, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = Conv2D(16, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = MaxPooling2D(2)(x)
    x = BatchNorm()(x)

    x = Conv2D(16, 2, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = Conv2D(16, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = MaxPooling2D(2)(x)
    x = BatchNorm()(x)

    x = Conv2D(32, 2, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = Conv2D(32, 3, padding='valid', activation='selu', kernel_initializer='lecun_normal', dtype=policy)(x)
    x = MaxPooling2D(2)(x)
    x = BatchNorm()(x)

    x = Conv2D(16, 1, activation="selu", kernel_initializer="lecun_normal", dtype=policy)(x)
    x = Conv2D(1, 1, activation='linear')(x)
    x_map = Activation("sigmoid")(x)

    y = GlobalMaxPooling2D()(x_map)

    if opt == "SGD":
        # looks like pretty good but noisy results with no momentum, lets check 0.9...
        optimizer = SGD(lr=lr)
    if opt == "RMSprop":
        optimizer = RMSprop(learning_rate=lr)
    if opt == "Adam":
        optimizer = Adam(learning_rate=lr)

    if out_type == "classify":
        model = Model(inputs=input_img, outputs=y)
    if out_type == "map":
        model = Model(inputs=input_img, outputs=x_map)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy', 'AUC'])

    return model


####================Data Generator, mostly used for rotation/flip augmentation
# idk why this is separated exaclty
def augment_image(img):
    if np.random.rand() > 0.5:
        img = np.transpose(img, axes=(1, 0, 2))
    img = np.rot90(img, np.random.randint(0, 3), axes=(0, 1))
    return img


class Augment_Generator(Sequence):
    def __init__(self, images, y_data, batch_size, shuffle, augment, mode):
        'Initialization'
        self.images = images
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode

        self.epoch_ix = np.arange(len(self.images))

        if self.shuffle:
            np.random.shuffle(self.epoch_ix)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.epoch_ix) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_ixs = self.epoch_ix[index * self.batch_size:(
                                                                      index + 1) * self.batch_size]  # exchange sequential call with shuffleable ix list
        x_batch = self.images[batch_ixs]
        if self.augment:
            x_batch = np.asarray([augment_image(img) for img in x_batch])

        if self.mode == "Train" or self.mode == "Validate":
            y_batch = self.y_data[batch_ixs]
            return x_batch, y_batch
        if self.mode == "Test":
            return x_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.epoch_ix)  # shuffle the order of lines


####================Callbacks and stuff for monitoring progress
class PlotLoss(Callback):
    def __init__(self, freq):
        self.freq = freq
        self.logs = []
        self.train_auc = []
        self.val_auc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.freq == 0:
            self.logs.append(logs)
            self.train_auc.append(logs.get('auc'))
            self.loss.append(logs.get('loss'))
            self.val_auc.append(logs.get('val_auc'))
            self.val_loss.append(logs.get('val_loss'))
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].plot(self.train_auc)
            ax[0].plot(self.val_auc)
            ax[1].plot(self.loss)
            ax[1].plot(self.val_loss)
            plt.show()
        return




####================Model Construction

shuffle_ix = list(range(len(x_data)))
print(len(shuffle_ix))
np.random.shuffle(shuffle_ix)
size = len(x_data)
print(size)
x_data = x_data[0:1]
# x_data = x_data[shuffle_ix]
# y_data = y_data[shuffle_ix]

val_ix = int(np.floor(len(x_data)*0.8))
x_train = x_data[:val_ix]
y_train = y_data[:val_ix]

x_valid = x_data[val_ix:]
y_valid = y_data[val_ix:]


train_batch, valid_batch = 16, 32
train_gen = Augment_Generator(x_train, y_train, batch_size=train_batch, shuffle=True, augment=True, mode="Train")
valid_gen = Augment_Generator(x_valid, y_valid, batch_size=valid_batch, shuffle=False, augment=False, mode="Validate")
train_steps = int(np.ceil(len(x_train)/train_batch))
valid_steps = int(np.ceil(len(x_valid)/valid_batch))


map_model = build_model("Adam", 1e-3, "map")
#Plot_Boxes = PlotBoxes(map_model, disp_imgs, (29, 64, 16), 5)
plot_Loss = PlotLoss(5)

model = build_model("Adam", 1e-3, "classify")
model.summary()



hist = model.fit(train_gen, steps_per_epoch=train_steps, epochs=30,
                 validation_data=valid_gen, validation_steps=valid_steps,
                 verbose=2)
plt.plot(hist.history['auc'])
plt.plot(hist.history['val_auc'])
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()


# save the model into h5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:/Users/dell/PycharmProjects/Graphical_interface_project/model/box.h5"
torch.save(model, model_path)
# model = torch.load(model_path, map_location=device)
model.save('C:/Users/dell/PycharmProjects/Graphical_interface_project/model/bounding_box.h5')


# takes in 6 images and activation maps, reutrns a 3x2 grid of images with box highlighting
def plot_maps(images, maps, k, dims):
    map_size = dims[0]
    r_size = dims[1]
    r_stride = dims[2]
    colors = ['#ff0000', '#ff0080', '#ff00ff', '#8000ff', '#0080ff', '#00ffff', '#00ff80']

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for j, (img, mask) in enumerate(zip(images, maps)):
        mask = mask.flatten()
        j_a = (j - j % 3) // 3
        j_b = j % 3
        img = np.asarray(img[:, :, 0], dtype=np.float32) / 255
        ax[j_a][j_b].imshow(img, cmap=plt.get_cmap('gray'))

        for i in range(k):
            a_max = np.argmax(mask)
            x_region = a_max % map_size  # note, numpy addresses work on arr[y, x, z], compared to image coordinates
            y_region = (a_max - (a_max % map_size)) // map_size
            prob = mask[a_max]
            if prob < 0.5: break
            x_pixel = r_stride * (x_region)
            y_pixel = r_stride * (y_region)

            # rectangle expects bottom left coordinates. We've generated Top Right
            rectangle = mplot.patches.Rectangle((x_pixel, y_pixel), r_size, r_size, edgecolor=colors[i - 1],
                                                facecolor="none")
            ax[j_a][j_b].add_patch(rectangle)

            font = {'color': colors[i - 1]}
            ax[j_a][j_b].text(x_pixel, y_pixel, s="{0:.3f}".format(prob), fontdict=font)

            mask[a_max] = 0  # to help get the next most maximum
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


#we select out some display images, 3 positive and 3 negative samples
disp_imgs_pos = x_valid[np.where(y_valid==1)[0][:3]]
disp_imgs_neg = x_valid[np.where(y_valid==0)[0][:3]]
disp_imgs = np.concatenate([disp_imgs_pos, disp_imgs_neg], axis=0)

map_model.set_weights(model.get_weights())

pred_maps = map_model.predict(disp_imgs)

plot_maps(disp_imgs, pred_maps, 5, (29, 64, 16))
