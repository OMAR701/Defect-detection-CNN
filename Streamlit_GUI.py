import base64
import os
import pathlib
from distutils.command.install import key
from io import BytesIO
from pyecharts.charts import Bar
from pyecharts import options as opts
import streamlit as st
import streamlit as st
from PIL import Image
from click import style
import tensorflow as tf
# load images
from matplotlib import pyplot as plt
from opt_einsum.backends import torch
import numpy as np
from pyparsing import col
from tensorflow import keras
model = keras.models.load_model('C:/Users/dell/PycharmProjects/Graphical_interface_project/model/casting_model.h5')
from datetime import datetime
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import time
import pandas as pd  # read csv, df manipulation

import sys
# Getting the current date and time
dt = datetime.now()
# getting the timestamp
logo = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/omarTUNALOGO2.jpg")

import os
from PIL import Image


# Given information




def main():
    # st.title("The result")
    # menu = ["Image","Dataset","DocumentFiles","About"]
    # choice = st.sidebar.selectbox("Menu",menu)
    number_of_defect = 0
    number_of_good = 0
    with st.sidebar:
        choose = option_menu("serdine project",
                             ["About", "Image", "real_time", "Statistics"],
                             icons=['house', 'camera fill', 'kanban', 'bar-chart-line', 'person lines fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
                                 "container": {"padding": "5!important", "background-color": "#000000"},
                                 "icon": {"color": "orange", "font-size": "25px"},
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                              "--hover-color": "#eee"},
                                 "nav-link-selected": {"background-color": "#02ab21"},
                             }
                             )
    if choose == "About":
        col1, col2= st.columns([0.8,0.1])
        with col2:
            st.image(logo, width=130)
        with col1: # display general information about the sociaty
            st.markdown(""" <style> .font {
                            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
                            </style> """, unsafe_allow_html=True)
            st.markdown("<p class='font'> Concept Les Freres is a programming and IT services company.",unsafe_allow_html=True)
            st.markdown("<i class='fa-solid fa-user'>It is specialized in:</i>",unsafe_allow_html=True)
            st.markdown("<i>Training of new modules.</i>",unsafe_allow_html=True)
            st.markdown("<i>Consulting and development of new technology IT solutions.</i>",unsafe_allow_html=True)
            st.markdown("<i>It recommends to its customer’s scalable projects locally as well asinternationally, meets all their expectations by</i>",unsafe_allow_html=True)
    if choose == "Image":

        
        col1, col2  ,col3 = st.columns([2,3,1])
        var = 0
        with col1:  # To display the header text using css style
            st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">defect detction </p>', unsafe_allow_html=True)
        with col3:  # To display brand log
            st.image(logo, width=130)
        st.subheader("Image")
        # file = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        file_uploaded = st.file_uploader("Upload an image", accept_multiple_files=True)
        match len(file_uploaded):
            case 1:
                col4 ,col5 ,col6 = st.columns([10,10,10])
                if file_uploaded[0] is not None:
                    image = Image.open(file_uploaded[0])
                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image
                        pred = model.predict(def_img)
                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area = [x, y, 85 + width, height - 40]

                    with col4:
                        image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)
            case 2:
                col4, col5  ,col6 = st.columns([2,2,2])
                if file_uploaded[0] is not None:
                    image = Image.open(file_uploaded[0])
                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image

                        pred = model.predict(def_img)

                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area =[x, y, 85 + width,  height-40]

                    with col4:
                        image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else :
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)

                if file_uploaded[1] is not None:
                    image = Image.open(file_uploaded[1])

                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path(
                        "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image

                        pred = model.predict(def_img)

                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area = [x, y, 85 + width, height - 40]

                    with col5:
                        image = Image.open(
                            'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)


            case 3:
                col4, col5, col6 = st.columns([2,2,2])
                if file_uploaded[0] is not None:
                    image = Image.open(file_uploaded[0])
                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path(
                        "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image

                        pred = model.predict(def_img)

                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area = [x, y, 85 + width, height - 40]

                    with col4:
                        image = Image.open(
                            'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)
                if file_uploaded[1] is not None:
                    image = Image.open(file_uploaded[1])
                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path(
                        "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image

                        pred = model.predict(def_img)

                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area = [x, y, 85 + width, height - 40]

                    with col5:
                        image = Image.open(
                            'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)
                if file_uploaded[2] is not None:
                    image = Image.open(file_uploaded[2])
                    img = tf.image.resize(image, size=[224, 224])
                    img = img / 255.
                    def_img = tf.expand_dims(img, axis=0)
                    data_dir = pathlib.Path(
                        "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image

                        pred = model.predict(def_img)

                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(image, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            var = 2
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            var = 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                        plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    width, height = 224, 224
                    x, y = 80, 100
                    area = [x, y, 85 + width, height - 40]

                    with col6:
                        image = Image.open(
                            'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        cropped = image.crop(area)
                        cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        img = Image.open(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                        rows = [st.container()]
                        if var == 1:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        else:
                            link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        imgs = Image.open(link)
                        rows[0].image(imgs)
                        st.image(img)

            case 4:
                col4, col5 ,col6 = st.columns([2,2,2])
                for i in range(2):
                    if file_uploaded[i] is not None:
                        image = Image.open(file_uploaded[i])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col4:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
                for i in range(2):
                    if file_uploaded[i+2] is not None:
                        image = Image.open(file_uploaded[i+2])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col5:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)

            case 5:
                col4, col5, col6 = st.columns([2,2,2])
                for i in range(2):
                    if file_uploaded[i] is not None:
                        image = Image.open(file_uploaded[i])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col4:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
                for i in range(2):
                    if file_uploaded[i + 2] is not None:
                        image = Image.open(file_uploaded[i + 2])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col5:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
                for i in range(1):
                    if file_uploaded[i + 4] is not None:
                        image = Image.open(file_uploaded[i + 4])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col6:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)

            case 6:
                col4, col5, col6 = st.columns([2, 2, 2])
                for i in range(2):
                    if file_uploaded[i] is not None:
                        image = Image.open(file_uploaded[i])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col4:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
                for i in range(2):
                    if file_uploaded[i + 2] is not None:
                        image = Image.open(file_uploaded[i + 2])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col5:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
                for i in range(2):
                    if file_uploaded[i + 4] is not None:
                        image = Image.open(file_uploaded[i + 4])
                        img = tf.image.resize(image, size=[224, 224])
                        img = img / 255.
                        def_img = tf.expand_dims(img, axis=0)
                        data_dir = pathlib.Path(
                            "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                        class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                        for i in range(1):
                            ax = plt.subplot(2, 2, i + 1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.grid(False)
                            img2 = image

                            pred = model.predict(def_img)

                            pred_class = class_names[int(tf.round(pred)[0][0])]
                            ax2 = plt.imshow(image, cmap=plt.cm.binary)
                            if pred_class == 'def_front':
                                var = 2
                                for l in ax.spines:
                                    ax.spines[l].set_color('red')
                            else:
                                var = 1
                                for l in ax.spines:
                                    ax.spines[l].set_color('green')
                            plt.xlabel(f" predicted:{pred_class}")
                            plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        width, height = 224, 224
                        x, y = 80, 100
                        area = [x, y, 85 + width, height - 40]

                        with col6:
                            image = Image.open(
                                'C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                            cropped = image.crop(area)
                            cropped.save(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            img = Image.open(
                                "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                            rows = [st.container()]
                            if var == 1:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                            else:
                                link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                            imgs = Image.open(link)
                            rows[0].image(imgs)
                            st.image(img)
            case default:
                st.text("please select some images")
        file_uploaded = None
        # if len(file_uploaded) == 1:
        #     for file in file_uploaded:
        #         if file is not None:
        #             image = Image.open(file)
        #             col4, col5 ,col6= st.columns([2,2,2])
        #             img = tf.image.resize(image, size=[224, 224])
        #             img = img / 255.
        #             def_img = tf.expand_dims(img, axis=0)
        #             data_dir = pathlib.Path("C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
        #             class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
        #             for i in range(1):
        #                 ax = plt.subplot(2, 2, i + 1)
        #                 plt.xticks([])
        #                 plt.yticks([])
        #                 plt.grid(False)
        #                 img2 = image
        #
        #                 pred = model.predict(def_img)
        #
        #                 pred_class = class_names[int(tf.round(pred)[0][0])]
        #                 ax2 = plt.imshow(image, cmap=plt.cm.binary)
        #                 if pred_class == 'def_front':
        #                     var = 2
        #                     for l in ax.spines:
        #                         ax.spines[l].set_color('red')
        #                 else:
        #                     var = 1
        #                     for l in ax.spines:
        #                         ax.spines[l].set_color('green')
        #                 plt.xlabel(f" predicted:{pred_class}")
        #                 plt.savefig('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
        #             st.set_option('deprecation.showPyplotGlobalUse', False)
        #             width, height = 224, 224
        #             x, y = 80, 100
        #             area =[x, y, 85 + width,  height-40]
        #
        #             with col4:
        #                 image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
        #                 cropped = image.crop(area)
        #                 cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
        #                 img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
        #                 rows = [st.container()]
        #                 if var == 1:
        #                     link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
        #                 else :
        #                     link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
        #                 imgs = Image.open(link)
        #                 rows[0].image(imgs)
        #                 st.image(img)

                        # rows = [st.container()]
                        # if var == 1:
                        #     link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                        # else :
                        #     link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                        # imgs = Image.open(link)
                        # rows[0].image(imgs)
                        # st.image(img)
                        # st.markdown(""" <style> .font {
                        #                font-size:35px ; font-family: 'Cooper Black'; color: #FFFFFF;}
                        #                </style> """, unsafe_allow_html=True)
                    # with col5:
                    #     image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    #     cropped = image.crop(area)
                    #     cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                    #     img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                    #     rows = [st.container()]
                    #     if var == 1:
                    #         link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                    #     else :
                    #         link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                    #     imgs = Image.open(link)
                    #     rows[0].image(imgs)
                    #     st.image(img)
                    # #     # rows = [st.container()]
                    # #     # imgs = Image.open(link)
                    # #     # rows[0].image(imgs)
                    # #     # st.image(img)
                    # with col6:
                    #     container = st.container()
                    #     image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                    #     cropped = image.crop(area)
                    #     cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                    #     img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                    #     rows = [st.container()]
                    #     if var == 1:
                    #         link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg"
                    #     else :
                    #         link = "C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/red2.jpg"
                    #     imgs = Image.open(link)
                    #     rows[0].image(imgs)
                    #     st.image(img)
                    # with container:
                    #     color = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/green2.jpg")
                    #     st.image(color)
                    # rows2 = [st.container()]
                    # imgs = Image.open(link)
                    # rows2[0].image(imgs)
                    # st.image(img)
                # n_rows = 1 + 10 // int(10)
                # rows = [st.container() for _ in range(n_rows)]
                # cols_per_row = [r.columns(6) for r in rows]
                # cols = [column for row in cols_per_row for column in row]
                #
                # for image_index in range(6):
                #     cols[image_index].image(img)

                #     st.image(img)
                #
                #
                # with col2:
                #     image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                #     cropped = image.crop(area)
                #     cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     st.image(img)
                #     st.image(img)
                # with col2:
                #     image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                #     cropped = image.crop(area)
                #     cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     st.image(img)
                #
                # with col1:
                #     image = Image.open('C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/png.jpg')
                #     cropped = image.crop(area)
                #     cropped.save("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     img = Image.open("C:/Users/dell/PycharmProjects/Graphical_interface_project/saved_file/cropped.jpg")
                #     st.image(img)



    if choose == "real_time":
        # st.title("Webcam Live Feed")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            ##
            import time
            # Initialise variables to store current time difference as well as previous time call value
            previous = time.time()
            delta = 0
            # Keep looping
            while True:
                # Get the current time, increase delta and update the previous variable
                current = time.time()
                delta += current - previous
                previous = current
                # Check if 3 (or some other value) seconds passed
                if delta > 3:
                    # Operations on image
                    # Reset the time counter
                    delta = 0

                # Show the image and keep streaming
                _, img = camera.read()
                width, height = 670,240
                x, y = 0, 200
                area = (x, y, x + width, y + height)
                cv2.imwrite("C:/Users/dell/Desktop/image_to_save/img.jpg", img)
                image = Image.open("C:/Users/dell/Desktop/image_to_save/img.jpg")
                cropped = image.crop(area)
                cropped.save("C:/Users/dell/Desktop/image_to_save/save/cropped.jpg")

                st.image(cropped)
                # = image.resize(cropped, (224, 224))  # resize image to match model's expected sizing
                #cropped = cropped.reshape(1, 224, 224, 3)
                ## the result
                if cropped is not None:
                    image = tf.image.resize(cropped, (224,224))
                    image = image/255.
                    def_img = tf.expand_dims(image, axis=0)
                    data_dir = pathlib.Path(
                        "C:/Users/dell/PycharmProjects/Graphical_interface_project/data_set/serdine2/test")
                    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
                    for i in range(1):
                        ax = plt.subplot(2, 2, i + 1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        img2 = image
                        pred = model.predict(def_img)
                        pred_class = class_names[int(tf.round(pred)[0][0])]
                        ax2 = plt.imshow(img2, cmap=plt.cm.binary)
                        if pred_class == 'def_front':
                            number_of_defect += 1
                            for l in ax.spines:
                                ax.spines[l].set_color('red')
                        else:
                            number_of_good += 1
                            for l in ax.spines:
                                ax.spines[l].set_color('green')
                        plt.xlabel(f" predicted:{pred_class}")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.header("Its prediction")
                    st.pyplot(plt.show())


                ##
                cv2.imshow("Frame", img)
                cv2.waitKey(1)
            ##
        else:
            st.write('Stopped')

    if choose == "Statistics":
        col1 ,col2 = st.columns([3,1])
        import altair as alt
        with col1:
            energy_source = pd.DataFrame({
                "Class": ["Defected","Defected","Defected","Defected","Defected","Defected","Defected","Defected","Defected","Defected","Defected","Defected"],
                "Number": [600,100,200,400,140,200,420,83,323,657,34,897],
                "Date": ["2022-1-23","2022-2-23","2022-3-23","2022-4-23","2022-5-23","2022-6-23","2022-7-23","2022-8-23","2022-9-23","2022-10-23","2022-11-23","2022-12-23"]
            })
            "Defected can"
            bar_chart = alt.Chart(energy_source).mark_bar().encode(
                y="month(Date):O",
                x="Number:Q",
                color="Class:N"
            )
            st.altair_chart(bar_chart, use_container_width=True)


main()
