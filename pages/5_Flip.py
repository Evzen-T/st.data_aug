import streamlit as st
import cv2

import os,glob
from os import listdir,makedirs
from os.path import isfile,join

demo_img = './media/flipped_demo.jpg'
st.subheader("**Flipping images/folder with images**")
st.image(demo_img)
st.caption("Demo flipped image ^")

kpi1, kpi2 = st.columns(2)

with kpi1:
    fpath = st.text_input("Input file path to images", "./images/demo/")
    for f in fpath:
        imglist_1 = listdir(fpath)
        img_len_1 = len(imglist_1)

    if img_len_1 == 0:
        show_img = demo_img
        st.image(show_img)
    else:
        img_slider_1 = st.slider("Preview original image", 0, img_len_1-1 , 1)
        img_name_1 = imglist_1[img_slider_1]
        show_img_1 = fpath + '/' + str(img_name_1)
        st.image(show_img_1)

with kpi2:
    dpath = st.text_input("Input destination path", "./images/augmented")

try:
    makedirs(dpath)
except:
    print ("Directory already exist, images will be written in same folder")

files = list(filter(lambda f: isfile(join(fpath,f)), listdir(fpath)))
flip = st.button("Flip Images")
c_flip = st.sidebar.selectbox("Choose flip orientation", ["Flip Horizontally", "Flip Vertically", "Flip H & V"])
if c_flip == "Flip Horizontally":
    if flip:
        for image in files:
            try:
                img = cv2.imread(os.path.join(fpath,image))
                flipping = cv2.flip(img,1)
                dstPath = join(dpath,image)
                cv2.imwrite(dstPath,flipping)
            except:
                print ("{} is not converted".format(image))

if c_flip == "Flip Vertically":
    if flip:
        for image in files:
            try:
                img = cv2.imread(os.path.join(fpath,image))
                flipping = cv2.flip(img,0)
                dstPath = join(dpath,image)
                cv2.imwrite(dstPath,flipping)
            except:
                print ("{} is not converted".format(image))

if c_flip == "Flip H & V":
    if flip:
        for image in files:
            try:
                img = cv2.imread(os.path.join(fpath,image))
                flipping = cv2.flip(img,-1)
                dstPath = join(dpath,image)
                cv2.imwrite(dstPath,flipping)
            except:
                print ("{} is not converted".format(image))

with kpi2:
    for d in dpath:
        imglist_2 = listdir(dpath)
        img_len_2 = len(imglist_2)

    if img_len_2 == 0:
        show_img_2 = demo_img
        st.image(show_img_2)
    else:
        img_slider_2 = st.slider("Choose flipped image", 0, img_len_2-1 , 1)
        img_name_2 = imglist_2[img_slider_2]
        show_img_2 = dpath + '/' + str(img_name_2)
        st.image(show_img_2)