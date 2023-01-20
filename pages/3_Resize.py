import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import streamlit as st

demo_img = './media/resized_demo.jpg'
st.subheader("**Resize images/folder with images**")
st.image(demo_img)
st.caption("Demo resized image ^")

kpi1, kpi2 = st.columns(2)

with kpi1:
    fpath = st.text_input("Input file path to images", "./images/demo")
    img_width = st.number_input("Insert image width", max_value=1280, min_value=32, value=640, step=32)
    img_height = st.number_input("Insert image height", max_value=800, min_value=32, value=416, step=32)
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
resize = st.button("Resize Images")
if resize:
    for image in files:
        try:
            img = cv2.imread(os.path.join(fpath,image))
            resized = cv2.resize(img, (img_width,img_height))
            dstPath = join(dpath,image)
            cv2.imwrite(dstPath,resized)
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
        img_slider_2 = st.slider("Choose resized image", 0, img_len_2-1 , 1)
        img_name_2 = imglist_2[img_slider_2]
        show_img_2 = dpath + '/' + str(img_name_2)
        st.image(show_img_2)