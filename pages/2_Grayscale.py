import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import streamlit as st

demo_img = './media/grayed_demo.jpg'
st.subheader("**Grayscale images/folder with images**")
st.image(demo_img)
st.caption("Demo grayscale image ^")

kpi1, kpi2 = st.columns(2)

with kpi1:
    fpath = st.text_input("Input file path to images", "./images/gray/original")
    for f in fpath:
        imglist_1 = listdir(fpath)
        img_len_1 = len(imglist_1)

    if img_len_1 == 0:
        show_img = demo_img
        st.image(show_img)
    else:
        img_slider_1 = st.slider("Preview colour image", 0, img_len_1-1 , 1)
        img_name_1 = imglist_1[img_slider_1]
        show_img_1 = fpath + '/' + str(img_name_1)
        st.image(show_img_1)

with kpi2:
    dpath = st.text_input("Input destination path", "./images/gray/augmented")

try:
    makedirs(dpath)
except:
    print ("Directory already exist, images will be written in same folder")

files = list(filter(lambda f: isfile(join(fpath,f)), listdir(fpath)))
gray = st.button("Grayscale Images")
if gray:
    for image in files:
        try:
            img = cv2.imread(os.path.join(fpath,image))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            dstPath = join(dpath,image)
            cv2.imwrite(dstPath,gray)
        except:
            print ("{} is not converted".format(image))
    for fil in glob.glob("*.jpg"):
        try:
            image = cv2.imread(fil) 
            gray_image = cv2.cvtColor(os.path.join(fpath,image), cv2.COLOR_BGR2GRAY) # convert to greyscale
            cv2.imwrite(os.path.join(dpath,fil),gray_image)
        except:
            print('{} is not converted')

with kpi2:
    for d in dpath:
        imglist_2 = listdir(dpath)
        img_len_2 = len(imglist_2)

    if img_len_2 == 0:
        show_img_2 = demo_img
        st.image(show_img_2)
    else:
        img_slider_2 = st.slider("Choose grayed image", 0, img_len_2-1 , 1)
        img_name_2 = imglist_2[img_slider_2]
        show_img_2 = dpath + '/' + str(img_name_2)
        st.image(show_img_2)