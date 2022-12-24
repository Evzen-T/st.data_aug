import cv2
import streamlit as st

import random
import os,glob
from os.path import isfile,join
from os import listdir,makedirs

demo_img = './media/overlayed_demo.jpg'
st.subheader("**Overlay images**")
st.image(demo_img)
st.caption("Demo overlayed image ^")

def image_overlay(img1, img2, location):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    x, y = location
    img1[y:y+h2, x:x+w2] = img2
    return img1


backgrd_img = st.text_input("Insert background image", './images/overlay/original_1/og_demo_1.jpg')
st.image(backgrd_img)

fpath = st.text_input("Insert overlaying image folder", "./images/overlay/original_2")
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

dpath = st.text_input("Insert destination path", "./images/overlay/augmented")
overlay = st.button("Overlay Images")

opened_img = cv2.imread(show_img_1)

max_width = opened_img.shape[1]
xlocation = random.randint(0, max_width)

max_height = opened_img.shape[0]
ylocation = random.randint(0, max_height)

try:
    makedirs(dpath)
except:
    print ("Directory already exist, images will be written in same folder")

files = list(filter(lambda f: isfile(join(fpath,f)), listdir(fpath)))
if overlay:
    for image in files:
        try:
            img1 = cv2.imread(backgrd_img, cv2.IMREAD_COLOR)
            img2 = cv2.imread(os.path.join(fpath,image), cv2.IMREAD_COLOR)
            overlayed = image_overlay(img1, img2, location=(int(xlocation),int(ylocation)))
            dstPath = join(dpath,image)
            cv2.imwrite(dstPath, overlayed)
        except:
            print ("{} is not converted".format(image))
    for fil in glob.glob("*.jpg"):
        try:
            image = cv2.imread(fil) 
            overlayed_image = image_overlay(img1, img2, location=(int(xlocation),int(ylocation)))
            cv2.imwrite(os.path.join(dpath,fil),overlayed_image)
        except:
            print('{} is not converted')

kpi1, kpi2 = st.columns(2)
with kpi1:
    backgrd_img = st.text_input("Insert background image", './images/overlay/original/og_demo_2.jpg')
    st.image(backgrd_img)

with kpi2:
    overlaying_img = st.text_input("Insert overlaying image", "./images/overlay/original/og_demo_1.jpg")
    st.image(overlaying_img)
    ylocation = st.number_input("Insert top left y coordinates", min_value=0, max_value=max_height, value=100, step=10)

    if img_len_2 == 0:
        show_img_2 = demo_img
        st.image(show_img_2)
    elif img_len_2 == 1:
        img_name_2 = imglist_2[0]
        show_img_2 = dpath + '/' + str(img_name_2)
        st.image(show_img_2)
    else:
        img_slider_2 = st.slider("Overlayed image", 0, img_len_2-1 , 1)
        img_name_2 = imglist_2[img_slider_2]
        show_img_2 = dpath + '/' + str(img_name_2)
        st.image(show_img_2)