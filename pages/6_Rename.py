import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import streamlit as st

demo_img = './media/renamed_demo.jpg'
st.subheader("**Renamed images(s)**")
st.image(demo_img)
st.caption("Demo renamed image ^")

kpi1, kpi2 = st.columns(2)
with kpi1:
    fpath = st.text_input("Input file path to images", "./images/demo")
    for f in fpath:
        imglist_1 = listdir(fpath)
        img_len_1 = len(imglist_1)

if img_len_1 == 0 and fpath == './images/demo':
    show_img = demo_img
    with kpi1:
        st.image(show_img)
else:
    img_slider_1 = st.slider("Preview original image", 1, img_len_1-1 , 1)
    with kpi1:
        if fpath == './images/demo':
            img_name_1 = imglist_1[img_slider_1]
        else:
            img_name_1 = 'Frame_{}.jpg'.format(img_slider_1)
        show_img_1 = fpath + '/' + str(img_name_1)
    st.image(show_img_1)
    st.caption('Img: {}'.format(show_img_1))
    img_num_1 = st.number_input("Insert starting number", min_value=0, step=1)

    with kpi2:
        dpath = st.text_input("Input destination path", "./images/augmented")

try:
    makedirs(dpath)
except:
    print ("Directory already exist, images will be written in same folder")

files = list(filter(lambda f: isfile(join(fpath,f)), listdir(fpath)))
rename = st.button("Rename Images")
if rename:
    for i in range(0,img_len_1):
        try:
            print(fpath)
            img = cv2.imread(os.path.join(fpath,'Frame_{}.jpg'.format(i)))
            dstPath = join(dpath, 'Frame_{}.jpg'.format(img_num_1))
            cv2.imwrite(dstPath,img)
            img_num_1+=1
        except:
            print ("{} is not converted".format('Frame_{}'.format(i)))

for d in dpath:
    imglist_2 = listdir(dpath)
    img_len_2 = len(imglist_2)

if img_len_2 == 0:
    show_img_2 = demo_img
    st.image(show_img_2)
    st.caption('Demo image')
else:
    img_slider_2 = st.slider("Choose renamed image", 0, img_len_2-1 , 1)
    img_name_2 = imglist_2[img_slider_2]
    show_img_2 = dpath + '/' + str(img_name_2)
    st.image(show_img_2)
    st.caption('Img: {}'.format(show_img_2))