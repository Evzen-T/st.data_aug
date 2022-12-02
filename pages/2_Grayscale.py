import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join
import streamlit as st

demo_img = './media/grayed_demo.jpg'

kpi1, kpi2 = st.columns(2)

with kpi1:
    fpath = st.text_input("Input file path to images", "./images/gray/original")

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

show_img = dpath + '/' + 'overlayed_1.jpg'
st.image(demo_img)
length = len(listdir(dpath))
st.slider("Choose grayed image", 0, length, 1)