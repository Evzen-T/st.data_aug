import cv2
import streamlit as st

def image_overlay(img1, img2, location):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    x, y = location
    img1[y:y+h2, x:x+w2] = img2
    return img1

backgrd_img = st.text_input("Insert background image", './images/overlay/original/image1.jpg')
overlaying_img = st.text_input("Insert overlaying image", "./images/overlay/original/image2.jpg")

if backgrd_img == None:
    st.error("Background image cannot be found")

img1 = cv2.imread(backgrd_img, cv2.IMREAD_COLOR)
img2 = cv2.imread(overlaying_img, cv2.IMREAD_COLOR)
overlayed = image_overlay(img1, img2, location=(100,100))
st.image('Overlayed img', overlayed)

filename='overlayed_2.jpg'
cv2.imwrite(filename, overlayed)
st.success("Image successfully saved")