import cv2
import streamlit as st

demo_img = './media/grayed_demo.jpg'
st.subheader("**Overlay images**")
st.image(demo_img)
st.caption("Demo overlayed image ^")

def image_overlay(img1, img2, location):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    x, y = location
    img1[y:y+h2, x:x+w2] = img2
    return img1

kpi1, kpi2 = st.columns(2)
with kpi1:
    backgrd_img = st.text_input("Insert background image", './images/overlay/original/og_demo_2.jpg')
    st.image(backgrd_img)
    read = cv2.imread(backgrd_img, cv2.IMREAD_UNCHANGED)
    img_dims = read.shape
    max_height = img_dims[0]
    max_width = img_dims[1]
    xlocation = st.number_input("Insert top left x coordinates", min_value=0, max_value=max_width, value=100, step=10)

with kpi2:
    overlaying_img = st.text_input("Insert overlaying image", "./images/overlay/original/og_demo_1.jpg")
    st.image(overlaying_img)
    ylocation = st.number_input("Insert top left y coordinates", min_value=0, max_value=max_height, value=100, step=10)

if 'count' not in st.session_state:
    st.session_state.count = 0
def increment_counter():
    st.session_state.count += 1

previous = 0
overlay = st.button("Overlay Images", on_click=increment_counter)

if overlay and st.session_state.count > previous:
    img1 = cv2.imread(backgrd_img, cv2.IMREAD_COLOR)
    img2 = cv2.imread(overlaying_img, cv2.IMREAD_COLOR)
    overlayed = image_overlay(img1, img2, location=(int(xlocation),int(ylocation)))
    overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    st.image(overlayed_rgb)

    filename='./images/overlay/augmented/overlayed_{}.jpg'.format(st.session_state.count)
    cv2.imwrite(filename, overlayed)
    st.success("Image successfully saved")
    previous = st.session_state.count