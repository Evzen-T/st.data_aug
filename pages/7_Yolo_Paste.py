import os
import cv2
import random
import numpy as np
import streamlit as st
from PIL import Image

cd = os.getcwd()
st.caption('Current Working Directory (CWD)')
st.code(cd)

img         = st.text_input('Insert image filepath',f'{cd}/dataset/images')
dir         = os.path.dirname(img)
labels      = f'{dir}/labels'
save        = f'{dir}/saved'
save_seg    = f'{dir}/saved_seg'
data        = f'{dir}/data.txt'
names       = f'{dir}/names.txt'
curr        = 0
stframe     = st.empty()

if 'count' not in st.session_state:
    st.session_state.count = 0

def increment_counter():
    st.session_state.count+=1

def decrement_counter():
    st.session_state.count-=1

def name_list(img_file, list_path):
    #Path
    img_list = os.listdir(img_file)
    name_list = open(list_path, 'w')

    #Write list
    for file_name in img_list:
        name, extension = os.path.splitext(file_name)
        name_list.write(name+'\n')
    
    name_list.close()

def norm_img(img_file: str, labels_file: str, img_name: str, normalize: bool = False):
    #Path
    img_path    = os.path.join(img_file, f'{img_name}.jpg')
    txt_path    = os.path.join(labels_file, f'{img_name}.txt')
    # save_path   = os.path.join(save_dir, f'{img_names}.jpg')
    
    # read image
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]
  
    # read .txt file for this image
    with open(txt_path, "r") as f:
        txt_file = f.readlines()[0].split()
        cls_idx = txt_file[0]
        coords = txt_file[1:]
        polygon = np.array([[eval(x), eval(y)] for x, y in zip(coords[0::2], coords[1::2])]) # convert list of coordinates to numpy massive
  
    # Convert normilized coordinates of polygons to coordinates of image
    if normalize:
        polygon[:,0] = polygon[:,0]*img_w
        polygon[:,1] = polygon[:,1]*img_h
    return image, polygon.astype(np.int32)
    

def segmentation(img, polygon, alpha: float = 0.7, labels_file=None, names=None, save_dir=None, classes=None, colors=None):
    #Path
    txt_path    = os.path.join(labels_file, f'{names}.txt')
    save_path   = os.path.join(save_dir, f'{names}.jpg')
    
    # Create zero array for mask
    overlay = img.copy()
    
    if os.path.exists(txt_path):
        txt_file = open(txt_path)
    else:
        print('No txt file detected')
    
    for info in txt_file:
        idx_info = info.split()
        idx = int(idx_info[0])

    # Draw polygon on the image and mask
    cv2.fillPoly(img, pts=[polygon], color=colors[idx])
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, f'{classes[idx]}', org =(polygon[0][0], polygon[0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colors[idx], thickness=2)
    Ann_img  = cv2.polylines(img, pts=[polygon], isClosed=True,color=colors[idx],thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path,img)
    txt_file.close()
    return Ann_img

def crop_seg(pts, save_dir, names):
    #Path
    save_path = os.path.join(save_dir, f'{names}.jpg')
    
    #Array pts
    arr_poly = np.array(pts)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(arr_poly)
    x,y,w,h = rect
    crop_img = image[y:y+h, x:x+w].copy()
    
    cv2.imwrite(save_path, crop_img)
    return crop_img

annotate = st.button('Polygon segmentation')
if annotate:
    name_list(img, names)
    classes     = open(data).read().strip().split('\n')
    image_names = open(names).read().strip().split()

    color_list = []
    color = []
    for _ in range(len(classes)):
        for _ in range(3):
            color.append(random.randint(0,255))
        color_list.append(color)
        color = []

    for img_names in image_names:
        image, polygon  = norm_img(img_file=img, labels_file=labels, img_name=img_names, normalize=True)
        crop_img        = crop_seg(pts=polygon, save_dir=save, names=img_names)
        Ann_img         = segmentation(img=image, polygon=polygon, labels_file=labels, names=img_names, save_dir=save_seg, classes=classes, colors=color_list)

st.markdown('---')
image_names = open(names).read().strip().split()
kpi1, kpi2, kpi3= st.columns(3)
with kpi1:
    prev_img = st.button(':arrow_backward: Prev', on_click=decrement_counter)

with kpi2:
    chosen = st.selectbox('Select save files', ['save', 'save_seg'])
    

with kpi3:
    next_img = st.button('Next :arrow_forward:', on_click=increment_counter)

if chosen == 'save':
    if prev_img and st.session_state.count > 0:
        img_name = f'{save}/{str(image_names[st.session_state.count])}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    elif next_img and st.session_state.count <= len(os.listdir(save)):
        img_name = f'{save}/{image_names[st.session_state.count]}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    else:
        st.session_state.count=0

if chosen == 'save_seg':
    if prev_img and st.session_state.count > 0:
        img_name = f'{save_seg}/{str(image_names[st.session_state.count])}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    elif next_img and st.session_state.count <= len(os.listdir(save_seg)):
        img_name = f'{save_seg}/{image_names[st.session_state.count]}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    else:
        st.session_state.count=0