import os
import cv2
import random
import numpy as np
import streamlit as st
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

cd = os.getcwd()
st.caption('Current Working Directory (CWD)')
st.code(cd)

img         = st.text_input('Insert image filepath',f'{cd}/dataset/images')
dir         = os.path.dirname(img)
labels      = f'{dir}/labels'
crop        = f'{dir}/1_cropped'
cpts        = f'{dir}/1_cropped_pts'
augment     = f'{dir}/2_augment'
apts        = f'{dir}/2_augment_pts'
paste       = f'{dir}/3_paste'
backgrd     = f'{dir}/3_backgrd'
segment     = f'{dir}/4_segment'
data        = f'{dir}/data.txt'
names       = f'{dir}/names.txt'
ctr         = 0
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

def read(img_file: str, label_file: str, img_name: str, normalize: bool = False, cropped: bool = False):
    #Path
    img_path    = os.path.join(img_file, f'{img_name}.jpg')
    txt_path    = os.path.join(label_file, f'{img_name}.txt')
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
    if normalize is True:
        polygon[:,0] = polygon[:,0]*img_w
        polygon[:,1] = polygon[:,1]*img_h
    if cropped is True:
        polygon[:,0] = polygon[:,0]
        polygon[:,1] = polygon[:,1]
    return image, polygon.astype(np.int32), cls_idx
    

def seg_ann(img, polygon, alpha: float = 0.7, label_file=None, name=None, save_dir=None, classes=None, color=None):
    #Path
    txt_path    = os.path.join(label_file, f'{name}.txt')
    save_path   = os.path.join(save_dir, f'{name}.jpg')
    
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
    cv2.fillPoly(img, pts=[polygon], color=color[idx])
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, f'{classes[idx]}', org =(polygon[0][0], polygon[0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color[idx], thickness=2)
    cv2.polylines(img, pts=[polygon], isClosed=True,color=color[idx],thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite(save_path,img)
    txt_file.close()
    return

def crop_seg(pts, img, name, save_dir, spts_dir, idx):
    #Path
    crop_path = os.path.join(save_dir, f'{name}.jpg')
    cpts_path = os.path.join(spts_dir, f'{name}.txt')
    
    #Array pts
    arr_poly = np.array(pts)
    ##Crop the bounding rect
    rect = cv2.boundingRect(arr_poly)
    x,y,w,h = rect
    crop_img = img[y:y+h, x:x+w].copy()

    #Save cropped points
    offset = np.zeros((1,2))
    offset[0][0] = x
    offset[0][1] = y
    arr_poly_offset = arr_poly - offset
    crop_idx = f'{idx}'

    with open(cpts_path, 'w') as f:
        f.write(crop_idx)
        for i in range(arr_poly_offset.shape[0]):
            f.write(' ')
            f.write(f'{int(arr_poly_offset[i][0])}')
            f.write(' ')
            f.write(f'{int(arr_poly_offset[i][1])}')
        
    cv2.imwrite(crop_path, crop_img)
    return crop_img

def aug_crop(img, pts, name, save_dir, pts_dir, spts_dir, idx, color):
    #Path
    aug_path = os.path.join(save_dir, f'{name}.jpg')
    cpts_path = os.path.join(pts_dir, f'{name}.txt')
    spts_path = os.path.join(spts_dir, f'{name}.txt')
    ia.seed(1)
    poly_pts = Polygon(pts)
    psoi = PolygonsOnImage([poly_pts],shape=img.shape)
    sequential = iaa.Sequential(
        (iaa.Affine(rotate=(-20,20), fit_output=True))
    )
    
    augment = sequential(
                        images=[img],
                        polygons=psoi, return_batch=True)

    aug_img = augment.images_aug
    psoi_img = augment.polygons_aug
    psoi_pts = np.array(psoi_img.to_xy_array()).flatten()
    if os.path.exists(cpts_path):
        txt_file = open(cpts_path)
    else:
        print('No txt file detected')
        
    for info in txt_file:
        idx_info = info.split()
        idx = int(idx_info[0])
    
    with open(spts_path, 'w') as f:
        f.write(f'{idx}')
        for i in range(psoi_pts.size):
            f.write(' ')
            f.write(f'{psoi_pts[i]}')

    drawn = psoi_img.draw_on_image(aug_img[0],alpha_face=0.2, size_points=3, color=color[idx])
    cv2.imwrite(aug_path, drawn)
    return drawn, psoi_pts

def yolo_paste(src, dst, src_pts, location, name, simg_dir):
    #Path
    save_path   = os.path.join(simg_dir, f'{name}.jpg')
    in1_img     = os.path.join(src, f'{name}.jpg')
    in2_img     = os.path.join(dst, f'1.jpg')
    read_1      = cv2.imread(in1_img)
    read_2      = cv2.imread(in2_img)

    h1, w1      = read_1.shape[:2]
    h2, w2      = read_2.shape[:2]
    x, y        = location
    roi         = read_1[y:y+h1, x:x+w1]

    gray        = cv2.cvtColor(read_2, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)

    img_bg      = cv2.bitwise_and(roi,roi, mask=mask)
    img_fg      = cv2.bitwise_and(read_2, read_2, mask=mask_inv)
    dst         = cv2.add(img_bg, img_fg)

    read_1[y:y+h1, x:x+w1] = dst
    cv2.imwrite(save_path, read_1)
    return read_2

def image_overlay(location, simg_dir, name, src1, src2):
    save_path   = os.path.join(simg_dir, f'{name}.jpg')
    in1_img     = os.path.join(src1, f'{name}.jpg')
    in2_img     = os.path.join(src2, f'1.jpg')
    read_1      = cv2.imread(in1_img)
    read_2      = cv2.imread(in2_img)
    h1,w1       = read_1.shape[:2]
    h2,w2       = read_2.shape[:2]
    x, y        = location
    
    read_2[y:y+h1, x:x+w1] = read_1
    cv2.imwrite(save_path, read_2)
    return read_2

aug_seg= st.button('Paste segments')
if aug_seg:
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
        image, polygon, cls_idx = read(img_file=img, label_file=labels, img_name=img_names, normalize=True, cropped=False)
        crop_img                = crop_seg(img=image, pts=polygon, name=img_names, save_dir=crop, spts_dir=cpts, idx=cls_idx)
        image, polygon, cls_idx = read(img_file=crop, label_file=cpts, img_name=img_names, normalize=False, cropped=True)
        aug_img, aug_pts        = aug_crop(img=image, pts=polygon, name=img_names, save_dir=augment, pts_dir=cpts, spts_dir=apts, idx=cls_idx, color=color_list)
        image, polygon, cls_idx = read(img_file=augment, label_file=apts, img_name=img_names, normalize=True, cropped=False)
        paste_img               = yolo_paste(src=augment, dst=backgrd, src_pts=polygon, location=(100,200), name=img_names, simg_dir=paste)
        # seg_ann(img=image, polygon=polygon, label_file=cpts, name=img_names, save_dir=segment, classes=classes, color=color_list)

st.markdown('---')
image_names = open(names).read().strip().split()
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    prev_img = st.button('⬅️ Prev', on_click=decrement_counter)
    
with kpi2:
    next_img = st.button('Next ➡️', on_click=increment_counter)

with kpi3:
    chosen = st.selectbox('Select save files', ['crop', 'augment', 'segment'])
    
if chosen == 'crop':
    if prev_img and st.session_state.count >= 0:
        img_name = f'{crop}/{str(image_names[st.session_state.count])}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    elif next_img and st.session_state.count <= len(os.listdir(crop)):
        img_name = f'{crop}/{image_names[st.session_state.count]}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    else:
        st.session_state.count=0

if chosen == 'augment':
    if prev_img and st.session_state.count >= 0:
        img_name = f'{augment}/{str(image_names[st.session_state.count])}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    elif next_img and st.session_state.count <= len(os.listdir(augment)):
        img_name = f'{augment}/{image_names[st.session_state.count]}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    else:
        st.session_state.count=0

if chosen == 'segment':
    if prev_img and st.session_state.count >= 0:
        img_name = f'{segment}/{str(image_names[st.session_state.count])}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    elif next_img and st.session_state.count <= len(os.listdir(segment)):
        img_name = f'{segment}/{image_names[st.session_state.count]}.jpg'
        img_name_2 = cv2.imread(img_name)
        st.image(img_name_2)
        st.caption(f'image {st.session_state.count}')
        curr = st.session_state.count

    else:
        st.session_state.count=0