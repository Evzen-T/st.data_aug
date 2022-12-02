import cv2
import os,glob
from os import listdir,makedirs
from os.path import isfile,join

path = '/home/evzen/Yolo/BL_YoloV5/dataset/coloured/images' # Source Folder
dstpath = '/home/evzen/Yolo/BL_YoloV5/dataset/mono/images' # Destination Folder
try:
    makedirs(dstpath)
except:
    print ("Directory already exist, images will be written in same folder")
# Folder won't used
files = list(filter(lambda f: isfile(join(path,f)), listdir(path)))
for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        resized = cv2.resize(img, (640,640))
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,resized)
    except:
        print ("{} is not converted".format(image))
for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil) 
        resized_image = cv2.resize(os.path.join(path,image), (640,640)) # resize
        cv2.imwrite(os.path.join(dstpath,fil),resized_image)
    except:
        print('{} is not converted')