import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from api import PRN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # gpu 0
prn = PRN(is_dlib = True, prefix="../")
from utils.render import render_texture
from utils.rotate_vertices import frontalize
from utils.estimate_pose import estimate_pose
import numpy as np


import pickle
import time
import os

def save_information(img, output):
    pos = prn.process(img) 
    vertices = prn.get_vertices(pos)
    cam_mat, pose, R = estimate_pose(vertices)

    dictionary = {'img': img, 'pos':pos, 'vertices': vertices,
                  'cam_mat': cam_mat, 'pose':pose, 'R': R}
    pickle.dump(dictionary,open(output, 'wb'))

def save_3d_info_sync(video_synthesis, output="./synthesis1"):
    if  not os.path.isdir(output):
        os.makedirs(output)

    obama_syn = cv2.VideoCapture(video_synthesis)
    x1 = 685
    y1 = 85
    x2 = 1250
    y2 = 650

    i = 0
    while True:
        i+=1
        # for j in range(3):
        ret, img_obama_syn = obama_syn.read()

        if not ret:
            print(f"Break at frame {i}")
            break
        
        img_obama_syn = cv2.resize(img_obama_syn, (x2-x1, y2-y1))
        save_information(img_obama_syn,os.path.join(output, f'{i}.pickle' ))
        print(f"Finish dump one file {output}/{i}.pickle")


def save_3d_info_HD(video_HD, output="./HD1-30fps"):
    if  not os.path.isdir(output):
        os.makedirs(output)

    obama_fullhd = cv2.VideoCapture(video_HD)
    x1 = 685
    y1 = 85
    x2 = 1250
    y2 = 650

    i = 0
    while True:
        i+=1
        # for j in range(3):
        ret1, img_obamahd = obama_fullhd.read()

        if not ret1 or i>= 43*30:  # thus frame obama change view
            print(f"Break at frame {i}")
            break
        
        obama_crop = img_obamahd[y1:y2, x1:x2]
        save_information(obama_crop,os.path.join(output, f'{i}.pickle' ))
        print(f"Finish dump one file {output}/{i}.pickle")


# save_3d_info_sync(video_synthesis="resultdeenglish.mp4")
save_3d_info_HD(video_HD="obama_fullhd.mp4")