import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from api import PRN
from utils.render import render_texture
import cv2


def texture_editing(prn, args):
    # read image
    print(args.output_path)
    image = imread(args.image_path)
    [h, w, _] = image.shape

    #-- 1. 3d reconstruction -> get texture. 
    pos = prn.process(image) 
    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    cv2.imwrite("Texture.png", (texture*255).astype(int))
    



    #-- 2. Texture Editing
    Mode = args.mode
    # change part of texture(for data augumentation/selfie editing. Here modify eyes for example)
    if Mode == 0: 
        # load eye mask
        uv_face_eye = imread('Data/uv-data/uv_face_eyes.png', as_grey=True)/255. 
        uv_face = imread('Data/uv-data/uv_face.png', as_grey=True)/255.
        eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)

        # texture from another image or a processed texture
        ref_image = imread(args.ref_path)
        ref_pos = prn.process(ref_image)
        ref_image = ref_image/255.
        ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

        # modify texture
        new_texture = texture*(1 - eye_mask[:,:,np.newaxis]) + ref_texture*eye_mask[:,:,np.newaxis]
    
    # change whole face(face swap)
    elif Mode == 1: 
        # texture from another image or a processed texture
        ref_image = imread(args.ref_path)
        ref_pos = prn.process(ref_image)
        ref_image = ref_image/255.
        ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        ref_vertices = prn.get_vertices(ref_pos)
        new_texture = ref_texture#(texture + ref_texture)/2.

    else:
        print('Wrong Mode! Mode should be 0 or 1.')
        exit()


    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)
    
    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
   
    # save output
    imsave(args.output_path, output) 
    print('Done.')



def write_video_to_files():
    obama_syn = cv2.VideoCapture("/home/vuthede/AI/first-order-model/result2.mp4")
    obama_fullhd = cv2.VideoCapture("/home/vuthede/AI/first-order-model/obama_fullhd.mp4")
    x1 = 685
    y1 = 85
    x2 = 1250
    y2 = 650

    i = 0
    while True:
        i+=1
        for j in range(3):
            ret, img_obama_syn = obama_syn.read()
            ret1, img_obamahd = obama_fullhd.read()

        if not ret or not ret1:
            print(f"Break at frame {i}")
            break
        
        obama_crop = img_obamahd[y1:y2, x1:x2]
        img_obama_syn = cv2.resize(img_obama_syn, (x2-x1, y2-y1))


        cv2.imwrite(f"/home/vuthede/Desktop/3D/input/{i}.png", obama_crop)
        cv2.imwrite(f"/home/vuthede/Desktop/3D/ref/{i}.png", img_obama_syn)

def merge_frames():
    syn_dir = "/home/vuthede/Desktop/3D/input/"
    full_hd_dir = "/home/vuthede/Desktop/3D/ref/"
    warping_dir = "/home/vuthede/Desktop/3D/output_swap/"
    out = cv2.VideoWriter('/home/vuthede/Desktop/3D/out.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (565*3, 565))
    for i in range(1,150):
        im1 = cv2.imread(syn_dir+f"{i}.png")
        im2 = cv2.imread(full_hd_dir+f"{i}.png")
        im3 = cv2.imread(warping_dir+f"{i}.png")

        concat = np.hstack([im1, im2, im3])
        out.write(concat)

        print(concat.shape)
        cv2.imshow("Concat", concat)
        k = cv2.waitKey(0)

        if k ==27:
            break

    out.release()
        





if __name__ == '__main__':
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='Texture Editing by PRN')

    parser.add_argument('-i', '--image_path', default='TestImages/AFLW2000/image00081.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-r', '--ref_path', default='TestImages/trump.jpg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-o', '--output_path', default='TestImages/output.jpg', type=str, 
                        help='path to save output')
    parser.add_argument('--mode', default=1, type=int, 
                        help='ways to edit texture. 0 for modifying parts, 1 for changing whole')
    parser.add_argument('--gpu', default='0', type=str, 
                        help='set gpu id, -1 for CPU')

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = True) 

    args = parser.parse_args()
    texture_editing(prn,args)

    #write_video_to_files()
    # for i in tqdm(range(1, 150)): 
    #     args.image_path = f"/home/vuthede/Desktop/3D/input/{i}.png"
    #     args.ref_path = f"/home/vuthede/Desktop/3D/ref/{i}.png"
    #     args.output_path = f"/home/vuthede/Desktop/3D/output_swap/{i}.png"
    #     texture_editing(prn,args)

    #     print(f"Finish warping {i}-th frame")

    # merge_frames()