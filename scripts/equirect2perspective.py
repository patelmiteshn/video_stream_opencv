"""
Created on Wed Oct 24

@author: Chelhwon Kim

extracting perspective photos from equirectangular image
"""

import numpy as np
import math
import argparse
import cv2

def sph2equi(x,        # (3, -1) array
             width,    # width of equirectangular image
             height):
    # length of the projection vector of x onto x-y plane
    length = np.linalg.norm(x[:2,:], axis=0)

    elevation = np.arctan2(x[2,:], length)
    azimuth = np.arctan2(x[1,:], x[0,:])

    # normalization to 0~1 and to 0~width-1
    azimuth = (azimuth / math.pi + 1.0)*0.5 * (width-1)
        
    elevation /= math.pi  # -1~1
    elevation[elevation > 0.5] = 1.0 - elevation[elevation > 0.5]
    elevation[elevation < -0.5] = -1.0 - elevation[elevation < -0.5]
    elevation = (height-1)*0.5*(-elevation + 1.0)   # 0~height-1
    
    pt = np.stack([azimuth, elevation], axis=0)

    return pt # (2, -1) array  [0~width-1], [0~height-1]

def cameraOrientation(yaw, pitch, roll):
    R_s_c0 = np.array([[0, 0, 1], 
                       [1, 0, 0],
                       [0, -1, 0]], 
                       dtype=np.float32)

    R_c0_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                         [0, 1, 0],
                         [-math.sin(yaw), 0, math.cos(yaw)]], 
                         dtype=np.float32)
    
    R_yaw_pitch = np.array([[1, 0, 0],
                            [0, math.cos(pitch), -math.sin(pitch)],
                            [0, math.sin(pitch), math.cos(pitch)]], 
                            dtype=np.float32)
    
    R_pitch_roll = np.array([[math.cos(roll), -math.sin(roll), 0],
                             [math.sin(roll), math.cos(roll), 0],
                             [0, 0, 1]], 
                             dtype=np.float32)
    R_s_c = R_s_c0
    R_s_c = np.matmul(R_s_c, R_c0_yaw)
    R_s_c = np.matmul(R_s_c, R_yaw_pitch)
    R_s_c = np.matmul(R_s_c, R_pitch_roll)

    return R_s_c

def interpolation(src,  # (H, W, 3) image
                  pt     # (H_, W_, 2) 2d point 
                 ):    
    pt_int = pt.astype(np.int32)
    w = pt - pt_int

    wx = w[:,:,0]
    wy = w[:,:,1]

    pt_x = pt_int[:,:,0]
    pt_y = pt_int[:,:,1]

    src_pad = np.pad(src, ((0,1),(0,1),(0,0)), mode='edge')
    lt = src_pad[pt_y, pt_x,:]
    rt = src_pad[pt_y, pt_x+1,:]
    lb = src_pad[pt_y+1, pt_x,:]
    rb = src_pad[pt_y+1, pt_x+1,:]

    wx = np.expand_dims(wx, axis=-1)
    wy = np.expand_dims(wy, axis=-1)
    
    out = (1-wx)*(1-wy)*lt + wx*(1-wy)*rt + (1-wx)*wy*lb + wx*wy*rb

    return out  # (H_, W_, 3) array

## conver equirectangular image to perspective image
def equirect2perspective(src,      # equirectangular image
                         K,        # camera intrinsic matrix
                         R_s_c,    # camera orientation in 3x3 matrix
                         width,    # width of perspective image
                         height):  # height of perspective image

    Kinv = np.linalg.inv(K)

    dst = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    x_range = np.arange(width)
    y_range = np.arange(height)
    x_mat = np.matmul(np.ones((height, 1)), x_range.reshape((1, width)))
    y_mat = np.matmul(y_range.reshape((height, 1)), np.ones((1, width)))
    x = np.stack([x_mat, y_mat, np.ones((height, width))], axis=0)  
    
    x = np.reshape(x, (3, -1))
    X_c = np.matmul(Kinv, x)
    X_s = np.matmul(R_s_c, X_c)
    norm = np.linalg.norm(X_s, axis=0)
    norm = np.reshape(norm, (1, -1))
    X_s = X_s / norm

    pt = sph2equi(X_s, src.shape[1], src.shape[0])   # (2, H*W)
    pt = np.reshape(pt, (2, height, width))  # (2, H, W)
    pt = np.transpose(pt, axes=(1,2,0))  # (H, W, 2)

    dst = interpolation(src, pt)
    dst = dst.astype(np.uint8)

    return dst


def run(src, dst_width, dst_height, fx, fy, yaw, pitch, roll):
    K = np.array([ [fx, 0, dst_width/2.0],
                   [0, fy, dst_height/2.0],
                   [0, 0, 1]
                 ], 
                 dtype=np.float32)

    R_s_c = cameraOrientation(yaw/180.0*math.pi,
                              pitch/180.0*math.pi,
                              roll/180.0*math.pi)

    dst = equirect2perspective(src, K, R_s_c, dst_width, dst_height)
    return dst
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--equirect_img', 
                        type=str,
                        help="eqirectangular image")
    parser.add_argument('--dst_width', default=320, type=int, help='width of output image')
    parser.add_argument('--dst_height', default=240, type=int, help='height of output image')
    parser.add_argument('--fx', default=250, help='focal length x in pixel')
    parser.add_argument('--fy', default=250, help='focal length y in pixel')
    parser.add_argument('--y', default=0, type=float, help='yaw in degree')
    parser.add_argument('--p', default=0, type=float, help='pitch in degree')
    parser.add_argument('--r', default=0, type=float, help='roll in degree')
    parser.add_argument('--dest_img', default='perspective.jpg', type=str, help='destination image name')


    args = parser.parse_args()

    src = cv2.imread(args.equirect_img)
    dst = run(src, args.dst_width, args.dst_height, args.fx, args.fy, args.y, args.p, args.r)

    cv2.imwrite(args.dest_img, dst, [cv2.IMWRITE_PNG_COMPRESSION, 0])

