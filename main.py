import cv2
import os
import numpy as np
import math
if not os.path.isdir('test'):
    os.makedirs('test')

import pygame
from OpenGL.GL import *
from model import OBJ
from utils import projection_matrix, render

obj = OBJ("low-poly-fox-by-pixelmannen.obj", swapyz=True)
camera_parameters = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
MIN_MATCHES = 10

model = cv2.imread('model2.jpg', 0)
orb = cv2.ORB_create() # ORB keypoint detector
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)

cam = cv2.VideoCapture(0)
while True:
    ret_val, img_rgb = cam.read()
    cap = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > MIN_MATCHES:
        # draw first 15 matches.
        cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                              matches[:MIN_MATCHES], 0, flags=2)
        # # show result
        # cv2.imshow('frame', cap)
        # cv2.waitKey(0)

        # assuming matches stores the matches found and
        # returned by bf.match(des_model, des_frame)
        # differenciate between source points and destination points
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # compute Homography
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)  
        # connect them with lines
        cap = cv2.polylines(cap, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        projection = projection_matrix(camera_parameters, homography)

        img_rgb = render(img_rgb, obj, projection, model, False)
    else:
        print("Not enough matches have been found - %d/%d" % (len(matches),
                                                            MIN_MATCHES))
    cv2.imshow('frame', cap)
    cv2.imshow('webcam', img_rgb)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit
cv2.destroyAllWindows()
