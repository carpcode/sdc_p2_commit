'''
Project 2

Advanced Lane Finding Project

The goals / steps of this project are the following:

    [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    [] Apply a distortion correction to raw images.
    [] Use color transforms, gradients, etc., to create a thresholded binary image.
    [] Apply a perspective transform to rectify binary image ("birds-eye view").
    [] Detect lane pixels and fit to find the lane boundary.
    [] Determine the curvature of the lane and vehicle position with respect to center.
    [] Warp the detected lane boundaries back onto the original image.
    [] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


In the context of Project 2 this file serves to create the camera calibration information and save it to a pickle-file.

'''

# imports
import cv2
import matplotlib.image as mpimg
import numpy as np
import glob
import pickle

# Dimension of the calibration chessboard
nx = 9
ny = 5

# Chessboard (9x5) camera_cal/calibration[1..20].jpg
ims = glob.glob('camera_cal/calibration*.jpg')

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D poiints in img-plane

objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates


for im in ims:

    # read in each image
    img = mpimg.imread(im)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if (ret == True):
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

dist_pickle = {
    "objpoints": objpoints,
    "imgpoints": imgpoints
    }

outfile = open('wide_dist_pickle.p', 'wb')
pickle.dump(dist_pickle, outfile)
outfile.close()
