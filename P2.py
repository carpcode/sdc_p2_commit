'''
Project 2

Advanced Lane Finding Project

The goals / steps of this project are the following:

    [] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    [] Apply a distortion correction to raw images.
    [] Use color transforms, gradients, etc., to create a thresholded binary image.
    [] Apply a perspective transform to rectify binary image ("birds-eye view").
    [] Detect lane pixels and fit to find the lane boundary.
    [] Determine the curvature of the lane and vehicle position with respect to center.
    [] Warp the detected lane boundaries back onto the original image.
    [] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from moviepy.editor import VideoFileClip


# Global variables

dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Functions

def put_text(img_in,left_curverad,center_diff,side_pos):

    img_with_text = np.copy(img_in)

    cv2.putText(img_with_text,'Radius of Curvature = '+str(round(left_curverad,1))+' (m)', 
    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(img_with_text, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',
    (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    return img_with_text

# The pipeline

def process_image(img_in):

    result = np.copy(img_in)
    
    result = put_text(result,300.4789,0.1789230,'left')

    return result


# Core-Computation for /test_images

# Tst on images
if(0):
    images = glob.glob('./test_images/test*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img = process_image(img)
        write_name = './test_images/tracked'+str(idx+1)+'.jpg'
        cv2.imwrite(write_name,img)

# Test on videos
if(1):
    Output_video = 'output1_tracked.mp4'
    Input_video = 'project_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(Input_video)
    video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    video_clip.write_videofile(Output_video, audio=False)


