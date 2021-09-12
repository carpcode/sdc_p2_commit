'''
Project 2

Advanced Lane Finding Project

The goals / steps of this project are the following:

    [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    [x] Apply a distortion correction to raw images.
    [x] Use color transforms, gradients, etc., to create a thresholded binary image.
    [x] Apply a perspective transform to rectify binary image ("birds-eye view").
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

# Data for camera calibration, see P2_compute_camera_calibration.py
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Perspctive Transform - src, dst points to define the transformation

# Input help forum
src_line =  [[568,468],
     [715,468],
     [1040,680],
     [270,680]]

# Perspective Transform requires float, cv2.line requires int
src = np.float32(src_line)

dst_line = [[200,0],
      [1000,0],
      [1000,680],
      [200,680]]

dst = np.float32(dst_line)

# Define conversions in x and y from pixels space to meters
# tbd.align with transformation src_dist values
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Helper Functions

def calibrate_camera(img_in, objpoints, imgpoints):
    gray = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist_img = cv2.undistort(img_in, mtx, dist, None, mtx)
    return undist_img


def create_binary(img_in, s_thresh=(170, 255), sx_thresh=(20, 100)):
# Create a threshold binary image - Color Transform
    img = np.copy(img_in)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # convert to grayscale for Sobel-Operation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combing the binary thresholds in one, according to lesson
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

def warp_the_image(img_in, src, dst):
    
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    
    # input, Transformation-Matrix, image.shape, How to interpolate
    img_warped = cv2.warpPerspective(img_in, M, (img_in.shape[1],img_in.shape[0]), flags=cv2.INTER_LINEAR)
    return M, M_inv, img_warped

# window-search
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    dbg_lane_px_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(dbg_lane_px_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(dbg_lane_px_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Colors in the left and right lane regions
    dbg_lane_px_img[lefty, leftx] = [255, 0, 0]
    dbg_lane_px_img[righty, rightx] = [0, 0, 255]

    return leftx, lefty, rightx, righty, dbg_lane_px_img

def fit_poly_pipeline(binary_warped):
    # Find our lane pixels first
    left_x_pxs, left_y_pxs, right_x_pxs, right_y_pxs, dbg_lane_px_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_poly_pars = np.polyfit(left_y_pxs, left_x_pxs, 2)
    right_poly_pars = np.polyfit(right_y_pxs, right_x_pxs, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_poly_pars[0]*ploty**2 + left_poly_pars[1]*ploty + left_poly_pars[2]
        right_fitx = right_poly_pars[0]*ploty**2 + right_poly_pars[1]*ploty + right_poly_pars[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fitx, right_fitx, ploty, dbg_lane_px_img

def color_img(window_img,left_fitx, right_fitx, ploty):

    # tbd align with search area?
    margin = 30

    #Color the left-lane in red
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    # Color the right-lane in blue 
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))

    # Color the surface in green
    window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    road_surface = np.hstack((window1, window2))
    cv2.fillPoly(window_img, np.int_([road_surface]), (0, 255, 0))

    color_layer = window_img

    return color_layer

def measure_curvature_real(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**(3/2))/(abs(2*left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**(3/2))/(abs(2*right_fit_cr[0]))  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def put_text(img_in,left_curverad,center_diff,side_pos):

    img_with_text = np.copy(img_in)

    cv2.putText(img_with_text,'Radius of Curvature = '+str(round(left_curverad,1))+' (m)', 
    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(img_with_text, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',
    (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    return img_with_text

# The pipeline

def process_image(img_in):

    undist_img = calibrate_camera(img_in, objpoints, imgpoints)

    combined_binary = create_binary(undist_img, s_thresh=(170, 255), sx_thresh=(20, 100))*255

    M, M_inv, warped_binary = warp_the_image(combined_binary,src,dst)

    left_fitx, right_fitx, ploty, dbg_lane_px_img = fit_poly_pipeline(warped_binary)

    window_img = np.zeros_like(undist_img)

    color_layer = color_img(window_img,left_fitx, right_fitx, ploty)

    unwarped_color_layer = cv2.warpPerspective(color_layer, M_inv, (color_layer.shape[1],color_layer.shape[0]), flags=cv2.INTER_LINEAR)

    colored_undist = cv2.addWeighted(undist_img, 1, unwarped_color_layer, 0.3, 0)

    left_curverad, right_curverad = measure_curvature_real(left_fitx, right_fitx, ploty, ym_per_pix, xm_per_pix)

    # Measure center position
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - colored_undist.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    result = put_text(colored_undist, left_curverad,center_diff, side_pos)

    # tweak to produce the output for the writeup
    # result = combined_binary

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
    if(1):
        Output_video = 'output1_tracked.mp4'
        Input_video = 'project_video.mp4'
    if(0):
        Output_video = 'output_hard_tracked.mp4'
        Input_video = 'harder_challenge_video.mp4'
    
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip(Input_video)
    video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    video_clip.write_videofile(Output_video, audio=False)


