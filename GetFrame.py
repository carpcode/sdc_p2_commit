import matplotlib.image as mpimg
import pickle
import glob
import numpy as np
from moviepy.editor import VideoFileClip


# this code shall grab single frames out of the project video for debugging purposes

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
    
    idx = 18
    ts = ['00:00:41.10',
    '00:00:41.20',
    '00:00:41.30',
    '00:00:41.40',
    '00:00:41.50',
    '00:00:42.00',
    '00:00:42.10',
    '00:00:42.20',
    '00:00:42.30',
    '00:00:42.40',
    '00:00:42.50',
    '00:00:42.59',
    '00:00:22.40']
    for id, t in enumerate(ts):
        clip1.save_frame('./output_images/test_'+str(idx)+'.jpg', t)
        idx = idx+1
    
    #video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    #video_clip.write_videofile(Output_video, audio=False)