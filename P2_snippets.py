# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 08:52:45 2021

@author: PCA4FE
"""

# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pdb

# This shall serve to understand the cv2.line pt parameter

# Input help forum
src = np.float32(
    [[568,468],
     [715,468],
     [1040,680],
     [270,680]])

src1 = [[568,468],
     [715,468],
     [1040,680],
     [270,680]]

src2 = np.float32(src1)

dst = np.float32(
     [[200,0],
      [1000,0],
      [1000,680],
      [200,680]])


img = mpimg.imread('test_images/straight_lines2.jpg')

print(tuple(int(src[0])))

plt.imshow(img)
plt.imshow(cv2.line(img,tuple(src1[0]),tuple(src1[1]),[255,0,0],12))
plt.show()