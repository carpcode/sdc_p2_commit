# This function shall plot the image with matplotlib 
# in order to identify the corners

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


img = mpimg.imread('test_images/straight_lines1.jpg')

#top-right
#bottom-right
#bottom left
#top left

src = np.float32(
    [[251,695],
      [1056,695],
      [774,513],
      [518,513]])


print(src[0])

plt.imshow(img)
plt.plot(251,695,'.',markersize=12)
plt.plot(1056,695,'.',markersize=12)
plt.plot(774,513,'.',markersize=12)
plt.plot(518,513,'.',markersize=12)
plt.plot(251,695,'.',markersize=12)
plt.plot(1056,695,'.',markersize=12)
plt.plot(251,400,'.',markersize=12)
plt.plot(1056,400,'.',markersize=12)
plt.show()

