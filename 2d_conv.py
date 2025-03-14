# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
"""
'full':
The output contains all possible overlaps of the input and the kernel.
The output size is larger than the input, as it includes regions where the kernel extends beyond the edges of the input.

'same':
The output is the same size as the input.
This is achieved by centering the kernel over the input and trimming edges as needed.
Often used in image processing to maintain the original size.

'valid':
The output contains only regions where the kernel fully overlaps with the input (no padding or boundary effects).
This results in a smaller output array.

'fill':

Pads the input with a constant value (specified by fillvalue).
Example: If fillvalue=0, the edges are padded with zeros (zero-padding).
'wrap':

Wraps the input around itself. This treats the input as if it's periodic.
Useful in certain signal processing tasks.
'symm':

Pads the input with its mirror image at the boundaries.
Helps preserve continuity near the edges.

The fillvalue parameter specifies the constant value used when boundary='fill'.

Example: If fillvalue=0, the padded area is filled with zeros.
If fillvalue=255, the padding will be white if treating the array as a grayscale image.
"""

im1=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")
im2=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")
im3=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center_180.png")
# im4=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_H180_128_128_S.png")
def conv(ima,imb):
    output = convolve2d(ima[:,:,0], imb[:,:,0], mode='same', boundary='fill', fillvalue=0)
    return output
# def conv_full(ima,imb):
#     output = convolve2d(ima[:,:,0], imb[:,:,0], mode='full', boundary='fill', fillvalue=0)
#     return output


# Correlation:invert image manually.
corr_result=conv(im1,im3)
h,w=corr_result.shape
x = np.arange(w//2-50, w//2+50)  
y = np.arange(h//2-50, h//2+50) 
X,Y=np.meshgrid(x,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, corr_result[h//2-50: h//2+50,w//2-50: w//2+50], cmap='jet')
ax.set_zlim(0, corr_result.max()) 
ax.view_init(elev=30, azim=135) 
plt.show()

plt.figure()
plt.imshow(corr_result,vmax=np.max(corr_result),cmap="hot")
# Get the current axis
ax = plt.gca()
# Add a visible border around the image
border = Rectangle((0, 0), corr_result.shape[1]-1, corr_result.shape[0]-1,linewidth=0.5, edgecolor='black', facecolor='none')
ax.add_patch(border)
# Hide axis (ticks and labels)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)  # Removes axis frame
plt.colorbar()
plt.title("Correlate")
plt.show()

"""# convolution"""
conv_result=conv(im1,im2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, conv_result[h//2-50: h//2+50,w//2-50: w//2+50], cmap='jet')
ax.set_zlim(0, corr_result.max()) 
ax.view_init(elev=30, azim=135) 
plt.show()

plt.figure()
plt.imshow(conv_result,vmax=np.max(corr_result),cmap="hot")
# Get the current axis
ax = plt.gca()
# Add a visible border around the image
border = Rectangle((0, 0), corr_result.shape[1]-1, corr_result.shape[0]-1,linewidth=0.5, edgecolor='black', facecolor='none')
ax.add_patch(border)
# Hide axis (ticks and labels)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)  # Removes axis frame
plt.colorbar()
plt.title("Convolve")
plt.show()

# plt.figure()
# plt.imshow(im1,cmap="gray")
# plt.axis("off")
# plt.colorbar()


# plt.figure()
# plt.imshow(im2,cmap="gray")
# plt.axis("off")
# plt.colorbar()


# plt.figure()
# plt.imshow(im3,cmap="gray")
# plt.colorbar()
# plt.axis("off")
