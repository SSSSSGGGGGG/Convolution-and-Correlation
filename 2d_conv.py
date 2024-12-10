# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import os
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
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

im1=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_FH_0_S.png")
im2=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_V_S.png")
output = convolve2d(im1[:,:,0], im2[:,:,0], mode='full', boundary='fill', fillvalue=0)
plt.imshow(output,cmap="Reds")
plt.colorbar()