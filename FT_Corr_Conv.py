# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:52:17 2025

@author:Shang Gao 
"""

import numpy as np
from scipy.fftpack import ifft2, ifftshift, fft2,fftshift
import matplotlib.pyplot as plt

im1=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")
im2=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center_180.png")
im3=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")


im1shift=fftshift(im1[:,:,0])
im2shift=fftshift(im2[:,:,0])

h,w=im1shift.shape
"""No noise"""
file1="No niose"
FT_forward=fftshift(fft2(im1shift))
FT_forward2=fftshift(fft2(im2shift))

# plt.figure()
# plt.imshow(abs(FT_forward),cmap="hot")
# plt.colorbar()
# plt.title(f"G of {file1}")
# plt.axis("off")
# plt.show()

con1=FT_forward*FT_forward
cor1=FT_forward*np.conjugate(FT_forward)

In_FT_con1=ifftshift(ifft2(con1))
In_FT_cor1=ifftshift(ifft2(cor1))

# plt.figure()
# plt.imshow(abs(In_FT_con1)**2,cmap="hot")
# plt.title(f"Convolution {file1}")
# plt.colorbar()
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(abs(In_FT_cor1)**2,cmap="hot")
# plt.title(f"Correaltion {file1}")
# plt.colorbar()
# plt.axis("off")
# plt.show()

"""With noise"""
file2="with niose"
rand=np.random.uniform(0, 1, size=(h, w))*2*np.pi
rand2=np.random.uniform(0, 1, size=(h, w))*2*np.pi
im1shift_n=fftshift(im1[:,:,0])#*rand
im2shift_n=fftshift(im2[:,:,0])#*rand2

FT_n=fftshift(fft2(im1shift_n))
FT_n2=fftshift(fft2(im2shift_n))

# plt.figure()
# plt.imshow(abs(FT_n),cmap="hot")
# plt.colorbar()
# plt.title(f"G of {file2}")
# plt.axis("off")
# plt.show()
# plt.figure()
# plt.imshow(np.angle(FT_n),cmap="hot")
# plt.colorbar()
# plt.title(f"phase of {file2}")
# plt.axis("off")
# plt.show()

con2=FT_n*FT_n
cor2=FT_n*np.conjugate(FT_n)
In_FT_con2=ifftshift(ifft2(con2))
In_FT_cor2=ifftshift(ifft2(cor2))

# plt.figure()
# plt.imshow(abs(In_FT_con2)**2,cmap="hot")
# plt.colorbar()
# plt.title(f"Convolution {file2}")
# plt.axis("off")
# plt.show()

# plt.figure()
# plt.imshow(abs(In_FT_cor2)**2,cmap="hot")
# plt.colorbar()
# plt.title(f"Correaltion {file2}")
# plt.axis("off")
# plt.show()

"""With noise only phase"""
file3="with niose only phase"

con3=np.exp(1j*np.angle(FT_n))*np.exp(1j*np.angle(FT_n2))
cor3=np.exp(1j*np.angle(FT_n))*np.exp(-1j*np.angle(FT_n2))
In_FT_con3=ifftshift(ifft2(con3))
In_FT_cor3=ifftshift(ifft2(cor3))

con4=np.exp(1j*np.angle(FT_n))*np.exp(1j*np.angle(FT_n))
cor4=np.exp(1j*np.angle(FT_n))*np.exp(-1j*np.angle(FT_n))
In_FT_con4=ifftshift(ifft2(con4))
In_FT_cor4=ifftshift(ifft2(cor4))

plt.figure()
plt.imshow(abs(In_FT_con3)**2,cmap="hot")
plt.colorbar()
plt.title(f"Convolution 1 and 2 {file3}")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(abs(In_FT_cor3)**2,cmap="hot")
plt.colorbar()
plt.title(f"Correaltion 1 and 2 {file3}")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(abs(In_FT_con4)**2,cmap="hot")
plt.colorbar()
plt.title(f"Convolution 1 and 1 {file3}")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(abs(In_FT_cor4)**2,cmap="hot")
plt.colorbar()
plt.title(f"Correaltion 1 and 1 {file3}")
plt.axis("off")
plt.show()