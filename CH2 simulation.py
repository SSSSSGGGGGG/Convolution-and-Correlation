# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:52:34 2025

@author:Shang Gao 
"""

import numpy as np
from scipy.fftpack import ifft2, ifftshift, fft2,fftshift
import matplotlib.pyplot as plt

im1=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")
im2=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_H_0_s.png")
im3=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_V_s.png")
im4=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_H_180_s.png")

h,w=im1[:,:,0].shape
im1shift=fftshift(im1[:,:,0])
im2shift=fftshift(im2[:,:,0]) 
im3shift=fftshift(im3[:,:,0]) 
im4shift=fftshift(im4[:,:,0])

"""No noise"""
file1="No niose"

FT_forward=fftshift(fft2(im1shift))
FT_forward_L=fftshift(fft2(im2shift))
FT_forward_V=fftshift(fft2(im3shift))
FT_forward_L_In=fftshift(fft2(im4shift))

con1=FT_forward*FT_forward
cor1=FT_forward*np.conjugate(FT_forward)

In_FT_con1=ifftshift(ifft2(con1))
In_FT_cor1=ifftshift(ifft2(cor1))
I_con1=abs(In_FT_con1)**2
I_cor1=abs(In_FT_cor1)**2

con2=FT_forward*FT_forward_L
cor2=FT_forward_L*np.conjugate(FT_forward)

In_FT_con2=ifftshift(ifft2(con2))
In_FT_cor2=ifftshift(ifft2(cor2))
I_con2=abs(In_FT_con2)**2
I_cor2=abs(In_FT_cor2)**2

con3=FT_forward_V*FT_forward_L
cor3=FT_forward_V*FT_forward_L_In

In_FT_con3=ifftshift(ifft2(con3))
In_FT_cor3=ifftshift(ifft2(cor3))
I_con3=abs(In_FT_con3)**2
I_cor3=abs(In_FT_cor3)**2

max_=I_cor1.max()

def plotim(I_con,I_cor,t):
    # plt.figure()
    # plt.imshow(I_con,vmax=max_,cmap="hot")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Convolution {file1} {t}.png", dpi=300, bbox_inches="tight")
    # plt.close()
    
    x = np.arange(0, w)  
    y = np.arange(0, h) 
    X,Y=np.meshgrid(x,y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, I_con, cmap='jet')
    ax.set_zlim(0, max_) 
    ax.view_init(elev=35, azim=-45) 
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_facecolor('none')  # Makes the 3D axis background transparent
    fig.patch.set_alpha(0)  # Makes the figure background transparent
    # Save figure with transparent background
    plt.savefig(f"Convolution {file1} {t} 3d.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    
    # plt.figure()
    # plt.imshow(I_cor,vmax=max_,cmap="hot")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Correlation {file1} {t}.png", dpi=300, bbox_inches="tight")
    # plt.close()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, I_cor, cmap='jet')
    ax2.set_zlim(0, max_) 
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.view_init(elev=35, azim=-45) 
    ax2.set_facecolor('none')  # Makes the 3D axis background transparent
    fig2.patch.set_alpha(0) 
    plt.savefig(f"Correlation {file1} {t} 3d.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
plot1=plotim(I_con1, I_cor1, 1)
plot2=plotim(I_con2, I_cor2, 2)
plot1=plotim(I_con3, I_cor3, 3)

