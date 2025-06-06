# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:52:34 2025

@author:Shang Gao 
"""

import numpy as np
from scipy.fftpack import ifft2, ifftshift, fft2,fftshift
import matplotlib.pyplot as plt

im1=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_center.png")
# im2=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_FH_0_S.png")
# im3=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_V_S.png")
# im4=plt.imread("C:/Users/Laboratorio/Conv and Corr/lotus_FH_180_S.png")

im1shift=fftshift(im1[:,:,0])

h,w=im1shift.shape
c=h//2
rand1=np.random.uniform(0,1,(h,w))*2*np.pi
rand2=np.random.uniform(0,1,(h,w))*2*np.pi

im1shift_N=fftshift(im1[:,:,0])*rand1
im1shift_N2=fftshift(im1[:,:,0])*rand2
# im2shift=fftshift(im2[:,:,0]) 
# im3shift=fftshift(im3[:,:,0]) 
# im4shift=fftshift(im4[:,:,0])


"""No noise_magnitude+phase"""
file1="No niose"

FT_forward=fftshift(fft2(im1shift))
FT_forward_N=fftshift(fft2(im1shift_N))
FT_forward_N2=fftshift(fft2(im1shift_N2))
# FT_forward_V=fftshift(fft2(im3shift))
# FT_forward_L_In=fftshift(fft2(im4shift))

con1=FT_forward*FT_forward
cor1=FT_forward*abs(FT_forward)*np.exp(-1j*np.angle(FT_forward))

In_FT_con1=ifftshift(ifft2(con1))
In_FT_cor1=ifftshift(ifft2(cor1))
I_con1=abs(In_FT_con1)**2
I_cor1=abs(In_FT_cor1)**2
"""No noise_phase"""
con2=np.exp(1j*np.angle(FT_forward))*np.exp(1j*np.angle(FT_forward))
cor2=np.exp(1j*np.angle(FT_forward))*np.exp(-1j*np.angle(FT_forward))

In_FT_con2=ifftshift(ifft2(con2))
In_FT_cor2=ifftshift(ifft2(cor2))
I_con2=abs(In_FT_con2)**2
I_cor2=abs(In_FT_cor2)**2

"""With noise_magnitude+phase"""
con3=FT_forward_N*FT_forward_N
cor3=FT_forward_N*abs(FT_forward_N)*np.exp(-1j*np.angle(FT_forward_N))

In_FT_con3=ifftshift(ifft2(con3))
In_FT_cor3=ifftshift(ifft2(cor3))
I_con3=abs(In_FT_con3)**2
I_cor3=abs(In_FT_cor3)**2

"""With Diff noise_magnitude+phase"""
con5=FT_forward_N*FT_forward_N2
cor5=FT_forward_N*abs(FT_forward_N2)*np.exp(-1j*np.angle(FT_forward_N2))

In_FT_con5=ifftshift(ifft2(con5))
In_FT_cor5=ifftshift(ifft2(cor5))
I_con5=abs(In_FT_con5)**2
I_cor5=abs(In_FT_cor5)**2

"""With Same noise_phase"""
con4=np.exp(1j*np.angle(FT_forward_N))*np.exp(1j*np.angle(FT_forward_N))
cor4=np.exp(1j*np.angle(FT_forward_N))*np.exp(-1j*np.angle(FT_forward_N))

In_FT_con4=ifftshift(ifft2(con4))
In_FT_cor4=ifftshift(ifft2(cor4))
I_con4=abs(In_FT_con4)**2
I_cor4=abs(In_FT_cor4)**2

"""With Diff noise_phase"""
con6=np.exp(1j*np.angle(FT_forward_N))*np.exp(1j*np.angle(FT_forward_N2))
cor6=np.exp(1j*np.angle(FT_forward_N))*np.exp(-1j*np.angle(FT_forward_N2))

In_FT_con6=ifftshift(ifft2(con6))
In_FT_cor6=ifftshift(ifft2(cor6))
I_con6=abs(In_FT_con6)**2
I_cor6=abs(In_FT_cor6)**2

def plotim(I_con,I_cor,t,max_):   
    
    # plt.figure()
    # plt.imshow(I_con,cmap="hot")
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f"Convolution {t}.png", dpi=300, bbox_inches="tight")
    # plt.close()
    
    x = np.arange(c-20, c+20)  
    y = np.arange(c-20, c+20) 
    X,Y=np.meshgrid(x,y)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, I_con, cmap='jet')
    # ax.set_zlim(0, max_) 
    # ax.view_init(elev=35, azim=-45) 
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # plt.savefig(f"Convolution {t} 3d.png", dpi=300, bbox_inches="tight",pad_inches=0.4)
    # plt.close()
    
    plt.figure()
    plt.imshow(I_cor,cmap="hot")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"Correlation {t}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(X, Y, I_cor[c-20: c+20,c-20: c+20], cmap='jet')
    ax2.set_zlim(0, max_)
    # ax2.set_xlim(c-20, c+20)
    # ax2.set_ylim(c-20, c+20)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.view_init(elev=35, azim=-45) 
    plt.savefig(f"Correlation {t} 3d.png", dpi=300, bbox_inches="tight",pad_inches=0.4)
    plt.close()
# plot1=plotim(I_con1, I_cor1, "No noise_M+P",I_cor1.max())
# plot2=plotim(I_con2, I_cor2, "No noise_P",I_cor2.max())
# plot3=plotim(I_con3, I_cor3, "With noise_M+P",I_cor3.max())
plot4=plotim(I_con4, I_cor4, "With noise_P",I_cor4.max())
# plot5=plotim(I_con5, I_cor5, "With Diff noise_M+P",I_cor5.max())
plot6=plotim(I_con6, I_cor6, "With Diff noise_P",I_cor6.max())
