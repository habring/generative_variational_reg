## A Generative Variational Model for Inverse Problems in Imaging
##
## Copyright (C) 2021 Andreas Habring, Martin Holler
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.
##
##
## radon_tgv_primal_dual_2d.py:
## Code for 2D variational reconstruction of tomographic data.
##
## -------------------------
## Richard Huber (richard.huber@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## Kristian Bredies (kristian.bredies@uni-graz.at)
## 
## 21.02.2019
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] R. Huber, G. Haberfehlner, M. Holler, G. Kothleitner,
##     K. Bredies. Total Generalized Variation regularization for
##     multi-modal electron tomography. *Nanoscale*, 2019. 
##     DOI: [10.1039/C8NR09058K](https://doi.org/10.1039/C8NR09058K).
##
## [2] M. Holler, R. Huber, F. Knoll. Coupled regularization with
##     multiple data discrepancies. Inverse Problems, Special
##     issue on joint reconstruction and multi-modality/multi-spectral
##     imaging, 34(8):084003, 2018.


import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

from gen_reg import *
import time

import imageio
import matpy as mp
import numpy as np
import matplotlib.pyplot as plt




N = 128
M = 128
n = 3
ksz = 8
nf = 1
stride = [1 for i in range(n)]


#initialize conv operator
K_list=[]
N_out = N
M_out = M
for i in range(n):
    Ki = cp_conv([N_out,M_out,ksz,nf],stride=stride[i])
    K_list.append(Ki)
    N_out,M_out,nfi = Ki.indim

conv = cp_conv_nlayers(K_list)


#initialize c, coefficients
c = np.random.rand(*conv.K[-1].indim)



#initialize atoms
D = []
D0 = np.random.rand(ksz,ksz,nf)
D0 -= D0.sum(axis=(0,1),keepdims=True)/(ksz*ksz)
D0 /= np.maximum(1.0, np.sqrt(np.square(D0).sum(axis=(0,1),keepdims=True)) )
D = [D0]
for i in range(1,n):
    Di = np.zeros([ksz,ksz,nf,nf])
    for j in range(nf):
        Di[:,:,j,j] = np.random.rand(ksz,ksz)
    Di /= np.maximum(1.0, np.sqrt(np.square(Di).sum(axis=(0,1),keepdims=True)) )
    D.append(Di)


#compute conv
C = [c]
c0 = c
for i in range(n-1):
    ci = conv.K[n-1-i].fwd(c0, D[n-1-i])
    C = [ci]+C
    c0 = ci

imsyn = conv.K[0].fwd(c0,D[0])


# vidsualize result
D[0] = D[0][...,np.newaxis]




fig = plt.figure(figsize=(15, 15))
    

#Concatenate D according to coefficients
Dflat = []
for i in range(len(D)):
    
    if i == 0:
        Dflat.append( np.zeros( (D[i].shape[0] + D[i].shape[0]*(D[i].shape[3]-1),D[i].shape[1],D[i].shape[2] ) ) )
    else:
        Dflat.append( np.zeros( (D[i].shape[0],D[i].shape[1],D[i].shape[2] ) ) )
    
    for j in range(D[i].shape[2]):
        if i==0:
            Dflat[i][...,j] = np.concatenate( [ D[i][...,j,l] for l in range(D[i].shape[3]) ], axis=0 )
        else:
            Dflat[i][...,j] = np.concatenate( [ D[i][...,j,l] for l in [j] ], axis=0 )


#Get subplot dimensions
nl = len(D)

#gridspec is used to set the ratio of subplots
rts = [1] + [1 if (np.remainder(i,2)==1) else 1./6. for i in range(2*nl)]
gs = gridspec.GridSpec(1,2*nl+1,width_ratios=rts)

#Show synthesized image
plt.subplot(gs[0,-1])
plt.imshow(imsyn,interpolation='none',cmap='gray')

plt.title('Network output')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
        
#Show layers
for l in range(nl):


    plt.subplot(gs[0, 2*nl-2*l-2])
    
    cc = imnormalize(C[l][...,0])
    plt.imshow(cc,interpolation='none',cmap='gray')

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.title('Latent variables '+str(l+1))
    
    if l<nl:
        
        plt.subplot(gs[0, 2*nl-2*l-1 ])
        
        DD = imnormalize(Dflat[l][...,0])
        plt.imshow(DD,interpolation='none',cmap='gray')
        
        plt.title('Kernels '+str(l+1))
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

plt.savefig('random_network_'+str(n)+'_layers.png')
