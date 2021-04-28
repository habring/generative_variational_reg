from gen_reg_properly_discretized import *
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
show_network_mh(C,D,imsyn,cmap='gray',vrange=[],save=False, fname = 'network')
plt.savefig('random_network_'+str(n)+'_layers.png')


