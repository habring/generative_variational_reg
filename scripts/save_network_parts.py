import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import gen_reg as gr

import imageio
import matpy as mp
import numpy as np
import matplotlib.pyplot as plt



####### for this, you first need the results of reproduce_generative_regularization_results.py 

res = mp.pload('experiments/gen_reg/inpainting/barbara_crop')

folder = 'barbara_network'
if not os.path.isdir(folder):
    os.mkdir(folder)

mp.imsave('barbara_network/data.png',res.u0,rg=[])
mp.imsave('barbara_network/reconstructed.png',res.u,rg=[])

gen = mp.imnormalize(res.imsyn)
mp.imsave('barbara_network/generative.png',gen,rg=[0,1])


for i in range(res.par.n):
	for j in range(res.par.nf[i]):

		c = res.c[i][:,:,j]
		rg = []
		if i == 2 and j==5:
			rg = [c.min(), -c.min()]
		c = mp.imnormalize(c,rg=rg)

		mp.imsave('barbara_network/latent_'+str(i)+'_'+str(j)+'.png',c,rg=[])
		if i>0:
			D = mp.imnormalize(res.D[i][:,:,j,j])
			mp.imsave('barbara_network/kernel_'+str(i)+'_'+str(j)+'.png',D,rg=[])
		else:
			D = mp.imnormalize(res.D[i][:,:,j])
			mp.imsave('barbara_network/kernel_'+str(i)+'_'+str(j)+'.png',D,rg=[])






