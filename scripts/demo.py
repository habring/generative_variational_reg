import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import gen_reg as gr
import time

import imageio
import matpy as mp
import numpy as np
import matplotlib.pyplot as plt

num_iter = 8000
niter_steps = 500 #iterations for the smaller networks as initialization
show_every = 0
check = 0
check_opt = 500
L_max_count = 0

cases = ['inpainting']

# architecture:
L=3
ksz = [8 for i in range(L)] #Kernel sizes
nl = [8 for i in range(L)] #Kernel sizes
stride=[1,2,2]

niter = [niter_steps for i in range(L-1)]+[num_iter]



folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/demo'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/demo/gen_reg'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/demo/tgv'
if not os.path.isdir(folder):
    os.mkdir(folder)


########### generative regularization

foldername = 'experiments/demo/gen_reg'


pars = {'application':'inpainting','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'inpaint_perc':30, 'nu':0.975, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt}

mask = np.load('imsource/corrupted/inpainting/barbara_crop_mask.npy')
original = mp.imread('imsource/barbara_crop.png')
corrupted = np.copy(original)
corrupted[mask==0] = 0.0
res = gr.gen_reg_successively(u0=corrupted, orig=original, mask = mask, data_is_corrupted=True, **pars)
gr.save_results(res, 'barbara_crop', foldername = foldername)



########### TGV regularization

folder = 'experiments/demo/tgv/'
fixpars = {'niter':num_iter,'noise':0.0,'ld':1,'dtype':'inpaint','check':500}

mask = np.load('imsource/corrupted/inpainting/barbara_crop_mask.npy')
res = mp.tgv_recon(imname='imsource/barbara_crop.png',mask=mask,**fixpars)
mp.save_output(res,with_psnr=True,folder=folder)





