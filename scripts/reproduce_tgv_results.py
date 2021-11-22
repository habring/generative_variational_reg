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
## reproduce_tgv_results.py:
## Reproduce the results with the TGV regularization shown in [1].
##
## -------------------------
## Andreas Habring (andreas.habring@uni-graz.at)
## Martin Holler (martin.holler@uni-graz.at)
## 
## 18.11.2021
## -------------------------
## If you consider this code to be useful, please cite:
## 
## [1] @misc{habring2021generative,
##          title={A Generative Variational Model for Inverse Problems in Imaging}, 
##          author={Andreas Habring and Martin Holler},
##          year={2021},
##          eprint={2104.12630},
##          archivePrefix={arXiv},
##          primaryClass={math.OC}
##          journal={SIAM Journal on Mathematics of Data Science}}
##


import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import matpy as mp
import numpy as np
import os



## Choose result type to compute
niter = 8000
cases = ['inpainting', 'denoising', 'deconvolution']
outfolder = 'experiments/tgv/'



folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv'
if not os.path.isdir(folder):
    os.mkdir(folder)


for case in cases:
  folder = 'experiments/tgv/'+case
  if not os.path.isdir(folder):
      os.mkdir(folder)


#Inpainting
if 'inpainting' in cases:


    folder = outfolder + 'inpainting/'
    fixpars = {'niter':niter,'noise':0.0,'ld':1,'dtype':'inpaint','check':500}

    # patchtest
    mask = np.load('imsource/corrupted/inpainting/patchtest_mask.npy')
    res = mp.tgv_recon(imname='imsource/patchtest.png',mask=mask,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)
    # Mix
    mask = np.load('imsource/corrupted/inpainting/cart_text_mix_mask.npy')
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',mask=mask,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)
    #Barbara
    mask = np.load('imsource/corrupted/inpainting/barbara_crop_mask.npy')
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',mask=mask,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)


    #fish
    mask = np.load('imsource/corrupted/inpainting/fish_mask.npy')
    res = mp.tgv_recon(imname='imsource/fish.png',mask=mask,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    #zebra
    mask = np.load('imsource/corrupted/inpainting/zebra_mask.npy')
    res = mp.tgv_recon(imname='imsource/zebra.png',mask=mask,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)


#Denoising
if 'denoising' in cases:

    folder = outfolder + 'denoising/'
    fixpars = {'niter':niter,'check':500}

    # Patches
    corrupted = np.load('imsource/corrupted/denoising/patchtest_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/patchtest.png', corrupted = corrupted, noise=0.1,ld=10.0,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)
    # Mix
    corrupted = np.load('imsource/corrupted/denoising/cart_text_mix_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png', corrupted = corrupted, noise=0.1,ld=12.5,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)
    #Barbara
    corrupted = np.load('imsource/corrupted/denoising/barbara_crop_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/barbara_crop.png', corrupted = corrupted, noise=0.1,ld=17.5,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)


    #fish
    corrupted = np.load('imsource/corrupted/denoising/fish_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/fish.png', corrupted = corrupted, noise=0.1,ld=15.0, **fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    #zebra
    corrupted = np.load('imsource/corrupted/denoising/zebra_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/zebra.png', corrupted = corrupted, noise=0.1,ld=20.0,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    
#Deblurring
if 'deconvolution' in cases:
    
    folder = outfolder + 'deconvolution/'
    F = mp.gconv([128,128],9,0.25)
    fixpars = {'niter':niter,'noise':0.025,'F':F,'check':500}


    #Mix
    corrupted = np.load('imsource/corrupted/deconvolution/cart_text_mix_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png', corrupted = corrupted, ld=750,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)
    #Barbara
    corrupted = np.load('imsource/corrupted/deconvolution/barbara_crop_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/barbara_crop.png', corrupted = corrupted, ld=500,**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    fixpars = {'niter':niter,'noise':0.025, 'check':niter}

    #fish
    corrupted = np.load('imsource/corrupted/deconvolution/fish_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/fish.png', corrupted = corrupted, ld=400.0,F = mp.gconv([240,320],13,0.25),**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    #zebra
    corrupted = np.load('imsource/corrupted/deconvolution/zebra_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/zebra.png', corrupted = corrupted, ld=600.0,F = mp.gconv([388,584],15,0.25),**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)

    #patchtest
    corrupted = np.load('imsource/corrupted/deconvolution/patchtest_corrupted.npy')
    res = mp.tgv_recon(imname='imsource/patchtest.png', corrupted = corrupted, ld=300.0, F = mp.gconv([120,120],9,0.25),**fixpars)
    mp.save_output(res,with_psnr=True,folder=folder)




