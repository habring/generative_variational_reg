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
## reproduce_generative_regularization_results.py:
## Reproduce the results with the proposed method shown in [1].
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

cases = ['inpainting', 'denoising', 'deconvolution', 'supres', 'jpeg']

# architecture:
L=3
ksz = [8 for i in range(L)] #Kernel sizes
nl = [8 for i in range(L)] #Kernel sizes
stride=[1,2,2]

niter = [niter_steps for i in range(L-1)]+[num_iter]



folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/gen_reg'
if not os.path.isdir(folder):
    os.mkdir(folder)

for case in cases:
  folder = 'experiments/gen_reg/'+case
  if not os.path.isdir(folder):
      os.mkdir(folder)


foldername = 'experiments/gen_reg/'


if 'inpainting' in cases:

  fixpars = {'application':'inpainting','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'inpaint_perc':30, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt}


  images = ['barbara_crop', 'cart_text_mix', 'patchtest', 'zebra', 'fish']

  nu_dict = {
    'barbara_crop': 0.975,
    'cart_text_mix': 0.975,
    'patchtest': 0.975,
    'zebra': 0.975,
    'fish': 0.925}

  for name in images:
    nu = nu_dict[name]
    mask = np.load('imsource/corrupted/inpainting/'+name+'_mask.npy')
    original = mp.imread('imsource/'+name+'.png')
    corrupted = np.copy(original)
    corrupted[mask==0] = 0.0
    res = gr.gen_reg_successively(u0=corrupted, orig=original, mask = mask, data_is_corrupted=True, nu=nu, **fixpars)
    gr.save_results(res, name, foldername = foldername+'inpainting')


if 'denoising' in cases:

  fixpars = {'application':'denoising','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'noise':0.1, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt, 'nu':0.925}


  images = ['barbara_crop', 'cart_text_mix', 'patchtest', 'zebra', 'fish']


  ld_dict = {
    'barbara_crop': 22.5,
    'cart_text_mix': 20.0,
    'patchtest': 20.0,
    'zebra': 30.0,
    'fish': 30.0}

  for name in images:
    ld = ld_dict[name]
    original = mp.imread('imsource/'+name+'.png')
    corrupted = np.load('imsource/corrupted/denoising/'+name+'_corrupted.npy')
    res = gr.gen_reg_successively(u0=corrupted, orig=original, data_is_corrupted=True, ld = ld, **fixpars)
    gr.save_results(res, name, foldername = foldername+'denoising')


if 'deconvolution' in cases:

  fixpars = {'application':'deconvolution','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'noise':0.025, 'blur_sig':0.25, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt, 'nu':0.925}


  ld_dict = {
    'barbara_crop': 600.0,
    'cart_text_mix': 700.0,
    'zebra': 1000.0,
    'fish': 500.0,
    'patchtest': 800.0}


  images = ['barbara_crop', 'cart_text_mix', 'zebra', 'fish', 'patchtest']

  # blur_size needs to be odd
  for name in images:
    if name == 'barbara_crop' or name == 'cart_text_mix' or name == 'patchtest':
      blur_size = 9
    elif name == 'fish':
      blur_size = 13
    elif name == 'zebra':
      blur_size = 15
    else:
      raise Exception('invalid name')

    ld = ld_dict[name]
    original = mp.imread('imsource/'+name+'.png')
    corrupted = np.load('imsource/corrupted/deconvolution/'+name+'_corrupted.npy')
    res = gr.gen_reg_successively(u0=corrupted, orig=original, data_is_corrupted=True, blur_size=blur_size, ld = ld, **fixpars)
    gr.save_results(res, name, foldername = foldername+'deconvolution')


if 'supres' in cases:

  fixpars = {'application':'supres','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'sr_fac':4, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt}


  images = ['zebra', 'fish_large', 'barbara_crop']

  nu_dict = {
    'fish_large': 0.95,
    'zebra': 0.9,
    'barbara_crop': 0.975}

  for name in images:
    nu = nu_dict[name]
    if name == 'barbara_crop':
      data_is_corrupted = True
    else:
      data_is_corrupted = False

    res = gr.gen_reg_successively(imname='imsource/'+name+'.png',nu=nu, data_is_corrupted = data_is_corrupted, **fixpars)
    gr.save_results(res, name, foldername = foldername+'supres')


if 'jpeg' in cases:

  fixpars = {'application':'jpeg','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt}


  images = ['barbara_crop', 'cart_text_mix', 'patchtest', 'fish_cropped', 'zebra_cropped']
  
  nu_dict = {
    'barbara_crop': 0.875,
    'cart_text_mix': 0.875,
    'patchtest': 0.95,
    'fish_cropped': 0.85,
    'zebra_cropped': 0.775}

  for name in images:
    nu = nu_dict[name]
    res = gr.gen_reg_successively(imnamejpeg='imsource/'+name+'.jpg', imname='imsource/'+name+'.png',nu=nu, **fixpars)
    gr.save_results(res, name, foldername = foldername+'jpeg')


  