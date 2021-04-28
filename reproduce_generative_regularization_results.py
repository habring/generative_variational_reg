import gen_reg as gr
import time

import imageio
import matpy as mp
import numpy
import matplotlib.pyplot as plt

num_iter = 8000
niter_steps = 500 #iterations for the smaller networks as initialization
show_every = 0
check = 0
check_opt = 500
L_max_count = 0

cases = ['inpainting', 'denoising', 'deconvolution', 'supres', 'jpeg']

# architecture:
n=3
ksz = [8 for i in range(n)] #Kernel sizes
nf = [8 for i in range(n)] #Kernel sizes
stride=[1,2,2]

niter = [niter_steps for i in range(n-1)]+[num_iter]


foldername = 'experiments/gen_reg/'

if 'inpainting' in cases:

  fixpars = {'application':'inpainting','n':n,'ksz':ksz,'nf':nf,
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
    res = gr.gen_reg_successively(imname='imsource/'+name+'.png',nu=nu, **fixpars)
    gr.save_results(res, name, foldername = foldername+'inpainting')





if 'denoising' in cases:

  fixpars = {'application':'denoising','n':n,'ksz':ksz,'nf':nf,
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
    res = gr.gen_reg_successively(imname='imsource/'+name+'.png',ld = ld, **fixpars)
    gr.save_results(res, name, foldername = foldername+'denoising')



if 'deconvolution' in cases:

  fixpars = {'application':'deconvolution','n':n,'ksz':ksz,'nf':nf,
                                      'stride':stride, 'noise':0.025, 'blur_sig':0.25, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt, 'nu':0.925}


  ld_dict = {
    'barbara_crop': 600.0,
    'cart_text_mix': 700.0,
    'zebra': 1000.0,
    'fish': 500.0}


  images = ['barbara_crop', 'cart_text_mix', 'zebra', 'fish']

  # blur_size needs to be odd
  for name in images:
    if name == 'barbara_crop' or name == 'cart_text_mix':
      blur_size = 9
    elif name == 'fish':
      blur_size = 13
    elif name == 'zebra':
      blur_size = 15
    else:
      raise Exception('invalid name')

    ld = ld_dict[name]
    res = gr.gen_reg_successively(imname='imsource/'+name+'.png', blur_size=blur_size, ld = ld, **fixpars)
    gr.save_results(res, name, foldername = foldername+'deconvolution')


if 'supres' in cases:

  fixpars = {'application':'supres','n':n,'ksz':ksz,'nf':nf,
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

  fixpars = {'application':'jpeg','n':n,'ksz':ksz,'nf':nf,
                                      'stride':stride, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt}


  images = ['barbara_crop', 'cart_text_mix', 'patchtest']
  
  nu_dict = {
    'barbara_crop': 0.875,
    'cart_text_mix': 0.875,
    'patchtest': 0.95}

  for name in images:
    nu = nu_dict[name]
    res = gr.gen_reg_successively(imnamejpeg='imsource/'+name+'.jpg', imname='imsource/'+name+'.png',nu=nu, **fixpars)
    gr.save_results(res, name, foldername = foldername+'jpeg')


  