import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import gen_reg as gr
import sys, os

import imageio
import matpy as mp
import numpy as np
import matplotlib.pyplot as plt
import time

num_iter = 5000
niter_steps = 500 #iterations for the smaller networks as initialization
show_every = 0
check = 0
check_opt = 1000
L_max_count = 0


# architecture:
L=3
ksz = [8 for i in range(L)] #Kernel sizes
nl = [8 for i in range(L)] #Kernel sizes
stride=[1,2,2]

nu = 0.9

niter = [niter_steps for i in range(L-1)]+[num_iter]

fixpars = {'application':'inpainting','L':L,'ksz':ksz,'nl':nl,
                                      'stride':stride, 'inpaint_perc':30, 'niter':niter,
                                      'check':check,'show_every':show_every,
                                      'check_opt':check_opt, 'nu':nu}


folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture/gen_reg'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture/gen_reg/inpainting'
if not os.path.isdir(folder):
    os.mkdir(folder)



psnr_values = []
ssim_values = []

file1 = open(folder+'/results.txt',"w")
file1.close()

for i in range(1,27):
  original = mp.imread('imsource/imagenet/texture/img_'+str(i)+'.png')
  mask = mp.imread('imsource/imagenet/texture/mask.png')
  corrupted = original*mask

  res = gr.gen_reg_successively(imname='imsource/imagenet/texture/img_'+str(i)+'.png',
                                data_is_corrupted = True, orig = original, u0 = corrupted, mask = mask, **fixpars)

  name = 'img_'+str(i)
  gr.save_results(res, name, foldername = folder)

  psnr_values.append(res.psnr)
  ssim_values.append(res.ssim)

  file1 = open(folder+'/results.txt',"a")
  file1.write("Image:"+name+" \n")
  file1.write("PSNR: "+str(res.psnr)+ "\n")
  file1.write("SSIM: "+str(res.ssim)+ "\n"+"\n")
  file1.close()


mean_psnr = np.mean(psnr_values)
std_psnr = np.std(psnr_values)

mean_ssim = np.mean(ssim_values)
std_ssim = np.std(ssim_values)


file1 = open(folder+'/results.txt',"a")
file1.write("MEAN of PSNR values: "+str(mean_psnr)+ "\n")
file1.write("STD of PSNR values: "+str(std_psnr)+ "\n")
file1.write("MEAN of SSIM values: "+str(mean_ssim)+ "\n")
file1.write("STD of SSIM values: "+str(std_ssim)+ "\n")
file1.close()


