import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import matpy as mp
import numpy as np
import os


#Function to save the output and compute PSNR values
def save_output(res,psnr=True,folder=''):

    #Create folder if necessary
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    #Generate output name
    outname = res.output_name(folder=folder)

    #Save result file
    res.save(folder=folder)

    if psnr:
        psnr_val = np.round(mp.psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=True),decimals=2)
        print( res.output_name() + '\nPSNR: ' + str(psnr_val) )
    else:
        psnr_val = np.nan

    res_txt = open(folder+'psnr_results.txt','a') 
    res_txt.write('TGV '+res.par.imname + ' PSNR = '+str(psnr_val)+'\n')
    res_txt.close()

    #Save original image
    #rg = [res.orig.min(),res.orig.max()]
    rg = []
    mp.imsave(outname+'_orig.png',res.orig,rg=rg)
    #Save data images
    mp.imsave(outname+'_data.png',res.u0,rg=rg)
    #Save reconstructed image
    mp.imsave(outname+'_recon.png',res.u,rg=rg)


################################
################################



folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv/inpainting'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv/denoising'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv/deconvolution'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv/supres'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/tgv/jpeg'
if not os.path.isdir(folder):
    os.mkdir(folder)





## Choose result type to compute
niter = 8000
types = ['inpaint','denoise','deblurring']
outfolder = 'experiments/tgv/'



#Inpainting
if 'inpaint' in types:


    folder = outfolder + 'inpainting/'
    fixpars = {'niter':niter,'noise':0.0,'ld':1,'dtype':'inpaint','check':500}

    # patchtest
    data = mp.pload('imsource/tgv_data/inpainting/patchtest')
    mask = data.mask
    mask = np.abs(mask - 1.0)<1e-2
    u0 = data.orig
    res = mp.tgv_recon(imname='patchtest.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)
    # Mix
    data = mp.pload('imsource/tgv_data/inpainting/cart_text_mix')
    mask = data.mask
    mask = np.abs(mask - 1.0)<1e-2
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    data = mp.pload('imsource/tgv_data/inpainting/barbara_crop')
    mask = data.mask
    mask = np.abs(mask - 1.0)<1e-2
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)


    #fish
    data = mp.pload('imsource/tgv_data/inpainting/fish')
    mask = data.mask
    mask = np.abs(mask - 1.0)<1e-2
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/fish.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)

    #zebra
    data = mp.pload('imsource/tgv_data/inpainting/zebra')
    mask = data.mask
    mask = np.abs(mask - 1.0)<1e-2
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/zebra.png',u0=u0,mask=mask,**fixpars)
    save_output(res,psnr=True,folder=folder)


#Denoising
if 'denoise' in types:

    folder = outfolder + 'denoising/'
    fixpars = {'niter':niter,'check':500}

    # Patches
    data = mp.pload('imsource/tgv_data/denoising/patchtest')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='patchtest.png',u0=u0, corrupted = corrupted, noise=0.1,ld=10.0,**fixpars)
    save_output(res,psnr=True,folder=folder)
    # Mix
    data = mp.pload('imsource/tgv_data/denoising/cart_text_mix')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',u0=u0, corrupted = corrupted, noise=0.1,ld=12.5,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    data = mp.pload('imsource/tgv_data/denoising/barbara_crop')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',u0=u0, corrupted = corrupted, noise=0.1,ld=17.5,**fixpars)
    save_output(res,psnr=True,folder=folder)


    #fish
    data = mp.pload('imsource/tgv_data/denoising/fish')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/fish.png',u0=u0, corrupted = corrupted, noise=0.1,ld=15.0, **fixpars)
    save_output(res,psnr=True,folder=folder)

    #zebra
    data = mp.pload('imsource/tgv_data/denoising/zebra')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/zebra.png',u0=u0, corrupted = corrupted, noise=0.1,ld=20.0,**fixpars)
    save_output(res,psnr=True,folder=folder)

    
#Deblurring
if 'deblurring' in types:
    
    folder = outfolder + 'deconvolution/'
    F = mp.gconv([128,128],9,0.25)
    fixpars = {'niter':niter,'noise':0.025,'F':F,'check':500}


    #Mix
    data = mp.pload('imsource/tgv_data/deconv/cart_text_mix')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/cart_text_mix.png',u0=u0, corrupted = corrupted, ld=750,**fixpars)
    save_output(res,psnr=True,folder=folder)
    #Barbara
    data = mp.pload('imsource/tgv_data/deconv/barbara_crop')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/barbara_crop.png',u0=u0, corrupted = corrupted, ld=500,**fixpars)
    save_output(res,psnr=True,folder=folder)

    fixpars = {'niter':niter,'noise':0.025, 'check':niter}

    #fish
    data = mp.pload('imsource/tgv_data/deconv/fish')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/fish.png',u0=u0, corrupted = corrupted, ld=400.0,F = mp.gconv([240,320],13,0.25),**fixpars)
    save_output(res,psnr=True,folder=folder)

    #zebra
    data = mp.pload('imsource/tgv_data/deconv/zebra')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/zebra.png',u0=u0, corrupted = corrupted, ld=600.0,F = mp.gconv([388,584],15,0.25),**fixpars)
    save_output(res,psnr=True,folder=folder)

    #patchtest
    data = mp.pload('imsource/tgv_data/deconv/patchtest')
    corrupted = data.u0
    u0 = data.orig
    res = mp.tgv_recon(imname='imsource/patchtest.png',u0=u0, corrupted = corrupted, ld=300.0, F = mp.gconv([120,120],9,0.25),**fixpars)
    save_output(res,psnr=True,folder=folder)




