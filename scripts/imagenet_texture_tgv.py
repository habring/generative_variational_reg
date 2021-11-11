import sys, os
sys.path.append(os.path.abspath(sys.argv[0] + "/../..") + "/source")

import matpy as mp
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


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


    if hasattr(res,'K'):
        res_txt = open(folder+'psnr_results.txt','a') 
        res_txt.write('convex learning '+res.par.imname + ' PSNR = '+str(psnr_val)+'\n')
        res_txt.close()
    else:
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
    
    if hasattr(res,'K') and hasattr(res,'c'):
        #Save cartoon part
        im = mp.imnormalize(res.u - res.K.fwd(res.c))
        mp.imsave(outname+'_cart.png',im,rg=rg)
        #Save texture part
        im = mp.imnormalize(res.K.fwd(res.c))
        mp.imsave(outname+'_text.png',im,rg=rg)

    if hasattr(res,'patch'):
        #Save patches
        patches = mp.imshowstack(res.patch[...,:9])
        mp.imsave(outname+'_patches.png',patches,rg=rg)


################################
################################

niter = 5000

fixpars = {'niter':niter,'noise':0.0,'ld':1,'dtype':'inpaint'}


folder = 'experiments'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture/tgv'
if not os.path.isdir(folder):
    os.mkdir(folder)

folder = 'experiments/imagenet/texture/tgv/inpainting'
if not os.path.isdir(folder):
    os.mkdir(folder)


psnr_values = []
ssim_values = []

file1 = open(folder+'/results.txt',"w")
file1.close()

for i in range(1,27):
    original = mp.imread('imsource/imagenet/texture/img_'+str(i)+'.png')
    mask = mp.imread('imsource/imagenet/texture/mask.png')
    res = mp.tgv_recon(imname='imsource/imagenet/texture/img_'+str(i),u0=original,mask=mask,check=niter, **fixpars)

    psnr_val = np.round(mp.psnr(res.u,original,smax = np.abs(original.max()-original.min()),rescaled=True),decimals=2)
    save_output(res,psnr=True,folder=folder)
    ssim_val = ssim(res.u, res.orig, data_range=res.u.max() - res.u.min())

    psnr_values.append(psnr_val)
    ssim_values.append(ssim_val)

    name = 'img_'+str(i)

    file1 = open(folder+'/results.txt',"a")
    file1.write("Image:"+name+" \n")
    file1.write("PSNR: "+str(psnr_val)+ "\n")
    file1.write("SSIM: "+str(ssim_val)+ "\n"+"\n")
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

