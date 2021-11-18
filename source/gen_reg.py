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
## Source code for the generative variational regularization method proposed in [1].
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



#!/usr/bin/env python
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.axes import *
import numpy as np
import pyopencl as cl
import pyopencl.array as array
import time
from datetime import datetime
import scipy.signal
import libjpeg as jlib
from skimage.metrics import structural_similarity as ssim

import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')

from matpy import *
#Doc
#https://documen.tician.de/pyopencl/
#https://documen.tician.de/pyopencl/array.html

#Get list of available opencl devices
def get_device_list():
    platforms = cl.get_platforms()
    devices = []
    for pf in platforms:
        devices.extend(pf.get_devices())

    #return devices
    devices_gpu = [dev for dev in devices if dev.type & cl.device_type.GPU != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.GPU == 0]
    devices_acc = [dev for dev in devices if dev.type & cl.device_type.ACCELERATOR != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.ACCELERATOR == 0]
    devices_cpu = [dev for dev in devices if dev.type & cl.device_type.CPU != 0]
    devices = [dev for dev in devices if dev.type & cl.device_type.CPU == 0]
    return devices_gpu+devices_acc+devices_cpu+devices
    
#Class to store OpenCL programs    
class Program(object):
    def __init__(self, code):
    
        self.code = code

    def build(self,ctx):
        self._cl_prg = cl.Program(ctx, self.code)
        self._cl_prg.build()
        self._cl_kernels = self._cl_prg.all_kernels()
        for kernel in self._cl_kernels:
            self.__dict__[kernel.function_name] = kernel
        
#Class to store the OpenCL context
class ContextStore(object):
    def __init__(self):
        self.contexts = {}
        return

    def __getitem__(self, dev):
        if not dev in self.contexts:
            self.contexts[dev] = cl.Context(devices=[dev])
        return self.contexts[dev]    


###########################################################
# Helper functions ########################################

def load_jpeg_data(filename):
    file = open(filename, 'rb')
    #file = urllib.urlopen(filename)
    (image_info, comp_info, coeff) = jlib.jpeg_read_file(file)
    file.close()

    Jmin = []
    Jmax = []
    subsampling = []
    for (comp, data) in zip(comp_info, coeff):
        Jmin.append(np.ascontiguousarray(((data.astype('float32')-0.5)
                     *comp['quant_tbl']).transpose(1,3,0,2).reshape(
                         8*data.shape[1],8*data.shape[0]).T))
        Jmax.append(np.ascontiguousarray(((data.astype('float32')+0.5)
                     *comp['quant_tbl']).transpose(1,3,0,2).reshape(
                         8*data.shape[1],8*data.shape[0]).T))
        subsampling.append((comp['v_subsampling'], comp['h_subsampling']))

    real_size = (np.array(subsampling)*np.array([J.shape for J in Jmin])).max(axis=0)
    for i in range(len(subsampling)):
        Jmin[i] = np.pad(Jmin[i], ((0, (int)(real_size[0]/subsampling[i][0] - Jmin[i].shape[0])),
                                   (0, (int)(real_size[1]/subsampling[i][1] - Jmin[i].shape[1]))),
                                'constant')
        Jmax[i] = np.pad(Jmax[i], ((0, (int)(real_size[0]/subsampling[i][0] - Jmax[i].shape[0])),
                                   (0, (int)(real_size[1]/subsampling[i][1] - Jmax[i].shape[1]))),
                                'constant')

    return (image_info, subsampling, Jmin, Jmax)

#Get mask for inpainting
def get_mask(shape,mtype='rand',perc=10):

    

    if mtype=='rand':

        mask = np.zeros(shape,dtype='int32')
        
        np.random.seed(998)
        
        rmask = np.random.rand(*shape)
        idx = rmask < perc/100.0
        
        mask[idx] = 1
    else:
        mask = np.ones(shape,dtype='int32')
        n,m = shape
        mask[50:80,50:80]=0


    return mask
      


def show_network(C,D,imsyn, imsyn_list,cmap='gray',vrange=[],save=False, fname = 'network'):

    #Make figure
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
    ncmax = max([C[i].shape[2] for i in range(len(C))])
    
    #gridspec is used to set the ratio of subplots
    rts = [1./2.] + [1 if (np.remainder(i,2)==1) else 1./6. for i in range(2*nl)]
    gs = gridspec.GridSpec(ncmax,2*nl+1,width_ratios=rts)
    
    
    #Show synthesized image
    plt.subplot(gs[0,-1])
    if not vrange:
        plt.imshow(imsyn,interpolation='none',cmap=cmap)
    else:
        plt.imshow(imsyn,interpolation='none',cmap=cmap,vmin=vrange[0],vmax=vrange[1])
    plt.title('Network output')
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
            
    #Show layers
    for l in range(nl):
        

        for c in range(C[l].shape[2]):

            plt.subplot(gs[c, 2*nl-2*l-2])
            
            if not vrange:
                if l < nl-1:
                    cc1 = imnormalize(C[l][...,c])
                    cc2 = imnormalize(imsyn_list[l][...,c])
                    cc1 = np.stack((cc1,)*3, axis=-1)
                    cc2 = np.stack((cc2,)*3, axis=-1)
                    separating_line = np.zeros([cc1.shape[0],1,3])
                    separating_line[:,:,1] = 1
                    cc = np.concatenate([cc2, separating_line, cc1], axis=1 )
                else:
                    cc = imnormalize(C[l][...,c])
                plt.imshow(cc,interpolation='none',cmap=cmap)
            else:
                if l < nl-1:
                    cc1 = C[l][...,c]
                    cc2 = imsyn_list[l][...,c]
                    cc1 = np.stack((cc1,)*3, axis=-1)
                    cc2 = np.stack((cc2,)*3, axis=-1)
                    separating_line = np.zeros([cc1.shape[0],1,3])
                    separating_line[:,:,1] = 1
                    cc = np.concatenate([cc2, separating_line, cc1], axis=1 )
                else:
                    cc = C[l][...,c]
                plt.imshow(C[l][...,c],interpolation='none',cmap=cmap,vmin=vrange[0],vmax=vrange[1])
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if(c==0):
                plt.title('Latent variables '+str(l+1))
            
            if l<nl:
                
                plt.subplot(gs[c, 2*nl-2*l-1 ])
                
                if not vrange:
                    DD = imnormalize(Dflat[l][...,c])
                    plt.imshow(DD,interpolation='none',cmap=cmap)
                else:
                    plt.imshow(Dflat[l][...,c],interpolation='none',cmap=cmap,vmin=vrange[0],vmax=vrange[1])
                if(c==0):
                    plt.title('Kernels '+str(l+1))
            ax = plt.gca()
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        
    
    if save:
        plt.savefig(fname)
    
    return fig



def save_results(res, name, foldername = ''):
    
    filename = name

    # remove '/' at end or beginning
    if(filename[0]=='/'):
        filename = filename[1:]

    if(filename[-1]=='/'):
        filename = filename[:-1]

    res.save(folder=foldername, fname=filename)

    # save comparison and result
    im = imnormalize(res.u - res.imsyn)
    imsave(foldername+'/'+filename+'_cart.png',im,rg=[])
    im = imnormalize(res.imsyn)
    imsave(foldername+'/'+filename+'_text.png',im,rg=[])
    imsave(foldername+'/'+filename+'_recon.png',res.u,rg=[])
    imsave(foldername+'/'+filename+'_corrupted.png',res.u0,rg=[])
    if np.any(res.orig):
        imsave(foldername+'/'+name+'_original.png',res.orig,rg=[])

    #Insert degenerate axis in D for consistency
    if len(res.D[0].shape)<4:
        res.D[0] = res.D[0][...,np.newaxis]

    # save network decomposition
    show_network(res.c,res.D,res.imsyn, res.imsyn_list, save=True, fname = foldername+'/'+filename+'_network.png')

    #plot(np.log(res.ob_val),title='log of objective functional. final psnr: '+str(res.psnr))

    fig = plt.figure()
    ax = plt.gca() 
    plt.figure(fig.number)
        
    ax.plot(np.log(res.ob_val))
    plt.title('objective functional. final psnr: '+str(res.psnr))    
    ax.set_xlabel('number of iterations')
    ax.set_ylabel(r'ln$(\mathcal{E}^{\;D}_y)$',rotation=0,labelpad=20)
    plt.tight_layout()
    plt.savefig(foldername+'/'+filename+'_obval_short.png')

    closefig()

    return

def optimality_conditions(res, fwd_operator = lambda u:u, fwd_adj = lambda u:u, eps_test = 1e-4, zero = 1e-4, print_res = False):
    # check optimality conditions:
    print('CHECK OPTIMALITY')
    optimality = True
    N,M = res.u.shape
    l1vec = nfun('l1eps',l1eps_par=res.par.eps_TV,vdims=(2),npar=res.par.ptv/(N*M))
    grad = gradient(res.u.shape)

    ############ u optimality #################
    guc = grad.fwd( res.u - res.imsyn )
    gradu = grad.adj(l1vec.grad(guc))

    if res.par.application == 'inpainting':
        if (np.abs((res.mask-1))).any():
            if res.par.data_fidelity=='0.5l2_sq':
                gradu += (res.par.ld/(N*M))*(res.u-res.u0)*res.mask
                if np.max(np.absolute(gradu))>eps_test:
                    optimality = False
                    if print_res:
                        print('u test')
                        print( str(np.max(np.absolute(gradu[res.mask==0]))) +' is not zero')
            elif res.par.data_fidelity=='indicator':
                # check if grad_u is zero at unknown pixels
                if np.max(np.absolute(gradu[res.mask==0]))>eps_test:
                    optimality = False
                    if print_res:
                        print('u test')
                        print( str(np.max(np.absolute(gradu[res.mask==0]))) +' is not zero')
            else:
                print('unknown u penalty/data fidelity')

    elif res.par.application == 'supres':
        # need to check if grad_u is orth. to kernel of subsampling operator
        for i in range(res.u.shape[0]//res.par.sr_fac):
            for j in range(res.u.shape[1]//res.par.sr_fac):
                p = gradu[i:i+res.par.sr_fac,j:j+res.par.sr_fac]
                #check if p (part of grad_u) is constant vector
                if (np.max(p)-np.min(p)>eps_test):
                    optimality = False
                if print_res:
                    print('u test')
                    print( str(np.max(p)-np.min(p)) +' is not zero')

    elif res.par.application == 'jpeg':
        print('u optimality not implemented yet')

    else:
        gradu += (res.par.ld/(N*M))*fwd_adj( fwd_operator(res.u)-res.u0 )

        if np.max(np.absolute(gradu))>eps_test:
            optimality = False
            if print_res:
                print('u test')
                print( str(np.max(np.absolute(gradu))) +' is not zero')

    ############ c optimality #################

    for i in range(res.par.L):
        if i == 0:
            nxi = res.c[i].shape[0]
            nyi = res.c[i].shape[1]

            gucn = grad.adj(l1vec.grad(guc))
            gradc = -res.K.K[i].adj(gucn,res.D[i])

            if res.par.L>1:
                gradc += res.par.splitting/(nxi*nyi)*(res.c[i]-res.K.K[i+1].fwd(res.c[i+1],res.D[i+1]))
        else:
            nxi_1 = res.c[i-1].shape[0]
            nyi_1 = res.c[i-1].shape[1]

            nxi = res.c[i].shape[0]
            nyi = res.c[i].shape[1]

            gradc = res.par.splitting/(nxi_1*nyi_1)*res.K.K[i].adj(res.K.K[i].fwd(res.c[i],res.D[i])-res.c[i-1],res.D[i])

            if i < res.par.L-1:
                gradc += res.par.splitting/(nxi*nyi)*(res.c[i]-res.K.K[i+1].fwd(res.c[i+1],res.D[i+1]))

        # check if grad_c is -pcoeff if where c>0, pcoeff where c<0 and in [-pcoeff, pcoeff] where c=0
        # i.e. if -grad_c is in subdiff of c \mapsto pcoeff * | c |
        tmp = np.zeros(res.c[i].shape)
        tmp[res.c[i] > zero] = (res.par.pcoeff[i]/(nxi*nyi)+gradc)[res.c[i] > zero]
        if (np.absolute( tmp )).max()>eps_test:
            optimality = False
            if print_res:
                print('c test 1 failed')
                print(str((np.absolute(tmp)).max())+' is not zero')

        tmp = np.zeros(res.c[i].shape)
        tmp[res.c[i] < -zero] = (-res.par.pcoeff[i]/(nxi*nyi)+gradc)[res.c[i] < -zero]
        if (np.absolute( tmp )).max()>eps_test:
            optimality = False
            if print_res:
                print('c test 2 failed')
                print(str((np.absolute(tmp)).max())+' is not zero')

        tmp = np.copy(gradc)
        tmp[res.c[i] < -zero] = 0.0
        tmp[res.c[i] > zero] = 0.0
        if tmp.max() > res.par.pcoeff[i]/(nxi*nyi)+eps_test or tmp.min() < -res.par.pcoeff[i]/(nxi*nyi)-eps_test:
            optimality = False
            if print_res:
                print('c test 3 failed')
                print(str(tmp.max())+' is not <'+str(res.par.pcoeff[i]/(nxi*nyi))+' or')
                print(str(tmp.min())+' is not >'+str(-res.par.pcoeff[i]/(nxi*nyi)))

    ############ D optimality #################
    
    for i in range(res.par.L):
        if i == 0:
            nxi = res.c[i].shape[0]
            nyi = res.c[i].shape[1]

            gucn = grad.adj(l1vec.grad(guc))
            gradD = -res.K.K[i].adj_ker(gucn,res.c[0])

        else:
            nxi_1 = res.c[i-1].shape[0]
            nyi_1 = res.c[i-1].shape[1]

            nxi = res.c[i].shape[0]
            nyi = res.c[i].shape[1]

            gradD = res.par.splitting/(nxi_1*nyi_1)*res.K.K[i].adj_ker(res.K.K[i].fwd(res.c[i],res.D[i])-res.c[i-1],res.c[i])

        for k in range(res.par.nl[i]):
            if i==0:
                p = -gradD[:,:,k]
                D = (res.D[i])[:,:,k]
                ksz = res.par.ksz[i]

                if( (D*D).sum() < ksz*ksz - zero ):
                    # in this case, we need to check if p is a constant vektor, i.e., all entries are the same (orthogonal on zero mean)

                    if (np.absolute(p-p[0,0])).max() > eps_test:
                        optimality = False
                        if print_res:
                            print('D test 1 i=0')
                            print(str((np.absolute(p-p[0,0])).max())+' is not zero')

                elif( np.absolute((D*D).sum() - ksz*ksz) <= zero ):
                    # in this case, we need to check, if the projection onto zero mean of p is colinear and has same orientation
                    # as D

                    # project on zero mean
                    p = p - p.sum()/(p.shape[0]*p.shape[1])

                    # normalize:
                    p_norm = np.sqrt( (p*p).sum() )
                    if p_norm>zero:
                        p_unit = p/p_norm
                        D_unit = D/( np.sqrt( (D*D).sum() ))

                        if np.absolute( (p_unit*D_unit).sum()-1.0 ) > eps_test:
                            optimality = False
                            if print_res:
                                print('D test 2 i=0')
                                print('Scalar product is '+str((p_unit*D_unit).sum())+' and not 1')

                else:
                    print('D not feasible')
                    print('D 2-norm^2 = '+str((D*D).sum()))

            else:
                for l in range(res.par.nl[i-1]):
                    if i <= res.par.communication or k==l:
                        p = -gradD[:,:,k,l]
                        D = res.D[i][:,:,k,l]
                        ksz = res.par.ksz[i]

                        if( (D*D).sum() < ksz*ksz-zero ):
                            # in this case, we need to check if p=0

                            if np.absolute(p).max() > eps_test:
                                optimality = False
                                if print_res:
                                    print('D test 1 i>0')
                                    print(str((np.absolute(p)).max())+' is not zero')

                        elif( np.absolute((D*D).sum() - ksz*ksz) <= zero ):
                            # in this case, we need to check, if p is colinear and has same orientation
                            # as D

                            # normalize
                            p_norm = np.sqrt( (p*p).sum() )
                            if p_norm>zero:
                                p_unit = p/p_norm
                                D_unit = D/( np.sqrt( (D*D).sum() ))

                                if np.absolute((p_unit*D_unit).sum()-1.0) > eps_test:
                                    optimality = False
                                    if print_res:
                                        print('D test 2 i>0')
                                        print('Scalar product is '+str((p_unit*D_unit).sum())+' and not 1')

                        else:
                            print('D not feasible')
                            print('D 2-norm^2 = '+str((D*D).sum()))


    return optimality

###########################################################
# Operators ###############################################


#Image synthesis via convlution of coefficient and dictionary. Operator provides fwd, adj and adj_ker
class cp_conv(object):


    def __init__(self,dims,stride = 1):
                    
        #Set dimensions
        N = dims[0]
        M = dims[1]
        ksz = dims[2]
        nl = dims[3]
        

        self.nx = int(np.ceil((float(N)+float(ksz)-1.0)/float(stride)))
        self.ny = int(np.ceil((float(M)+float(ksz)-1.0)/float(stride)))


        #Input and output dimensions
        self.indim = [self.nx,self.ny,nl]
        self.indim2 = [ksz,ksz,nl]
        self.outdim = [N,M]
        
        #Kernel size, stride and number of filters
        self.ksz = ksz
        self.stride=stride
        self.nl = nl


    def fwd(self,c,k):
    
        if len(k.shape)==3:
            u=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)])

            for ff in range(self.nl):
                for i in range(self.ksz):
                    for j in range(self.ksz):
                        u[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride] += c[:,:,ff]*k[i,j,ff]
            uu = u[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1]]

        else:
            u=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)]+[k.shape[-1]])

            for layer_out in range(k.shape[-1]):
                for ff in range(self.nl):
                    for i in range(self.ksz):
                        for j in range(self.ksz):
                            u[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride,layer_out] += c[:,:,ff]*k[i,j,ff,layer_out]
            
            uu = u[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1],:]
        
        return uu



    def adj(self,u,k):


        c = np.zeros(self.indim)

        if len(k.shape)==3:
            uu=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)])
            uu[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1]]=u

            for ff in range(self.nl):
                for i in range(self.ksz):
                    for j in range(self.ksz):
                        c[:,:,ff] += uu[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride]*k[i,j,ff]
        else:
            for layer in range(k.shape[-1]):
                uu=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)])
                uu[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1]]=u[:,:,layer]

                for ff in range(self.nl):
                    for i in range(self.ksz):
                        for j in range(self.ksz):
                            c[:,:,ff] += uu[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride]*k[i,j,ff,layer]

        return c


        
    def adj_ker(self,u,c):


        if len(u.shape)==2:
            k = np.zeros(self.indim2)
            uu=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)])
            uu[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1]]=u

            for ff in range(self.nl):
                for i in range(self.ksz):
                    for j in range(self.ksz):
                        k[i,j,ff] = (uu[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride]*c[:,:,ff]).sum()

        else:
            k = np.zeros(self.indim2+[u.shape[-1]])
            for layer in range(u.shape[-1]):
                uu=np.zeros([self.indim[i]*self.stride+self.ksz-1 for i in range(2)])
                uu[self.ksz-1:self.ksz-1+self.outdim[0],self.ksz-1:self.ksz-1+self.outdim[1]]=u[:,:,layer]

                for ff in range(self.nl):
                    for i in range(self.ksz):
                        for j in range(self.ksz):
                            k[i,j,ff,layer] = (uu[i:i+self.nx*self.stride:self.stride,j:j+self.ny*self.stride:self.stride]*c[:,:,ff]).sum()

        return k


    def test_adj(self):

        #Coefficient adjoint
        k = np.random.rand(*self.indim2)

        fwd = lambda x: self.fwd(x,k)
        adj = lambda x: self.adj(x,k)

        test_adj(fwd,adj,self.indim,self.outdim)

        #Kernel adjoint
        c = np.random.rand(*self.indim)

        fwd = lambda x: self.fwd(c,x)
        adj = lambda x: self.adj_ker(x,c)


        test_adj(fwd,adj,self.indim2,self.outdim)


class cp_conv_nlayers(object):
    
    #K is list containing as entries cp_conv elements
    def __init__(self,K_list):
        self.K = K_list


    #comput the result of all consecutive convolutions
    def fwd(self,c,D):

        n=len(self.K)
        c_old = c

        for i in range(n-1):
            u=np.zeros(self.K[n-i-2].indim)
            for ii in range(self.K[n-i-2].nl):
                u[:,:,ii] = self.K[n-i-1].fwd( c_old , D[n-i-1][:,:,:,ii] )
            c_old=u

        u=self.K[0].fwd(c_old,D[0])

        return u

    def fwd_coeffs(self,c,D):

        n=len(self.K)
        c_old = c
        c_list=[c]

        for i in range(n-1):
            u=np.zeros(self.K[n-i-2].indim)
            for ii in range(self.K[n-i-2].nl):
                u[:,:,ii] = self.K[n-i-1].fwd( c_old , D[n-i-1][:,:,:,ii] )
            c_old=u
            c_list=[c_old]+c_list

        return c_list


    #gradients:

    #gradient of coefficients in all layers wrt coefficients in deepest layer
    def grad_c(self,u,D):
        n = len(self.K)
        c = self.K[0].adj(u,D[0])

        for i in range(1,n):
            v = np.zeros(self.K[i].indim)
            for ii in range(self.K[i-1].nl):
                v += self.K[i].adj( c[:,:,ii] , D[i][:,:,:,ii] )
            c = v

        return c

    def coeff_grad_c(self,layer,u,D):
        n = len(self.K)

        starting_layer=layer+1

        if layer==-1:
            c = self.K[0].adj(u,D[0])
            starting_layer=1
        else:
            c = u

        for i in range(starting_layer,n):
            v = np.zeros(self.K[i].indim)
            for ii in range(self.K[i-1].nl):
                v += self.K[i].adj( c[:,:,ii] , D[i][:,:,:,ii] )
            c = v

        return c


    #gradient of consecutive convolutions wrt. the atoms/filter kernels in all layers
    def grad_D(self,c,D,u):
        n = len(self.K)

        #compute coefficients on all layers, I think this is necessary. compare to automatic differentiation
        c_list = [c]
        c_old = c

        for i in range(n-1):
            c_new = np.zeros(self.K[n-2-i].indim)
            for ii in range(self.K[n-i-2].nl):
                c_new[:,:,ii] = self.K[n-1-i].fwd(c_old,D[n-1-i][:,:,:,ii])

            c_list = [c_new] + c_list
            c_old = c_new


        #copmute grad on all layers
        gradD0 = self.K[0].adj_ker( u , c_list[0] )
        gradD = [gradD0]

        gradD_tmp = self.K[0].adj( u , D[0] )

        for i in range(1,n):
            gradDi = np.zeros([*self.K[i].indim2,self.K[i-1].nl])
            for ii in range(self.K[i-1].nl):
                gradDi[:,:,:,ii] = self.K[i].adj_ker( gradD_tmp[:,:,ii] , c_list[i] )
            gradD.append(gradDi)

            gradD_tmp_new = np.zeros(self.K[i].indim)
            for ii in range(self.K[i-1].nl):
                gradD_tmp_new += self.K[i].adj( gradD_tmp[:,:,ii] , D[i][:,:,:,ii] )

            gradD_tmp = gradD_tmp_new

        return gradD


    def coeff_grad_D(self,layer,c,D,u):
        n = len(self.K)

        #compute coefficients on all layers, I think this is necessary. compare to automatic differentiation
        c_list = [c]
        c_old = c

        for i in range(n-2-layer):
            c_new = np.zeros(self.K[n-2-i].indim)
            for ii in range(self.K[n-i-2].nl):
                c_new[:,:,ii] = self.K[n-1-i].fwd(c_old,D[n-1-i][:,:,:,ii])

            c_list = [c_new] + c_list
            c_old = c_new


        #copmute grad on all layers
        gradD = []
        starting_layer = layer+1

        if layer==-1:
            gradD0 = self.K[0].adj_ker( u , c_list[0] )
            gradD = [gradD0]
            gradD_tmp = self.K[0].adj( u , D[0] )
            starting_layer = 1
        else:
            gradD_tmp = u

        for i in range(starting_layer,n):
            gradDi = np.zeros([*self.K[i].indim2,self.K[i-1].nl])
            for ii in range(self.K[i-1].nl):
                gradDi[:,:,:,ii] = self.K[i].adj_ker( gradD_tmp[:,:,ii] , c_list[i] )
            gradD.append(gradDi)

            gradD_tmp_new = np.zeros(self.K[i].indim)
            for ii in range(self.K[i-1].nl):
                gradD_tmp_new += self.K[i].adj( gradD_tmp[:,:,ii] , D[i][:,:,:,ii] )

            gradD_tmp = gradD_tmp_new

        return gradD




#Define OpenCL kernel code (in "C" language)
prgs = Program("""
#define INIT_INDICES \\
  int Nx = get_global_size(0); \\
  int Ny = get_global_size(1); \\
  int x = get_global_id(0); \\
  int y = get_global_id(1); \\
  int i = y*Nx + x;


__kernel void malplus3D(__global float *b , __global const float *a1 , const float s1 , __global const float *a2, const float s2) {
    INIT_INDICES

    int z = 0;

    if ( get_work_dim()==3 ) {
        z = get_global_id(2);
    }

    i = x + y*Nx + z*Nx*Ny;

    b[i] = a1[i]*s1 + a2[i]*s2;

}

__kernel void malplus4D(__global float *b , __global const float *a1 , const float s1 , __global const float *a2, const float s2) {
    int ksz = get_global_size(0);
    int nl = get_global_size(1);

    int y = get_global_id(0);
    int z = get_global_id(1);
    int layer_out = get_global_id(2);
    int i;


    for (int x = 0; x < ksz; ++x) {
        i = x + y*ksz + z*ksz*ksz + layer_out*ksz*ksz*nl;
        b[i] = s1*a1[i] + s2*a2[i];
    }
}

__kernel void malplus_mask(__global float *b , __global const float *a1 , const float s1 , __global const float *a2, const float s2, __global const int *mask) {
    INIT_INDICES

    int z = 0;

    if ( get_work_dim()==3 ) {
        z = get_global_id(2);
    }

    i = x + y*Nx + z*Nx*Ny;

    if (mask[i]==1)
    {
        b[i] = a1[i]*s1 + a2[i]*s2;
    } else
    {
        b[i] = b[i];
    }

}

__kernel void pw_mul(__global float *b , __global const float *a1 , __global const float *a2) {
    INIT_INDICES

    int z = 0;

    if ( get_work_dim()==3 ) {
        z = get_global_id(2);
    }

    i = x + y*Nx + z*Nx*Ny;
    
    b[i] = a1[i]*a2[i];

}


__kernel void blur(__global float *ublur , __global const float *u, __global const float *kern, const int ksz) {
    INIT_INDICES

    float val = 0.0f;
    int mid = (int)(ksz/2.0);

    int kmin = -mid;
    
    if (x-Nx+1 > kmin)
    {
        kmin = x-Nx+1;
    }

    int kmax = ksz-1-mid;
    if (kmax > x)
    {
        kmax = x;
    }

    int lmin = -mid;
    if (y-Ny+1 > lmin)
    {
        lmin = y-Ny+1;
    }

    int lmax = ksz-1-mid;
    if (lmax > y)
    {
        lmax = y;
    }

    for (int k = kmin; k < kmax+1 ; ++k)
    {
        for (int l = lmin; l < lmax+1 ; ++l)
        {

            val = val + u[(y-l)*Nx + (x-k)] * kern[(mid+l)*ksz + (mid+k)];
        }
    }

    ublur[i] = val;

}


__kernel void conv_cD_fwd(__global float *u , __global const float *c, __global const float *D , const int nl , const int nlout , const int ksz , const int stride , const int nx , const int ny) {
    INIT_INDICES

    // we compute convolution of moved point in order to get rid of the values wich contain zero padding
    int x_conv = x + ksz -1;
    int y_conv = y + ksz -1;


    if (nlout == 1)
    {
    float val = 0.0f;

    for (int ff = 0; ff < nl; ++ff)
    {
        for (int k = (int)ceil( ((float)(x_conv-ksz+1))/((float)stride) ); x_conv - k*stride > -1 ; ++k)
        {
            for (int l = (int)ceil( (float)((y_conv-ksz+1)/(float)stride) ); y_conv - l*stride > -1; ++l)
            {
                val = val + c[ff*nx*ny + l*nx + k] * D[ff*ksz*ksz + (y_conv-l*stride)*ksz + (x_conv-k*stride)];
            }
        }
    }

    u[i] = val;

    } else
    {
    int z = get_global_id(2);
    i =z*Nx*Ny + y*Nx + x;
    float val = 0.0f;
    for (int ff = 0; ff < nl; ++ff)
    {
        for (int k = (int)ceil( ((float)(x_conv-ksz+1))/((float)stride) ); x_conv - k*stride > -1 ; ++k)
        {
            for (int l = (int)ceil( (float)((y_conv-ksz+1)/(float)stride) ); y_conv - l*stride > -1; ++l)
            {
                val = val + c[ff*nx*ny + l*nx + k] * D[z*ksz*ksz*nl + ff*ksz*ksz + (y_conv-l*stride)*ksz + (x_conv-k*stride)];
            }
        }
    }
    u[i] = val;

    }  
    
}

__kernel void conv_adj(__global float *c , __global const float *u , __global const float *D , const int nl , const int ksz , const int stride , const int Nx, const int Ny) {
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);


    int z = 0;

    if (nl > 1)
    {
        z = get_global_id(2);
    }


    int i = z*nx*ny + y*nx + x;

    int imin = 0;
    int jmin = 0;

    if( x*stride+1-ksz > 0 ) {
        imin = x*stride+1-ksz;
    }



    if( y*stride+1-ksz > 0 ) {
        jmin = y*stride+1-ksz;
    }

    float val = 0.0f;

    for(int i = imin; i+ksz-1-x*stride < ksz && i < Nx; ++i) {
        for(int j = jmin; j+ksz-1-y*stride < ksz && j < Ny; ++j) {

            val = val + D[z*ksz*ksz + (j+ksz-1-y*stride)*ksz + (i+ksz-1-x*stride)] * u[j*Nx + i];
        }

    }
    c[i] = val;

}


__kernel void conv_adj_multifilter(__global float *c , __global const float *u , __global const float *D , const int nl_c , const int nl_u , const int ksz , const int stride , const int Nx, const int Ny) {
    int nx = get_global_size(0);
    int ny = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);


    int z = 0;

    if (nl_c > 1)
    {
        z = get_global_id(2);
    }


    int i = z*nx*ny + y*nx + x;

    int imin = 0;
    int jmin = 0;

    if( x*stride+1-ksz > 0 ) {
        imin = x*stride+1-ksz;
    }



    if( y*stride+1-ksz > 0 ) {
        jmin = y*stride+1-ksz;
    }

    float val = 0.0f;

    for(int layer = 0; layer < nl_u; ++layer) {
        for(int i = imin; i+ksz-1-x*stride < ksz && i < Nx; ++i) {
            for(int j = jmin; j+ksz-1-y*stride < ksz && j < Ny; ++j) {

                val = val + D[layer*ksz*ksz*nl_c + z*ksz*ksz + (j+ksz-1-y*stride)*ksz + (i+ksz-1-x*stride)] * u[layer*Nx*Ny + j*Nx + i];
            }

        }
    }
    c[i] = val;

}

__kernel void conv_adj_kernel(__global float *D , __global const float *u , __global const float *c , const int layer , const int nl , const int stride , const int Nx, const int Ny , const int nx, const int ny) {
    int ksz = get_global_size(0);
    int x = get_global_id(0);
    int y = get_global_id(1);

    int z = 0;

    if (nl>0) {
        z = get_global_id(2);
    }

    int i = z*ksz*ksz + y*ksz + x;

    float val = 0.0f;

    int kmin = 0;
    int lmin = 0;

    if ( ksz-1-x > 0 ){
        kmin = (int)ceil( (float)(ksz-1-x) / (float)(stride) );
    }

    if ( ksz-1-y > 0 ){
        lmin = (int)ceil( (float)(ksz-1-y) / (float)(stride) );
    }


    for(int k = kmin; x-ksz+1+k*stride < Nx; ++k) {
        for(int l = lmin; y-ksz+1+l*stride < Ny; ++l) {
            
            val = val + c[z*nx*ny + l*nx + k]*u[layer*Nx*Ny + (y-ksz+1+l*stride)*Nx + (x-ksz+1+k*stride)];
        }
    }

    D[i] = val;


}


__kernel void grad_fwd(__global float2 *p , __global const float *u) {
    INIT_INDICES


    if (x < Nx-1) p[i].s0 = u[i+1]-u[i];  else p[i].s0 = 0.0f;
    if (y < Ny-1) p[i].s1 = u[i+Nx]-u[i]; else p[i].s1 = 0.0f;

}

__kernel void grad_fwd_diff(__global float2 *p , __global const float *u , __global const float *v) {
    INIT_INDICES

    if (x < Nx-1) p[i].s0 = (u[i+1]-v[i+1])-(u[i]-v[i]);  else p[i].s0 = 0.0f;
    if (y < Ny-1) p[i].s1 = (u[i+Nx]-v[i+Nx])-(u[i]-v[i]); else p[i].s1 = 0.0f;

}

__kernel void grad_adj_l1vec_grad(__global float *v , __global const float2 *p , const float e , const float ld) {
    INIT_INDICES

    float2 val;
    val.s0 = p[i].s0 / sqrt(p[i].s0*p[i].s0 + p[i].s1*p[i].s1 + e);
    val.s1 = p[i].s1 / sqrt(p[i].s0*p[i].s0 + p[i].s1*p[i].s1 + e);

    if (x == Nx-1) val.s0 = 0.0f;
    if (x > 0) val.s0 -= p[i-1].s0 / sqrt(p[i-1].s0*p[i-1].s0 + p[i-1].s1*p[i-1].s1 + e);
    if (y == Ny-1) val.s1 = 0.0f;
    if (y > 0) val.s1 -= p[i-Nx].s1 / sqrt(p[i-Nx].s0*p[i-Nx].s0 + p[i-Nx].s1*p[i-Nx].s1 + e);

    v[i] = -ld*(val.s0 + val.s1);

}

__kernel void l1vec_val(__global float *v , __global const float2 *p , const float e) {
    INIT_INDICES

    v[i] = sqrt(p[i].s0*p[i].s0 + p[i].s1*p[i].s1 + e);

}


__kernel void inpainting_prox(__global float *u , __global const float *u0 , __global const int *mask) {
    INIT_INDICES

    if( mask[i] == 1 ) {
        u[i] = u0[i];
    }

}

__kernel void supres_prox(__global float *u , __global const float *u0, const int sr_fac) {
    int Nx = get_global_size(0);
    int Ny = get_global_size(1);
    int x = get_global_id(0);
    int y = get_global_id(1);


    int i = x + Nx*y;

    int xx = x*sr_fac;
    int yy = y*sr_fac;

    float mean = 0.0;

    for (int k = 0; k < sr_fac; ++k) {
        for (int l = 0; l < sr_fac; ++l) {
            mean = mean + u[ (xx+k) + (yy+l)*(Nx*sr_fac) ];
        }
    }

    mean = mean/( (float)sr_fac*(float)sr_fac );

    for (int k = 0; k < sr_fac; ++k) {
        for (int l = 0; l < sr_fac; ++l) {
            u[ (xx+k) + (yy+l)*(Nx*sr_fac) ] = u[ (xx+k) + (yy+l)*(Nx*sr_fac) ] - mean + u0[i];
        }
    }


}


// according to definition of Pock_17 IPalm

__kernel void l1_prox_c(__global float *c , const float tau , const int nl) {
    INIT_INDICES

    int z = 0;

    if(nl > 1) {
        z = get_global_id(2);
    }

    i = z*Nx*Ny + y*Nx + x;

    if( tau*c[i] > 1.0f ) {
        c[i] = c[i] - 1.0f/tau;
    } else if ( tau*c[i] < -1.0f ) {
        c[i] = c[i] + 1.0f/tau;
    } else {
        c[i] = 0.0f;
    }

}



//D0 prox: zero mean and 2-norm <= 1

__kernel void D0_prox(__global float *D , const int ksz) {

    int z = get_global_id(0);
    int i = z*ksz*ksz;

    float sum = 0.0f;

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            sum += D[i + l*ksz + k];
        }
    }

    sum = sum/((float)ksz*(float)ksz);

    float l2_norm_centered = 0.0f;

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            l2_norm_centered += (D[i + l*ksz + k]-sum)*(D[i + l*ksz + k]-sum);
        }
    }
    

    l2_norm_centered = sqrt(l2_norm_centered);

    float div = 1.0f;

    if ( l2_norm_centered > 1.0f ) {
        div = l2_norm_centered;
    }

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            D[i + l*ksz + k] = (D[i + l*ksz + k]-sum)/div;
        }
    }

}



__kernel void D_prox_l2_leq1(__global float *D, const int ksz) {

    int nl = get_global_size(0);
    int nl_out = get_global_size(1);

    int z = get_global_id(0);
    int layer_out = get_global_id(1);

    int i0 = z*ksz*ksz + layer_out*ksz*ksz*nl;

    float l2_norm = 0.0f;

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            l2_norm += D[i0 + l*ksz + k]*D[i0 + l*ksz + k];
        }
    }
    

    l2_norm = sqrt(l2_norm);

    float div = 1.0f;

    if ( l2_norm > 1.0f ) {
        div = l2_norm;
    }

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            D[i0 + l*ksz + k] = D[i0 + l*ksz + k]/div;
        }
    }


}

__kernel void D_prox_l2_leq1_no_com(__global float *D, const int ksz) {

    int nl = get_global_size(0);
    int nl_out = get_global_size(1);

    int z = get_global_id(0);
    int layer_out = get_global_id(1);

    int i0 = z*ksz*ksz + layer_out*ksz*ksz*nl;

    float l2_norm = 0.0f;

    if (z==layer_out) 
    {
    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            l2_norm += D[i0 + l*ksz + k]*D[i0 + l*ksz + k];
        }
    }
    

    l2_norm = sqrt(l2_norm);

    float div = 1.0f;

    if ( l2_norm > 1.0f ) {
        div = l2_norm;
    }

    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            D[i0 + l*ksz + k] = D[i0 + l*ksz + k]/div;
        }
    }
    }
    else {
    for (int k = 0; k < ksz; ++k) {
        for (int l = 0; l < ksz; ++l) {
            D[i0 + l*ksz + k] = 0.0f;
        }
    }
    }
}


#define W0 1.4142135623730951f
#define W1 1.3870398453221475f
#define W2 1.3065629648763766f
#define W3 1.1758756024193588f
#define W4 1.0000000000000002f
#define W5 0.7856949583871023f
#define W6 0.5411961001461971f
#define W7 0.2758993792829431f
#define SQRT1_2 0.70710678118654757f

#define LOAD_FLOAT8_1(v, k) (float8)(v[k][0], v[k][1], v[k][2], v[k][3], \
                                     v[k][4], v[k][5], v[k][6], v[k][7])
#define LOAD_FLOAT8_2(v, k) (float8)(v[0][k], v[1][k], v[2][k], v[3][k], \
                                     v[4][k], v[5][k], v[6][k], v[7][k])
#define STORE_FLOAT8_1(v, x, k) v[k][0] = x.s0; v[k][1] = x.s1; \
                                v[k][2] = x.s2; v[k][3] = x.s3; \
                                v[k][4] = x.s4; v[k][5] = x.s5; \
                                v[k][6] = x.s6; v[k][7] = x.s7;
#define STORE_FLOAT8_2(v, x, k) v[0][k] = x.s0; v[1][k] = x.s1; \
                                v[2][k] = x.s2; v[3][k] = x.s3; \
                                v[4][k] = x.s4; v[5][k] = x.s5; \
                                v[6][k] = x.s6; v[7][k] = x.s7;

// DCT part
///////////
float8 dct8(float8 x) {
  float4 a, b, t;

  a = x.s0123 + x.s7654;
  b = x.s0123 - x.s7654;
  t.s01 = (float2)(b.s1 + b.s2, b.s1 - b.s2)*SQRT1_2;
  t.s23 = b.s03 - t.s01;
  t.s01 += b.s03;
  x.s7135 = (float4)(W7*t.s0 - W1*t.s1, W7*t.s1 + W1*t.s0,
                     W3*t.s2 - W5*t.s3, W3*t.s3 + W5*t.s2);
  t = (float4)(a.s0 + a.s3, a.s1 + a.s2, a.s0 - a.s3, a.s1 - a.s2);
  x.s0462 = (float4)(t.s0 + t.s1, t.s0 - t.s1,
                     W6*t.s2 - W2*t.s3, W6*t.s3 + W2*t.s2);

  return x;
}

float8 idct8(float8 x) {
  float4 a, b, t;

  t = (float4)(x.s0 + x.s4, x.s0 - x.s4,
               W6*x.s6 + W2*x.s2, W6*x.s2 - W2*x.s6);
  a = (float4)(t.s0 + t.s2, t.s1 + t.s3, t.s1 - t.s3, t.s0 - t.s2);
  t = (float4)(W7*x.s7 + W1*x.s1, W7*x.s1 - W1*x.s7,
               W3*x.s3 + W5*x.s5, W3*x.s5 - W5*x.s3);
  b.s03 = (float2)(t.s0 + t.s2, t.s1 + t.s3);
  t.s01 -= t.s23;
  b.s12 = (float2)((t.s0 + t.s1), (t.s0 - t.s1))*SQRT1_2;
  x.s0123 = a + b;
  x.s7654 = a - b;

  return x;
}


#define DCT8x8(y,x) \
  y = dct8(x); \
  STORE_FLOAT8_1(temp, y, k) \
  barrier(CLK_LOCAL_MEM_FENCE); \
  y = dct8(LOAD_FLOAT8_2(temp, k)); \
  STORE_FLOAT8_2(temp, y, k) \
  barrier(CLK_LOCAL_MEM_FENCE); \
  y = LOAD_FLOAT8_1(temp, k);

#define IDCT8x8(y,x) \
  y = idct8(x); \
  STORE_FLOAT8_1(temp, y, k) \
  barrier(CLK_LOCAL_MEM_FENCE); \
  y = idct8(LOAD_FLOAT8_2(temp, k)); \
  STORE_FLOAT8_2(temp, y, k) \
  barrier(CLK_LOCAL_MEM_FENCE); \
  y = LOAD_FLOAT8_1(temp, k);

__kernel void prox_jpeg(__global float8 *u, __global float8 *Jmin, __global float8 *Jmax) {
  int i = get_global_id(1)*get_global_size(0) + get_global_id(0);
  int k = get_local_id(1);

  __local float temp[8][8];
  float8 x, y;

  x = u[i];
  DCT8x8(y,x)
  y = clamp(y*(1.0f/8.0f), Jmin[i], Jmax[i]);
  IDCT8x8(x,y)
  u[i] = x*(1.0f/8.0f);
}


"""
)


#Store OpenCL information
cl_contexts = ContextStore()
cl_devices = get_device_list()




def gen_reg(**par_in):

    ###########################################
    ###### Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})
    data_in.mask = 0 #Inpaint requires a mask
    ##Set data
    data_in.u0 = 0 #Direct corrupted image input
    data_in.orig = 0
    #Version information:
    par.version='Version 0'
    par.application = 'inpainting' #or one of 'denoising', 'deconvolution'
    par.imname = ''
    par.imnamejpeg = ''             #jpeg image for jpeg decompression
    par.data_is_corrupted = False # tells if the input is corrupted data or original image
    par.niter = 5000
    par.L = 3 #number of layers
    par.nl = [8,8,8] #number of filters
    par.ksz = [8,8,8] #kernel size
    par.stride = [1,2,2] #kernel size
    par.communication = 0 #For all D[i] with i > par.communication, we enforce D[i][:,:,k,l]=0 for k \neq l
    par.nu = 0.95
    par.ld = 100.0
    par.splitting = 2e3
    par.data_fidelity = 'indicator'    # data fidelity in case of l2-inpainting
    par.eps_TV = 0.05
    par.alpha_bar = 0.7
    par.alpha_u = par.alpha_bar
    par.alpha_c = par.alpha_bar
    par.alpha_D = par.alpha_bar
    par.eps_algo = (1.0-par.alpha_bar)/10.0
    par.blur_size = 9
    par.blur_sig = 0.25
    par.noise = 0.025
    par.inpaint_perc = 30 #perc of known pixels
    par.sr_fac = 4  # scaling factor for super-resolution
    par.check = 100 #The higher check is, the faster
    par.show_every = 0 #Show_every = 0 means to not show anything
    par.check_opt = 0 #every check_opt iterations we check optimality conditions, if 0, we never check
    par.opt_accuracy = 1e-4
    #Select which OpenCL device to use
    par.cl_device_nr = 0
    par.timing = False #Set true to enable timing
    par.L_max_count = 0
    par.mask = np.zeros([1])
    par.c_init = []
    par.u_init = 0
    par.D_init = []
    par.init_splitting_zero = True #if True, insitialization is such that the splitting penalty is zero
    ##Data and parameter parsing
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])
    par.eps_algo = (1.0-par.alpha_bar)/10.0

    par.pcoeff = [(1.0-par.nu)/(min(par.nu,1.0-par.nu)) for i in range(par.L)]
    par.ptv = par.nu/(min(par.nu,1.0-par.nu))

    res = output(par)

    ## Initialize OpenCL
    print('Available OpenCL devices:')
    for i in range(len(cl_devices)):
        print('[' + str(i) + ']: ' + cl_devices[i].name)
    print('Choosing: ' + cl_devices[par.cl_device_nr].name)    
    cl_device = cl_devices[par.cl_device_nr]
    #Create context and queue
    ctx = cl_contexts[cl_device]
    queue = cl.CommandQueue(ctx)
    #Build programs   
    prgs.build(ctx)

    ##################################################################################
    ###### initialze variables

    # Read image and initialize data
    if np.any(data_in.u0): #image as direct input
        u0 = np.copy(data_in.u0)
    elif len(par.imname)>0:
        u0 = imread(par.imname)
    elif np.any(data_in.orig):
        u0 = np.copy(data_in.orig)
    else:
        raise Exception('no data given')

    if len(u0.shape)==3:
            u0 = 0.3*u0[:,:,0] + 0.59*u0[:,:,1] + 0.11*u0[:,:,2]
    N,M = u0.shape


    np.random.seed(1)

    if (par.mask.any()>0):
        mask_inpainting = par.mask
    else:
        mask_inpainting = get_mask(shape=u0.shape,mtype='rand',perc=par.inpaint_perc)
    blur_obj = gconv([N,M],par.blur_size,par.blur_sig)
    blur_kernel = blur_obj.k

    if not par.data_is_corrupted: #then we corrupt the data
        data_in.orig = np.copy(u0)

        if par.application == 'deconvolution':
            u0 = blur_obj.fwd(u0)
            rg = np.abs(u0.max() - u0.min()) #Image range
            u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise

        elif par.application == 'denoising':
            rg = np.abs(u0.max() - u0.min()) #Image range
            u0 += np.random.normal(scale=par.noise*rg,size=u0.shape) #Add noise

        elif par.application == 'inpainting':
            u0[mask_inpainting==0]=0.0

        elif par.application == 'supres':
            # crop data such that dimensions are multiples of sr_fac:
            N = (N//par.sr_fac)*par.sr_fac
            M = (M//par.sr_fac)*par.sr_fac


            data_in.orig = u0[0:N,0:M]
            u0 = np.zeros([N//par.sr_fac, M//par.sr_fac])

            for i in range(par.sr_fac):
                for j in range(par.sr_fac):
                    u0[:,:] += data_in.orig[i::par.sr_fac,j::par.sr_fac]/(par.sr_fac*par.sr_fac)
        elif par.application == 'jpeg':
            print('.')
        else:
            raise Exception('Application invalid')
    else:
        if par.application == 'supres':
            N = N*par.sr_fac
            M = M*par.sr_fac

    if par.application == 'jpeg':
        (image_info, subsampling, Jmin, Jmax) = load_jpeg_data(par.imnamejpeg)
        Jmin = array.to_device(queue, Jmin[0].astype(np.float32, order='F'))
        Jmax = array.to_device(queue, Jmax[0].astype(np.float32, order='F'))
        u0d = array.zeros(queue, u0.shape, np.float32 , order='F', allocator=None)
        u0d = u0d.__add__(-128.0)
        shape = (Jmin.shape[1]//8, Jmin.shape[0])
        prgs.prox_jpeg(u0d.queue, shape, (1,8), u0d.data, Jmin.data, Jmax.data)
        u0d = u0d.__add__(128.0)
        u0d = u0d.__mul__(1.0/255.0)
        u0 = u0d.get()


    ##########################################################################################
    ##### Opencl functions
    

    blur_kerneld = array.to_device(queue, blur_kernel.astype(np.float32, order='F'))
    maskd_int = array.to_device(queue, mask_inpainting.astype(np.int32, order='F'))
    mask_float = mask_inpainting*1.0
    maskd_float = array.to_device(queue, mask_float.astype(np.float32, order='F'))

    ## allocate memory on gpu
    if (np.any(par.u_init)):
        print('initial u given')
        u_init = np.copy(par.u_init)
    else:
        if par.application == 'supres':
            u_init = np.zeros([N,M])
            for i in range(par.sr_fac):
                for j in range(par.sr_fac):
                    u_init[i::par.sr_fac,j::par.sr_fac] = u0[:,:]
        else:
            u_init = u0

    u0d = array.to_device(queue, u0.astype(np.float32, order='F'))
    ud = array.to_device(queue, u_init.astype(np.float32, order='F'))
    ud_old = ud.copy()
    u_max = array.max(ud.__abs__()).get()

    res.u0 = u0
    res.orig = data_in.orig
    res.mask = mask_inpainting

    # compute b=a1*s1 + a2*s2
    # this is faster than pyopencl version, since we already give the result in the function declaration.
    # with pyopencl new memory for the result is allocated on gpu within the function -> slower
    def malplus3D(b , a1 , s1 , a2 , s2):
        return prgs.malplus3D(b.queue , b.shape , None , b.data , a1.data , np.float32(s1) , a2.data , np.float32(s2))
    
    def malplus4D(b , a1 , s1 , a2 , s2):
        return prgs.malplus4D(b.queue , b[0,:,:,:].shape , None , b.data , a1.data , np.float32(s1) , a2.data , np.float32(s2))

    def malplus(b , a1 , s1 , a2 , s2, mask=np.array([])):
        if mask.any():
            prgs.malplus_mask(b.queue , b.shape , None , b.data , a1.data , np.float32(s1) , a2.data , np.float32(s2), mask.data)
        else:
            if(len(b.shape)<=3):
                prgs.malplus3D(b.queue , b.shape , None , b.data , a1.data , np.float32(s1) , a2.data , np.float32(s2))
            else:
                prgs.malplus4D(b.queue , b[0,:,:,:].shape , None , b.data , a1.data , np.float32(s1) , a2.data , np.float32(s2))

    def pw_mul2D(b , a1 , a2):
        return prgs.pw_mul(b.queue , b.shape , None , b.data , a1.data , a2.data)

    def blur(u_blur , u , kernel):

        ksz = kernel.shape[0]
        prgs.blur(u_blur.queue, u_blur.shape, None, u_blur.data , u.data , kernel.data , np.int32(ksz))
        return
    
    def blur_adj(u_blur , u , kernel):
        ksz = kernel.shape[0]
        return prgs.blur(u_blur.queue, u_blur.shape, None, u_blur.data , u.data , kernel.data , np.int32(ksz))


    def conv_cD_fwd_layer(u , c , D , nl , ksz , stride , nlout=1):
        return prgs.conv_cD_fwd(u.queue, u.shape, None , u.data, c.data, D.data , np.int32(nl) , np.int32(nlout) , np.int32(ksz) , np.int32(stride) , np.int32(c.shape[0]) , np.int32(c.shape[1]))

    # compute u = iterated convolution of c with all kernels in D
    def conv_cD_fwd(conv , u , c , c_list , D):

        n = len(D)
        c_list[n-1] = c

        for layer in range(n-1):
            conv_cD_fwd_layer(c_list[n-2-layer],c_list[n-1-layer], D[n-1-layer] , conv.K[n-1-layer].nl , conv.K[n-1-layer].ksz , conv.K[n-1-layer].stride , conv.K[n-1-layer-1].nl)

        D_layer = D[0]
        conv_cD_fwd_layer(u , c_list[0] , D_layer , par.nl[0] , par.ksz[0] , par.stride[0] , 1)
        
        return 0

    def conv_cD_fwd_coeffs(conv , c , c_list , D):

        c_list[len(c_list)-1] = c

        n = len(D)

        for layer in range(n-1):
            conv_cD_fwd_layer(c_list[n-2-layer],c_list[n-1-layer], D[n-1-layer] , conv.K[n-1-layer].nl , conv.K[n-1-layer].ksz , conv.K[n-1-layer].stride , conv.K[n-1-layer-1].nl)

        return 0


    def conv_adj(c , u , D , nl , ksz , stride , Nx , Ny):
        return prgs.conv_adj(c.queue, c.shape, None , c.data, u.data, D.data , np.int32(nl) , np.int32(ksz) , np.int32(stride) , np.int32(Nx) , np.int32(Ny))

    
    def conv_adj_multifilter(c , u , D , nl_c , nl_u , ksz , stride , Nx , Ny):
        return prgs.conv_adj_multifilter(c.queue, c.shape, None , c.data , u.data , D.data , np.int32(nl_c) , np.int32(nl_u) , np.int32(ksz) , np.int32(stride) , np.int32(Nx) , np.int32(Ny))


    def conv_adj_kernel(D , u , c , layer , nl , stride , Nx , Ny ,nx , ny):
        return prgs.conv_adj_kernel(D.queue, D.shape, None , D.data , u.data , c.data , np.int32(layer) , np.int32(nl) , np.int32(stride) , np.int32(Nx) , np.int32(Ny) , np.int32(nx) , np.int32(ny))

    def conv_grad_c(conv , gradc , c_list , u , D):
        n = len(D)
        if(n>1):
            conv_adj(c_list[0] , u , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , conv.K[0].outdim[0] , conv.K[0].outdim[1])
        else:
            conv_adj(gradc , u , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , conv.K[0].outdim[0] , conv.K[0].outdim[1])



        for i in range(1,n):
            if( i < n-1 ):
                conv_adj_multifilter(c_list[i] , c_list[i-1] , D[i] , conv.K[i].nl , conv.K[i-1].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1])
            else:
                conv_adj_multifilter(gradc , c_list[i-1] , D[i] , conv.K[i].nl , conv.K[i-1].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1])

        return 0


    def conv_grad_D(conv , gradD , c_list , c_list_tmp , D , D_tmp , u):
        n = len(D)

        #copmute grad on all layers
        conv_adj_kernel(gradD[0] , u , c_list[0] , 0 , conv.K[0].nl , conv.K[0].stride , u.shape[0] , u.shape[1] , conv.K[0].indim[0] , conv.K[0].indim[1])

        conv_adj(c_list_tmp[0] , u , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , conv.K[0].outdim[0] , conv.K[0].outdim[1])


        for i in range(1,n):

            for ii in range(conv.K[i-1].nl):
                conv_adj_kernel(D_tmp[i], c_list_tmp[i-1] , c_list[i] , ii , conv.K[i].nl , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1] , conv.K[i].indim[0] , conv.K[i].indim[1])
                gradD[i][:,:,:,ii] = D_tmp[i]

            conv_adj_multifilter(c_list_tmp[i] , c_list_tmp[i-1] , D[i] , conv.K[i].nl , conv.K[i-1].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1])

        return 0

    def conv_grad_Di(conv , gradD , layer , c , c_list , c_list_tmp , D , D_tmp , u):
        n = len(D)

        #copmute grad on all layers
        conv_adj_kernel(gradD[0] , u , c_list[0] , 0 , conv.K[0].nl , conv.K[0].stride , u.shape[0] , u.shape[1] , conv.K[0].indim[0] , conv.K[0].indim[1])

        conv_adj(c_list_tmp[0] , u , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , conv.K[0].outdim[0] , conv.K[0].outdim[1])


        for i in range(1,layer+1):

            for ii in range(conv.K[i-1].nl):
                conv_adj_kernel(D_tmp[i], c_list_tmp[i-1] , c_list[i] , ii , conv.K[i].nl , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1] , conv.K[i].indim[0] , conv.K[i].indim[1])
                gradD[i][:,:,:,ii] = D_tmp[i]

            conv_adj_multifilter(c_list_tmp[i] , c_list_tmp[i-1] , D[i] , conv.K[i].nl , conv.K[i-1].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1])

        return 0

    def grad_fwd(p , u):
        return prgs.grad_fwd(p.queue , u.shape , None , p.data , u.data)

    #compute grad(u-v)
    def grad_fwd_diff(p , u , v):
        return prgs.grad_fwd_diff(p.queue , u.shape , None , p.data , u.data , v.data)

    def grad_adj_l1vec_grad(v , p , e , ld):
        return prgs.grad_adj_l1vec_grad(v.queue , v.shape , None , v.data , p.data , np.float32(e) , np.float32(ld))


    ##################################################################################
    # application dependant functions for the iteration


    ################################## deconvolution ################################################
    if par.application == 'deconvolution':
        print('load deconvolution functions')
        def compute_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):
            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            blur(v_tmp , u , blur_kerneld)
            malplus(v_tmp , v_tmp , 1.0 , u0d , -1.0)

            # TV penalty and data fidelity
            E = (par.ptv/(N*M))*array.sum(v_tveps) + 0.5*(par.ld/(N*M))*array.dot(v_tmp, v_tmp)

            for i in range(par.L):
                # coefficient penalty
                E = E + (par.pcoeff[i]/(c[i].shape[0]*c[i].shape[1]))*array.sum(c[i].__abs__())

                # add splitting penalty
                if i>0:
                    # compute convolution
                    conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                    # compute discrepancy to next coefficient
                    malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                    # add penalty to energie
                    E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E.get()


        # v = generative part needs to be computed already
        def compute_smooth_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):
            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            blur(v_tmp , u , blur_kerneld)
            malplus(v_tmp , v_tmp , 1.0 , u0d , -1.0)

            # TV penalty and data fidelity
            E = (par.ptv/(N*M))*array.sum(v_tveps) + (0.5*par.ld/(N*M))*array.dot(v_tmp, v_tmp)

            for i in range(1,par.L):
                # add splitting penalty
                # compute convolution
                conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                # compute discrepancy to next coefficient
                malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                # add penalty to energie
                E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E

        def compute_grad_u(conv, gradu , gradu_tmp,  u , v , v_tmp, c, D, g):
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            grad_adj_l1vec_grad(gradu , g , par.eps_TV , par.ptv/(N*M))

            blur(v_tmp , u , blur_kerneld)
            malplus(v_tmp , v_tmp , 1.0 , u0d , -1.0)
            blur_adj(gradu_tmp , v_tmp , blur_kerneld)

            malplus(gradu , gradu , 1.0 , gradu_tmp , (par.ld/(N*M)))

            return 0

        def prox_u(u):
            return u

        fwd_blur = lambda u:scipy.signal.convolve2d(u, blur_kernel , mode='same', boundary='fill', fillvalue=0)
        def check_optimality_conditions(res):
            return optimality_conditions(res, fwd_operator = fwd_blur, fwd_adj = fwd_blur, eps_test = par.opt_accuracy, zero = 1e-4, print_res = True)

    ################################## denoising ################################################
    elif par.application == 'denoising':
        print('load denoising functions')
        
        def compute_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):
            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty and data fidelity
            malplus(v_tmp , u , 1.0 , u0d , -1.0)
            E = (par.ptv/(N*M))*array.sum(v_tveps) + (0.5*par.ld/(N*M))*array.dot(v_tmp, v_tmp)

            for i in range(par.L):
                # coefficient penalty
                E = E + (par.pcoeff[i]/(c[i].shape[0]*c[i].shape[1]))*array.sum(c[i].__abs__())

                # add splitting penalty
                if i>0:
                    # compute convolution
                    conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                    # compute discrepancy to next coefficient
                    malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                    # add penalty to energie
                    E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E.get()

        def compute_smooth_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):
            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty and data fidelity
            malplus(v_tmp , u , 1.0 , u0d , -1.0)
            E = (par.ptv/(N*M))*array.sum(v_tveps) + (0.5*par.ld/(N*M))*array.dot(v_tmp, v_tmp)

            # add splitting penalty
            for i in range(1,par.L):
                # compute convolution
                conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                # compute discrepancy to next coefficient
                malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                # add penalty to energie
                E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E

        def compute_grad_u(conv, gradu , gradu_tmp,  u , v , v_tmp, c, D, g):
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            grad_adj_l1vec_grad(gradu , g , par.eps_TV , par.ptv/(N*M))

            malplus(gradu , u , (par.ld/(N*M)) , gradu , 1.0)
            malplus(gradu , u0d , -(par.ld/(N*M)) , gradu , 1.0)

            return 0

        def prox_u(u):
            return u

        def check_optimality_conditions(res):
            return optimality_conditions(res, eps_test = par.opt_accuracy, zero = 1e-4, print_res = True)


    ################################## inpainting ################################################
    elif par.application == 'inpainting':
        print('load inpainting functions')

        def compute_energy(conv , u , v, v_tmp , g , v_tveps , c, imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(par.L):
                # coefficient penalty
                E = E + (par.pcoeff[i]/(c[i].shape[0]*c[i].shape[1]))*array.sum(c[i].__abs__())


                # add splitting penalty
                if i>0:
                    # compute convolution
                    conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                    # compute discrepancy to next coefficient
                    malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                    # add penalty to energie
                    E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])


            if par.data_fidelity=='0.5l2_sq':
                pw_mul2D(v_tmp , u-u0d , maskd_float)
                E = E + (0.5*par.ld/(N*M))*array.dot(v_tmp,v_tmp)

            return E.get()

        def compute_smooth_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(1,par.L):
                # compute convolution
                conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                # compute discrepancy to next coefficient
                malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                # add penalty to energie
                E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])


            if par.data_fidelity=='0.5l2_sq':
                pw_mul2D(v_tmp , u-u0d , maskd_float)
                E = E + (0.5*par.ld/(N*M))*array.dot(v_tmp,v_tmp)

            return E

        def compute_grad_u(conv, gradu , gradu_tmp,  u , v , v_tmp, c, D, g):
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            grad_adj_l1vec_grad(gradu , g , par.eps_TV , par.ptv/(N*M))

            if par.data_fidelity=='0.5l2_sq':
                malplus(gradu , u , (par.ld/(N*M)) , gradu , 1.0, maskd_int)
                malplus(gradu , u0d , -(par.ld/(N*M)) , gradu , 1.0, maskd_int)

            return 0

        def prox_u(u):
            if par.data_fidelity=='indicator':
                prgs.inpainting_prox(u.queue , u.shape , None , u.data , u0d.data , maskd_int.data)
            return u

        def check_optimality_conditions(res):
            return optimality_conditions(res, eps_test = par.opt_accuracy, zero = 1e-4, print_res = True)

    ################################## super-resolution ################################################
    elif par.application == 'supres':
        print('load supres functions')

        def compute_energy(conv , u , v, v_tmp , g , v_tveps , c, imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(par.L):
                # coefficient penalty
                E = E + (par.pcoeff[i]/(c[i].shape[0]*c[i].shape[1]))*array.sum(c[i].__abs__())

                # add splitting penalty
                if i>0:
                    # compute convolution
                    conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                    # compute discrepancy to next coefficient
                    malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                    # add penalty to energie
                    E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E.get()

        def compute_smooth_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(1,par.L):
                # compute convolution
                conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                # compute discrepancy to next coefficient
                malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                # add penalty to energie
                E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E

        def compute_grad_u(conv, gradu , gradu_tmp,  u , v , v_tmp, c, D, g):
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            grad_adj_l1vec_grad(gradu , g , par.eps_TV , par.ptv/(N*M))

            return 0

        def prox_u(u):
            
            prgs.supres_prox(u.queue , u0d.shape , None , u.data , u0d.data , np.int32(par.sr_fac))
            return u

        def check_optimality_conditions(res):
            return optimality_conditions(res, eps_test = par.opt_accuracy, zero = 1e-4, print_res = True)

    ################################## jpeg decompression ################################################
    elif par.application == 'jpeg':
        print('load jpeg functions')

        def compute_energy(conv , u , v, v_tmp , g , v_tveps , c, imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(par.L):
                # coefficient penalty
                E = E + (par.pcoeff[i]/(c[i].shape[0]*c[i].shape[1]))*array.sum(c[i].__abs__())

                # add splitting penalty
                if i>0:
                    # compute convolution
                    conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                    # compute discrepancy to next coefficient
                    malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                    # add penalty to energie
                    E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E.get()

        def compute_smooth_energy(conv , u , v, v_tmp , g , v_tveps , c , imsyn_list , D):

            # compute network output
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            prgs.l1vec_val(v_tveps.queue , v_tveps.shape , None , v_tveps.data , g.data , np.float32(par.eps_TV))

            # TV penalty
            E = (par.ptv/(N*M))*array.sum(v_tveps)

            for i in range(1,par.L):
                # compute convolution
                conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
                # compute discrepancy to next coefficient
                malplus(imsyn_list[i], imsyn_list[i], 1.0, c[i-1], -1.0)
                # add penalty to energie
                E = E + (par.splitting/(2.0*c[i-1].shape[0]*c[i-1].shape[1]))*array.dot(imsyn_list[i], imsyn_list[i])

            return E

        def compute_grad_u(conv, gradu , gradu_tmp,  u , v , v_tmp, c, D, g):
            conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
            grad_fwd_diff(g , u , v)
            grad_adj_l1vec_grad(gradu , g, par.eps_TV , par.ptv/(N*M))

            return 0

        def prox_u(u):
            shape = (Jmin.shape[1]//8, Jmin.shape[0])
            u = u.__mul__(255.0)
            u = u.__add__(-128.0)
            prgs.prox_jpeg(u.queue, shape, (1,8), u.data, Jmin.data, Jmax.data)
            u = u.__add__(128.0)
            u = u.__mul__((1.0/255.0))
            return u

        def check_optimality_conditions(res):
            return optimality_conditions(res, eps_test = par.opt_accuracy, zero = 1e-4, print_res = True)

    else:
        raise Exception('Application invalid')


    def compute_grad_c(gradc , conv , u , v , guc , gradu , c , imsyn_list , D):
        # contributen from TV part
        conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
        grad_fwd_diff(guc , v , u)
        grad_adj_l1vec_grad(gradu , guc , par.eps_TV , par.ptv/(N*M))
        conv_adj(gradc[0] , gradu , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , conv.K[0].outdim[0] , conv.K[0].outdim[1])

        # contribution from splitting part
        for i in range(1,par.L):
            # compute convolution
            conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
            # compute discrepancy to next coefficient
            malplus(imsyn_list[i], imsyn_list[i], par.splitting/(c[i-1].shape[0]*c[i-1].shape[1]), c[i-1], -par.splitting/(c[i-1].shape[0]*c[i-1].shape[1]))
            conv_adj_multifilter(gradc[i] , imsyn_list[i] , D[i] , conv.K[i].nl , conv.K[i-1].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1])
            malplus(gradc[i-1], gradc[i-1], 1.0, imsyn_list[i], -1.0)

        return 0

    def compute_grad_D(gradD , conv , u , v , guc , gradu , c, imsyn_list , D , D_tmp):
        # contributen from TV part
        conv_cD_fwd_layer(v , c[0] , D[0] , conv.K[0].nl , conv.K[0].ksz , conv.K[0].stride , 1)
        grad_fwd_diff(guc , v , u)
        grad_adj_l1vec_grad(gradu , guc , par.eps_TV , par.ptv/(N*M))
        conv_adj_kernel(gradD[0] , gradu , c[0] , 0 , conv.K[0].nl , conv.K[0].stride , gradu.shape[0] , gradu.shape[1] , conv.K[0].indim[0] , conv.K[0].indim[1])

        # contribution from splitting part
        for i in range(1,par.L):
            # compute convolution
            conv_cD_fwd_layer(imsyn_list[i] , c[i] , D[i] , conv.K[i].nl , conv.K[i].ksz , conv.K[i].stride , conv.K[i-1].nl)
            # compute discrepancy to next coefficient
            malplus(imsyn_list[i], imsyn_list[i], par.splitting/(c[i-1].shape[0]*c[i-1].shape[1]), c[i-1], -par.splitting/(c[i-1].shape[0]*c[i-1].shape[1]))

            for ii in range(conv.K[i-1].nl):
                conv_adj_kernel(D_tmp[i], imsyn_list[i] , c[i] , ii , conv.K[i].nl , conv.K[i].stride , conv.K[i].outdim[0] , conv.K[i].outdim[1] , conv.K[i].indim[0] , conv.K[i].indim[1])
                gradD[i][:,:,:,ii] = D_tmp[i]


        return 0

    def prox_c(c , tau , nl):
        return prgs.l1_prox_c(c.queue , c.shape , None , c.data , np.float32(tau) , np.int32(nl))

    def prox_D0(D):
        return prgs.D0_prox(D.queue , tuple([D.shape[2]]) ,None ,  D.data , np.int32(D.shape[0]))

    #depth is the index of D in D_list
    def prox_D(D,depth):
        if depth > par.communication:
            prgs.D_prox_l2_leq1_no_com(D.queue, D[0,0,:,:].shape , None , D.data , np.int32(D.shape[0]))
        else:
            prgs.D_prox_l2_leq1(D.queue, D[0,0,:,:].shape , None , D.data , np.int32(D.shape[0]))
        return


    #Operators and norms
    K_list=[]
    N_out = N
    M_out = M
    for i in range(par.L):
        Ki = cp_conv([N_out,M_out,par.ksz[i],par.nl[i]],stride=par.stride[i])
        K_list.append(Ki)
        N_out,M_out,nli = Ki.indim
    conv = cp_conv_nlayers(K_list)
    res.K = conv


    # atoms/kernels
    if ( len(par.D_init)==0 ):
        D0 = np.random.rand(par.ksz[0],par.ksz[0],par.nl[0])
        D0 -= D0.sum(axis=(0,1),keepdims=True)/(par.ksz[0]*par.ksz[0])
        D0 /= np.maximum(1.0, np.sqrt(np.square(D0).sum(axis=(0,1),keepdims=True)) )
        D_init = [D0]
        for i in range(1,par.L):
            Di = np.zeros([par.ksz[i],par.ksz[i],par.nl[i],par.nl[i-1]])
            for j in range(par.nl[i]):
                Di[:,:,j,j] = np.random.rand(par.ksz[i],par.ksz[i])
            #Di -= Di.sum(axis=(0,1),keepdims=True)/(par.ksz[0]*par.ksz[0])
            Di /= np.maximum(1.0, np.sqrt(np.square(Di).sum(axis=(0,1),keepdims=True)) )
            D_init.append(Di)

    else:
        print('initial D given')
        # fill first layers with given initial values of atoms
        D_init = []
        for i in range(len(par.D_init)):
            Di = np.copy(par.D_init[i])
            D_init.append(Di)
        # fill remaining layers with random atoms
        for i in range(len(par.D_init), par.L):
            Di = np.zeros([par.ksz[i],par.ksz[i],par.nl[i],par.nl[i-1]])
            for j in range(par.nl[i]):
                Di[:,:,j,j] = np.random.rand(par.ksz[i],par.ksz[i])
            #Di -= Di.sum(axis=(0,1),keepdims=True)/(par.ksz[0]*par.ksz[0])
            Di /= np.maximum(1.0, np.sqrt(np.square(Di).sum(axis=(0,1),keepdims=True)) )
            D_init.append(Di)

    Dd = []
    Dd_old = []
    for i in range(par.L):
        Dd.append(array.to_device(queue, D_init[i].astype(np.float32, order='F')))
        Dd_old.append(array.to_device(queue, D_init[i].astype(np.float32, order='F')))
    
    # coefficients
    cd = []
    cd_old = []
    cd_tmp = []
    for i in range(par.L):
            cd.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))
            cd_old.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))
            cd_tmp.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))
    vd = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)

    if ( len(par.c_init)>0 ):
        print('initial c given')
        #in this case, we assume we are given coefficients on layer above as par.c_init

        for i in range(len(par.c_init)):
            cd[i] = array.to_device(queue, par.c_init[i].astype(np.float32, order='F'))

        for i in range(len(par.c_init), par.L):
            conv_adj_multifilter(cd[i] , cd[i-1] , Dd[i] , conv.K[i].nl , conv.K[i-1].nl , par.ksz[i] , par.stride[i] , conv.K[i].outdim[0] , conv.K[i].outdim[1])

        if par.init_splitting_zero:
            for i in range(par.L-1):
                conv_cD_fwd_layer(cd[par.L-2-i], cd[par.L-1-i], Dd[par.L-1-i] , conv.K[par.L-1-i].nl , conv.K[par.L-1-i].ksz , conv.K[par.L-1-i].stride , conv.K[par.L-2-i].nl)

            conv_cD_fwd_layer(vd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)

            text_max = array.max(vd.__abs__()).get()
            
            if text_max>u_max:
                for i in range(par.L):
                    malplus(cd[i], cd[i], u_max/text_max, cd[i], 0.0)

        for i in range(par.L):
            cd_old[i] = cd[i].copy()

    else:
        #coefficients initialized as low res of input image projected on zero mean
        u = ud.get()
        image_zero_mean = u - u.sum()/(np.prod(u.shape))
        vd = array.to_device(queue, image_zero_mean.astype(np.float32, order='F'))

        conv_adj_multifilter(cd[0] , vd , Dd[0] , cd[0].shape[-1] , 1 , par.ksz[0] , par.stride[0] , ud.shape[0] , ud.shape[1])
        for i in range(1,par.L):
            conv_adj_multifilter(cd[i] , cd[i-1] , Dd[i] , cd[i].shape[-1] , cd[i-1].shape[-1] , par.ksz[i] , par.stride[i] , cd[i-1].shape[0] , cd[i-1].shape[1])

        for i in range(par.L-1):
            conv_cD_fwd_layer(cd[par.L-2-i], cd[par.L-1-i], Dd[par.L-1-i] , conv.K[par.L-1-i].nl , conv.K[par.L-1-i].ksz , conv.K[par.L-1-i].stride , conv.K[par.L-2-i].nl)

        conv_cD_fwd_layer(vd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)

        text_max = array.max(vd.__abs__()).get()

        if text_max>u_max:
            for i in range(par.L):
                malplus(cd[i], cd[i], u_max/text_max, cd[i], 0.0)

        for i in range(par.L):
            cd_old[i] = cd[i].copy()


    # necessary auxiliary variables. allocate here, to save time in iteration
    vd_tmp = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    vd_old = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    udI = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    gradud = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    gradud_tmp = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    gradud_old = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    gucd = array.zeros(queue,tuple([2,*vd.shape]),dtype=np.float32, order='F')
    gucd_old = array.zeros(queue,tuple([2,*vd.shape]),dtype=np.float32, order='F')
    u_tveps = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    g = array.zeros(queue,tuple([2,*vd.shape]),dtype=np.float32, order='F')
    v_tveps = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
    
    Dd_tmp = [array.zeros(queue, Dd[0].shape , np.float32 , order='F', allocator=None)]
    DdI=[]
    cdI = []
    imsynd_list = [array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)]
    gradcd = []
    gradcd_old = []
    gradDd=[]
    gradDd_old=[]
    for i in range(par.L):
            DdI.append(array.to_device(queue, D_init[i].astype(np.float32, order='F')))
            if (i>0):
                Dd_tmp.append(array.zeros(queue, Dd[i][:,:,:,0].shape , np.float32 , order='F', allocator=None))
                imsynd_list.append(array.zeros(queue, cd[i-1].shape , np.float32 , order='F', allocator=None))
            gradDd.append(array.zeros(queue, Dd[i].shape , np.float32 , order='F', allocator=None))
            gradDd_old.append(array.zeros(queue, Dd[i].shape , np.float32 , order='F', allocator=None))
            cdI.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))
            gradcd.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))
            gradcd_old.append(array.zeros(queue, tuple(conv.K[i].indim) , np.float32 , order='F', allocator=None))



    ####################################################################################################
    ####################################################################################################

    Lu = 1.0
    Lu_max = 1.0
    Lc = 1.0
    Lc_max = 1.0
    LD = 1.0
    LD_max = 1.0


    steps = np.zeros([par.niter+1,3]) #function for which algo is a descent method (\psi _{\delta_1, \delta_2} from iPALM paper)
    ob_val = np.zeros(par.niter+1)
    ob_val[0] = compute_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd)
    print('Iter: ' + str(0) + ', E: ' + str(ob_val[0]) + ', Lu: ' + str(Lu) +', Lc: ' + str(Lc) + ', LD: ' + str(LD))

    if par.show_every:
        closefig()
        textd = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
        conv_cD_fwd_layer(textd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)
        u_show = ud.get()
        text = textd.get()
        uu0 = res.u0
        imnormalize(uu0)
        imnormalize(u_show)
        imnormalize(text)
        res_visual = np.concatenate([u_show,text,u_show-text],axis=1)
        res_visual = u_show
        imshow(res_visual,title='Iter: 0')

        C=[]
        D=[]
        imsyn_list = []
        for i in range(par.L):
            C.append(cd[i].get())
            D.append(Dd[i].get())
        for i in range(par.L-1):
            conv_cD_fwd_layer(cd_tmp[par.L-2-i], cd[par.L-1-i], Dd[par.L-1-i] , conv.K[par.L-1-i].nl , conv.K[par.L-1-i].ksz , conv.K[par.L-1-i].stride , conv.K[par.L-2-i].nl)
            imsyn_list = [cd_tmp[par.L-2-i].get()] + imsyn_list

        D[0] = D[0][...,np.newaxis]
        show_network(C,D,text, imsyn_list,cmap='gray',vrange=[],save=False, fname = 'network')


    # Start iteration

    alpha_u = par.alpha_u
    alpha_c = par.alpha_c
    alpha_D = par.alpha_D
    
    #Initialization to allow zero-iterations
    delta_u = 1.0 
    delta_c = 1.0 
    delta_D = 1.0


    ################# start iPALM
    s=0

    k=0
    opt = False
    while (k<par.niter and not(opt)):

        ############################# update u #############################

        # extrapolate
        malplus(udI , ud , 1.0+alpha_u , ud_old , -alpha_u)
        malplus(ud_old , ud , 1.0 , ud , 0.0)

        #compute gradient of H wrt u
        compute_grad_u(conv, gradud , gradud_tmp,  udI , vd , vd_tmp, cd, Dd, g)
        compute_grad_u(conv, gradud_old , gradud_tmp,  ud_old , vd , vd_tmp, cd, Dd, g)

        E_old = compute_smooth_energy(conv , ud_old , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd) 
        
        if par.timing:
            queue.finish()
            eltime.u_desc += time.time() - t_tmp
            t_tmp = time.time()

        # backtracking in order to get right tau/Lipschitz const
        bt = 0
        desc = False
        

        if par.L_max_count:
            if np.remainder(k+1,par.L_max_count) == 0:
                Lu_max = 1.0
            u_step = (array.dot(udI-ud_old,udI-ud_old)).get()
            if not u_step==0.0:
                Lu = np.sqrt((array.dot(gradud-gradud_old,gradud-gradud_old)).get()/u_step)
                if( Lu > Lu_max):
                    Lu_max = Lu
            else:
                Lu = 1.0

        else:
            u_step = (array.dot(udI-ud_old,udI-ud_old)).get()
            if not u_step==0.0:
                Lu = np.sqrt((array.dot(gradud-gradud_old,gradud-gradud_old)).get()/u_step)
            else:
                Lu = 1.0


        while( bt<=10 and not desc ):

            #parameters according to Pock iPalm
            if par.L_max_count:
                delta_u = Lu_max*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            else:
                delta_u = Lu*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            tau_u = ((1.0+par.eps_algo)*delta_u+(1.0+alpha_u)*Lu)/(2.0-alpha_u)

            if tau_u == 0.0:
                tau_u = 1.0

            malplus(ud , udI , 1.0 , gradud , -1.0/tau_u)
            ud = prox_u(ud)

            #check condition for Lipschitz constant:

            #Q is quadratic approximation, E new energy
            Q = E_old + array.dot(gradud_old,(ud - ud_old)) + (0.5*Lu)*array.dot((ud - ud_old) , (ud-ud_old))

            # new energy
            E = compute_smooth_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd)

            if E.__le__(Q):
                if (Lu>1.0):
                    Lu = Lu/2.0
                desc = True
            else:
                Lu = Lu*2.0
                bt += 1


        ############################# update c #############################

        # extrapolate
        for i in range(par.L):
            # extrapolate
            malplus(cdI[i] , cd[i] , 1.0+alpha_c , cd_old[i] , -alpha_c)
            malplus(cd_old[i] , cd[i] , 1.0 , cd[i] , 0.0)

        #compute gradient of H wrt c

        compute_grad_c(gradcd , conv , ud , vd , gucd , gradud , cdI , imsynd_list , Dd)
        compute_grad_c(gradcd_old , conv , ud , vd , gucd , gradud , cd_old , imsynd_list , Dd)

        E_old = compute_smooth_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd_old , imsynd_list , Dd)

        # backtracking in order to get right tau/Lipschitz const
        bt = 0
        desc = False

        if par.L_max_count:
            if np.remainder(k+1,par.L_max_count) == 0:
                Lc_max = 1.0
            c_step = 0.0
            Lc = 0.0
            for i in range(par.L):
                c_step = c_step + (array.dot(cdI[i]-cd_old[i],cdI[i]-cd_old[i])).get()
                Lc = Lc + (array.dot(gradcd[i]-gradcd_old[i],gradcd[i]-gradcd_old[i])).get()
            if not c_step==0.0:
                Lc = np.sqrt(Lc/c_step)
                if( Lc > Lc_max ):
                    Lc_max = Lc
            else:
                Lc = 1.0
        else:
            c_step = 0.0
            Lc = 0.0
            for i in range(par.L):
                c_step = c_step + (array.dot(cdI[i]-cd_old[i],cdI[i]-cd_old[i])).get()
                Lc = Lc + (array.dot(gradcd[i]-gradcd_old[i],gradcd[i]-gradcd_old[i])).get()
            if not c_step==0.0:
                Lc = np.sqrt(Lc/c_step)
            else:
                Lc = 1.0

        while( bt<=10 and not desc ):

            #parameters according to Pock iPalm
            if par.L_max_count:
                delta_c = Lc_max*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            else:
                delta_c = Lc*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            tau_c = ((1.0+par.eps_algo)*delta_c+(1.0+alpha_c)*Lc)/(2.0-alpha_c)

            if tau_c == 0.0:
                tau_c = 1.0


            Q = E_old.with_queue(queue)

            for i in range(par.L):
                malplus(cd[i] , cdI[i] , 1.0 , gradcd[i] , -1.0/tau_c)
                #prox
                prox_c(cd[i] , tau_c/(par.pcoeff[i]/(cd[i].shape[0]*cd[i].shape[1])) , par.nl[i])

                Q = Q + array.dot(gradcd_old[i],(cd[i] - cd_old[i])) + (0.5*Lc)*array.dot((cd[i] - cd_old[i]) , (cd[i]-cd_old[i]))

            # new energy
            E = compute_smooth_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd)

            if E.__le__(Q):
                if(Lc>1.0):
                    Lc = Lc/2.0
                desc = True
            else:
                Lc = Lc*2.0
                bt += 1


        ############################# update D #############################
        
        for i in range(par.L):
            # extrapolate
            malplus(DdI[i] , Dd[i] , 1.0+alpha_D , Dd_old[i] , -alpha_D)
            malplus(Dd_old[i] , Dd[i] , 1.0 , Dd[i] , 0.0)

        #compute vd = convolution result
        compute_grad_D(gradDd , conv , ud , vd , gucd , gradud , cd, imsynd_list , DdI , Dd_tmp)
        compute_grad_D(gradDd_old , conv , ud , vd , gucd , gradud , cd, imsynd_list , Dd_old , Dd_tmp)

        E_old = compute_smooth_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd_old)

        # backtracking in order to get right tau/Lipschitz const
        bt = 0
        desc = False

        if par.L_max_count:
            if np.remainder(k+1,par.L_max_count) == 0:
                LD_max = 1.0
            D_step = 0.0
            LD = 0.0
            for i in range(par.L):
                D_step = D_step + (array.dot(DdI[i]-Dd_old[i],DdI[i]-Dd_old[i])).get()
                LD = LD + (array.dot(gradDd[i]-gradDd_old[i],gradDd[i]-gradDd_old[i])).get()
            if not D_step==0.0:
                LD = np.sqrt(LD/D_step)
                if( LD > LD_max ):
                    LD_max = LD
            else:
                LD = 1.0
        else:
            D_step = 0.0
            LD = 0.0
            for i in range(par.L):
                D_step = D_step + (array.dot(DdI[i]-Dd_old[i],DdI[i]-Dd_old[i])).get()
                LD = LD + (array.dot(gradDd[i]-gradDd_old[i],gradDd[i]-gradDd_old[i])).get()
            if not D_step==0.0:
                LD = np.sqrt(LD/D_step)
            else:
                LD = 1.0


        while( bt<=10 and not desc ):


            #parameters according to Pock iPalm
            if par.L_max_count:
                delta_D = LD_max*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            else:
                delta_D = LD*(3.0*par.alpha_bar)/(2.0*(1-par.eps_algo-par.alpha_bar))
            tau_D = ((1.0+par.eps_algo)*delta_D+(1.0+alpha_D)*LD)/(2.0-alpha_D)

            if tau_D == 0.0:
                tau_D = 1.0

            Q = E_old.with_queue(queue)

            for i in range(par.L):
                malplus(Dd[i] , DdI[i] , 1.0 , gradDd[i] , -1.0/tau_D)
                #prox
                if ( i==0 ):
                    prox_D0(Dd[0])
                else:
                    prox_D(Dd[i],i)

                Q = Q + array.dot(gradDd_old[i],(Dd[i] - Dd_old[i])) + (0.5*LD)*array.dot((Dd[i] - Dd_old[i]) , (Dd[i]-Dd_old[i]))

            #check condition for Lipschitz constant:

            # new energy
            E = compute_smooth_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd)

            if E.__le__(Q):
                if (LD>1.0):
                    LD = LD/2.0
                desc = True

            else:
                LD = LD*2.0
                bt += 1


        #compute new energy
        ob_val[k+1] = compute_energy(conv , ud , vd, vd_tmp , g , v_tveps , cd , imsynd_list , Dd)

        if par.check:
            if np.remainder(k,par.check) == 0:
                print('Iter: ' + str(k+1) + ', E: ' + str(ob_val[k+1]) + ', Lu: ' + str(Lu) + ', Lc: ' + str(Lc) + ', LD: ' + str(LD) + ', Lu_max: ' + str(Lu_max) + ', Lc_max: ' + str(Lc_max) + ', LD_max: ' + str(LD_max))
          
     
        if par.show_every:
            if np.remainder(k+1,par.show_every) == 0:
                closefig()
                textd = array.zeros(queue, ud.shape , np.float32 , order='F', allocator=None)
                conv_cD_fwd_layer(textd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)
                u_show = ud.get()
                text = textd.get()
                uu0 = res.u0
                imnormalize(uu0)
                imnormalize(u_show)
                imnormalize(text)

                res_visual = np.concatenate([u_show,text,u_show-text],axis=1)
                res_visual = u_show
                

                C=[]
                D=[]
                imsyn_list = []
                for i in range(par.L):
                    C.append(cd[i].get())
                    D.append(Dd[i].get())
                for i in range(par.L-1):
                    conv_cD_fwd_layer(cd_tmp[par.L-2-i], cd[par.L-1-i], Dd[par.L-1-i] , conv.K[par.L-1-i].nl , conv.K[par.L-1-i].ksz , conv.K[par.L-1-i].stride , conv.K[par.L-2-i].nl)
                    imsyn_list = [cd_tmp[par.L-2-i].get()] + imsyn_list

                D[0] = D[0][...,np.newaxis]
                show_network(C,D,text, imsyn_list,cmap='gray',vrange=[],save=False, fname = 'network')
                imshow(res_visual,title='Iter: '+str(k+1))

        if par.check_opt:
            if np.remainder(k+1,par.check_opt) == 0:
                D = []
                c = []
                for i in range(par.L):
                    D.append(Dd[i].get())
                    c.append(cd[i].get())
                res.D = D
                res.c = c
                res.u = ud.get()
                conv_cD_fwd_layer(vd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)
                res.imsyn = vd.get()
                opt = check_optimality_conditions(res)
                if opt == True:
                    print('Optimality satisfied!!!')



        k += 1

        steps[k,0] = array.dot( ud-ud_old, ud-ud_old).get()
        for i in range(par.L):
            steps[k,1] = steps[k,1] + array.dot( cd[i]-cd_old[i], cd[i]-cd_old[i]).get()
            steps[k,2] = steps[k,2] + array.dot( Dd[i]-Dd_old[i], Dd[i]-Dd_old[i]).get()

        if ( np.isinf( ob_val[k] ) ):
            k = par.niter
            print('Energy = inf')
        if ( np.isnan( ob_val[k] ) ):
            k = par.niter
            print('Energy = Nan')


    # set results
    res.u = ud.get()
    D = []
    c = []
    for i in range(par.L):
        D.append(Dd[i].get())
        c.append(cd[i].get())
    res.D = D
    res.c = c
    conv_cD_fwd_layer(vd , cd[0] , Dd[0] , par.nl[0] , par.ksz[0] , par.stride[0] , 1)
    res.imsyn = vd.get()

    imsyn_list = []
    for i in range(par.L-1):
        conv_cD_fwd_layer(cd_tmp[par.L-2-i], cd[par.L-1-i], Dd[par.L-1-i] , conv.K[par.L-1-i].nl , conv.K[par.L-1-i].ksz , conv.K[par.L-1-i].stride , conv.K[par.L-2-i].nl)
        imsyn_list = [cd_tmp[par.L-2-i].get()] + imsyn_list

    res.imsyn_list = imsyn_list

    if np.any(res.orig):
        res.psnr = np.round(psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=True),decimals=2)
        res.ssim = ssim(res.u, res.orig, data_range=res.u.max() - res.u.min())
    else:
        res.psnr = np.nan
        res.ssim = np.nan

    res.K = conv
    res.ob_val = ob_val[0:k]
    res.descent_obval = ob_val[0:k] + steps[0:k,0]*delta_u/2.0 + steps[0:k,1]*delta_c/2.0 + steps[0:k,2]*delta_D/2.0
    res.optimality = check_optimality_conditions(res)
    res.niter_done = k-1

    return res




def gen_reg_successively(**par_in):

    ###########################################
    ###### Initialize parameters and data input
    par = parameter({})
    data_in = data_input({})
    ##Set data
    data_in.u0 = 0 #Direct image input
    data_in.orig = 0
    #Version information:
    par.version='Version 0'
    par.application = 'inpainting' #or one of 'denoising', 'deconvolution'
    par.imname = ''
    par.imnamejpeg = ''
    par.data_is_corrupted = False
    par.L = 3 #number of layers
    par.nl = [8,8,8] #number of filters
    par.ksz = [8,8,8] #kernel size
    par.stride = [1,2,2] #kernel size
    par.niter = [500 for i in range(par.L-1)] + [4000] #par.niter[i] = number of iterations with (i+1) layer net
    par.communication = 0 #For all D[i] with i > par.communication, we enforce D[i][:,:,k,l]=0 for k \neq l
    par.nu = 0.95
    par.ld = 100
    par.splitting = 2e3
    par.data_fidelity = 'indicator'    # data fidelity in case of l2-inpainting
    par.eps_TV = 0.05
    par.alpha_bar = 0.7
    par.alpha_u = par.alpha_bar
    par.alpha_c = par.alpha_bar
    par.alpha_D = par.alpha_bar
    par.eps_algo = (1.0-par.alpha_bar)/10.0
    par.blur_size = 9
    par.blur_sig = 0.25
    par.noise = 0.025
    par.inpaint_perc = 30
    par.sr_fac = 4  # scaling factor for super-resolution
    par.check = 0 #The higher check is, the faster
    par.show_every = 0 #Show_every = 0 means to not show anything
    par.check_opt = 0 #every check_opt iterations we check optimality conditions, if 0, we never check
    #Select which OpenCL device to use
    par.opt_accuracy = 1e-4
    par.cl_device_nr = 0
    par.timing = False #Set true to enable timing
    par.L_max_count = 0
    par.mask = np.zeros([1])
    par.c_init = []
    par.init_splitting_zero = True #if True, initialization is such that the splitting penalty is zero
    ##Data and parameter parsing
    #Set parameters according to par_in
    par_parse(par_in,[par,data_in])
    par.eps_algo = (1.0-par.alpha_bar)/10.0


    nu = par.nu
    L = par.L
    ksz = par.ksz
    nl = par.nl
    stride = par.stride
    niters = par.niter
    show_every = par.show_every

    par.L = 1
    par.ksz = ksz[0:par.L]
    par.nl = nl[0:par.L]
    par.stride = stride[0:par.L]
    par.niter = niters[0]
    par.show_every = 0

    print('1 layer net')
    res = gen_reg(**par.__dict__, **data_in.__dict__)

    for i in range(2,L):
        print(str(i)+' layer net')
        par.L = i
        par.ksz = ksz[0:par.L]
        par.nl = nl[0:par.L]
        par.stride = stride[0:par.L]
        par.niter = niters[i-1]
        par.show_every = 0

        par.u_init = res.u
        par.c_init = res.c
        par.D_init = res.D
        res = gen_reg(**par.__dict__, **data_in.__dict__)

    par.L = L
    par.ksz = ksz
    par.nl = nl
    par.stride = stride
    par.niter = niters[-1]
    par.show_every = show_every
    par.nu = nu

    par.u_init = res.u
    par.c_init = res.c
    par.D_init = res.D

    print('Final run: '+ str(L)+' layer net')
    res = gen_reg(**par.__dict__, **data_in.__dict__)

    return res
