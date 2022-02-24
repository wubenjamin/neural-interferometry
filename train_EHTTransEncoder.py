
from copy import Error
import os
import argparse
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

#import pylab as plt
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from pytorch_lightning.plugins import DDPPlugin
#from context_encoder.vae import VAE
import context_encoder.encoders as m_encoder

from mlp import PosEncodedMLP_FiLM
#from data import AstroMNIST_Sparse
#from data_galaxy10 import Galaxy10_Dataset_Sparse
from data_continuous_EHT import EHTIM_Dataset
from data_ehtim_cont import make_dirtyim, make_im_torch
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

#import astro_utils

from scipy import interpolate
from numpy.fft import fft2, ifft2, fftshift, ifftshift

import socket
hostname= socket.gethostname()

if hostname!= 'NV':
    matplotlib.use('Agg')

class ImplicitAstro_Trans(pl.LightningModule):

    def __init__(
            self, args, 
            learning_rate=1e-4, L_embed=5, 
            input_encoding='fourier', sigma=2.5, 
            hidden_dims=[256,256], 
            latent_dim=64, kl_coeff=100.0, 
            num_fourier_coeff=32, batch_size=16, 
            input_size=28, model_checkpoint='', ngpu=None):

        super().__init__()
        
        self.save_hyperparameters()
     
        self.loss_func = nn.MSELoss(reduction='mean')
        self.loss_type = args.loss_type
        self.ngpu = ngpu
        self.use_unet= False 
        
        self.uv_dense_sparse_index=None
        self.num_fourier_coeff = num_fourier_coeff
        self.scale_loss_image= False #args.scale_loss_image

        

        if self.use_unet:
            import unet.unet_model as unet
            if self.loss_type=='unet_direct':
                self.UNET=unet.UNet(2, 1) #input: sparse visibility map; output: image
            else:
                self.UNET=unet.UNet(2, 2) #input: sparse visibility map; output: dense visibility map
        else:
            self.cond_mlp = PosEncodedMLP_FiLM(
                context_dim=latent_dim, 
                input_size=2, output_size=2, 
                hidden_dims=hidden_dims, 
                L_embed=L_embed, embed_type=input_encoding, 
                activation=nn.ReLU, 
                sigma=sigma, 
                context_type='Transformer')

            encoder = m_encoder.Visibility_Transformer(
                input_dim=2, #value dim 
                # PE dim for MLP, we are going to use the same PE as the MLP
                pe_dim=self.cond_mlp.input_size, 
                dim=512, depth=4, heads=16, 
                dim_head=512//16, 
                output_dim=latent_dim,
                dropout=.1, emb_dropout=0., 
                mlp_dim=512, 
                output_tokens=args.mlp_layers,
                has_global_token=False)
            self.pe_encoder = self.cond_mlp.embed_fun
            self.context_encoder = encoder
            print(self.cond_mlp)
            print(encoder)    

        self.norm_fact=None

        self.numEpoch = 0

        self.uv_arr= None
        self.U, self.V= None, None
        self.uv_coords_grid_query= None

        #validation plots
        self.folder_val = f'{args.val_fldr}/imgs/'
        self.folder_anim = f'{args.val_fldr}/anims/'
        os.makedirs(self.folder_val, exist_ok=True)
        os.makedirs(self.folder_anim, exist_ok=True)
        self.numPlot = 0
        self.plotFreq = 10        

        # testing
        self.test_iter = 0
        self.test_log_step = 50
        self.test_zs = []
        self.test_imgs = []
        self.test_fldr= f'../test_res/{args.exp_name}'

        
    def forward(self, x, z):
        pred_visibilities = self.cond_mlp(x, context=z)
        return pred_visibilities

        
    def _f(self, x):
        return ((x+0.5)%1)-0.5    
        
    def inference_w_conjugate(self, uv_coords, z, nF=0, return_np=True):
        halfspace = self._get_halfspace(uv_coords)
        
        # does this modify visibilities in place?
        uv_coords_flipped = self._flip_uv(uv_coords, halfspace)
     
        pred_visibilities = self(uv_coords_flipped, z) 
        pred_vis_real = pred_visibilities[:,:,0]
        pred_vis_imag = pred_visibilities[:,:,1]
        pred_vis_imag[halfspace] = -pred_vis_imag[halfspace] 

        if nF == 0: nF = self.hparams.num_fourier_coeff 

        pred_vis_imag = pred_vis_imag.reshape((-1, nF, nF))
        pred_vis_real = pred_vis_real.reshape((-1, nF, nF))

        pred_vis_imag[:,0,0] = 0
        pred_vis_imag[:,0,nF//2] = 0
        pred_vis_imag[:,nF//2,0] = 0
        pred_vis_imag[:,nF//2,nF//2] = 0

        if return_np:
            pred_fft =  pred_vis_real.detach().cpu().numpy() + 1j*pred_vis_imag.detach().cpu().numpy() 
        else:
            pred_fft =  pred_vis_real + 1j*pred_vis_imag

        # NEW: set border to zero to counteract weird border issues
        pred_fft[:,0,:] = 0.0
        pred_fft[:,:,0] = 0.0
        pred_fft[:,:,-1] = 0.0
        pred_fft[:,-1,:] = 0.0  
        
        return pred_fft                                   

    def _get_halfspace(self, uv_coords):
        #left_halfspace = torch.logical_and(uv_coords[:,0] > 0, uv_coords[:,1] > 0)
        left_halfspace = torch.logical_and(torch.logical_or(
            uv_coords[:,:,0] < 0, 
            torch.logical_and(uv_coords[:,:,0] == 0, uv_coords[:,:,1] > 0)), 
            ~torch.logical_and(uv_coords[:,:,0] == -.5, uv_coords[:,:,1] > 0))   
        
        return left_halfspace
    
    def _conjugate_vis(self, vis, halfspace):
        # take complex conjugate if flipped uv coords
        # so network doesn't receive confusing gradient information        
        vis[halfspace] = torch.conj(vis[halfspace])
        return vis
        
    def _flip_uv(self, uv_coords, halfspace):    

        halfspace_2d = torch.stack((halfspace, halfspace), axis=-1)
        uv_coords_flipped = torch.where(halfspace_2d, self._f(-uv_coords), uv_coords)  

        '''plt.figure()
        for i in range(3):
            plt.plot(uv_coords[i,:,0].cpu(), uv_coords[i,:,1].cpu(), 'x', alpha=0.5)
            plt.plot(uv_coords_flipped[i,:,0].cpu(), uv_coords_flipped[i,:,1].cpu(), 'o', alpha=0.5)
        plt.show()'''        

        return uv_coords_flipped

    def _recon_image_rfft(self, uv_dense, z, imsize, max_base, eht_fov, ):
        #get the query uv's
        B= uv_dense.shape[0]
        img_res=imsize[0]
        uv_dense_per=uv_dense[0]
        u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
        u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 + 1).to(u_dense)
        v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 + 1).to(u_dense)
        uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
        scale_ux= max_base * eht_fov/ img_res
        uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
        U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
        uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(B,1,1)
        #get predicted visibilities
        pred_visibilities = self(uv_coords_grid_query, z)  #Bx (HW) x 2
        pred_visibilities_map= torch.view_as_complex(pred_visibilities).reshape(B, U.shape[0], U.shape[1])
        img_recon = make_im_torch(uv_arr, pred_visibilities_map, img_res, eht_fov,
                                  norm_fact=self.norm_fact if self.norm_fact is not None else 1., 
                                  return_im=True)

        return img_recon


    def _step_image_loss(self, batch, batch_idx, num_zero_samples=0, loss_type='image',):
        '''
        forward pass then calculate the loss in the image domain
        we will use rfft to ensure that the values in the image domain are real
        '''

        uv_coords, uv_dense, vis_sparse, visibilities, img_0s = batch 
        img_res= img_0s.shape[-1]

        eht_fov = 1.4108078120287498e-09 
        max_base = 8368481300.0
        scale_ux= max_base * eht_fov/ img_res

        pe_uv = self.pe_encoder(uv_coords* scale_ux)
        inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
        z = self.context_encoder(inputs_encoder)

        B= uv_dense.shape[0]
        nF= int( uv_dense.shape[1]**.5 )


        #get the query uv's
        if self.uv_coords_grid_query is None:
            uv_dense_per=uv_dense[0]
            u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
            u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ).to(u_dense)
            v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ).to(u_dense)
            uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
            uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
            U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
            uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(B,1,1)
            self.uv_arr= uv_arr
            self.U, self.V= U,V
            self.uv_coords_grid_query= uv_coords_grid_query
            print('initilized self.uv_coords_grid_query')


        #get predicted visibilities
        pred_visibilities = self(self.uv_coords_grid_query, z)  #Bx (HW) x 2

        #image recon
        if self.norm_fact is None: # get the normalization factor, which is fixed given image/spectral domain dimensions
            visibilities_map= visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
            uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
            _, _, norm_fact = make_dirtyim(uv_dense_physical,
                                   visibilities_map.detach().cpu().numpy()[ 0, :, :].reshape(-1),
                                   img_res, eht_fov, return_im=True)
            self.norm_fact= norm_fact
            print('initiliazed the norm fact')

        #visibilities_map: B x len(u_dense) x len(v_dense)

        pred_visibilities_map= torch.view_as_complex(pred_visibilities).reshape(B, self.U.shape[0], self.U.shape[1])
        img_recon = make_im_torch(self.uv_arr, pred_visibilities_map, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)

        vis_maps = visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
        img_recon_gt= make_im_torch(self.uv_arr, vis_maps, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)

        #energy in the frequency space
        freq_norms = torch.sqrt(torch.sum(self.uv_coords_grid_query**2, -1))  
        abs_pred = torch.sqrt(pred_visibilities[:,:,0]**2 + pred_visibilities[:,:,1]**2)
        energy = torch.mean(freq_norms*abs_pred)

        if loss_type=='image':
            # loss= (img_0s - img_recon.real ).abs().mean()
            loss= (img_recon_gt.real - img_recon.real ).abs().mean()
            return 0., 0., loss, loss, energy
            # loss= pred_visibilities.abs().mean()
            # return 0., 0., loss, loss,0. 
        else:
            raise Error('undefined loss_type')

    def _step_unet(self, batch, batch_idx, num_zero_samples=0):
        # batch is a set of uv coords and complex visibilities
        uv_coords, uv_dense, vis_sparse, visibilities, img = batch 
        B,img_res= img.shape[0], img.shape[-1]
        

        ###
        #UNET
        if self.uv_dense_sparse_index is None:
            print('getting uv_dense_sparse_index...')
            uv_coords_per= uv_coords[0] #S,2
            uv_dense_per= uv_dense[0]#N,2
            uv_dense_sparse_index= []
            for i_sparse in range(uv_coords_per.shape[0]):
                uv_coord= uv_coords_per[i_sparse]
                uv_dense_equal= torch.logical_and(uv_dense_per[:,0]==uv_coord[0], uv_dense_per[:,1]==uv_coord[1])
                uv_dense_sparse_index.append( uv_dense_equal.nonzero() )
            uv_dense_sparse_index= torch.LongTensor(uv_dense_sparse_index,).to(uv_coords.device)
            print('done')
            self.uv_dense_sparse_index= uv_dense_sparse_index

        #get the sparse visibility image (input to the UNet)
        uv_dense_sparse_map= torch.zeros((uv_coords.shape[0], self.num_fourier_coeff**2, 2), ).to(uv_coords.device)
        uv_dense_sparse_map[:,self.uv_dense_sparse_index,: ]=vis_sparse
        uv_dense_sparse_map= uv_dense_sparse_map.permute(0, 2, 1).contiguous().reshape(-1, 2, self.num_fourier_coeff, self.num_fourier_coeff)
        uv_dense_unet_output= self.UNET(uv_dense_sparse_map)# B,2,H,W or B,1,H,W
        ###

        if self.loss_type in ('image', 'image_spectral'):
            eht_fov = 1.4108078120287498e-09 
            max_base = 8368481300.0
            scale_ux= max_base * eht_fov/ img_res
            #get the query uv's
            if self.uv_coords_grid_query is None:
                uv_dense_per=uv_dense[0]
                u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
                u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ).to(u_dense)
                v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ).to(u_dense)
                uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
                uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
                U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
                uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(uv_coords.shape[0],1,1)
                self.uv_arr= uv_arr
                self.U, self.V= U,V
                self.uv_coords_grid_query= uv_coords_grid_query
                print('initilized self.uv_coords_grid_query')
            #image recon
            if self.norm_fact is None: # get the normalization factor, which is fixed given image/spectral domain dimensions
                visibilities_map= visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
                uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
                _, _, norm_fact = make_dirtyim(uv_dense_physical,
                                       visibilities_map.detach().cpu().numpy()[ 0, :, :].reshape(-1),
                                       img_res, eht_fov, return_im=True)
                self.norm_fact= norm_fact
                print('initiliazed the norm fact')
            uv_dense_sparse_recon = torch.view_as_complex(uv_dense_unet_output.permute(0,2,3,1).contiguous()) # B,H,W
            img_recon = make_im_torch(self.uv_arr, uv_dense_sparse_recon, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            #image recon loss
            loss= (img - img_recon.real ).abs().mean()
            return 0., 0., loss, loss, 0.

        elif self.loss_type in ('spectral'):
            #spectral loss
            vis_mat= torch.view_as_real(visibilities)
            real_loss = self.loss_func(vis_mat[...,0], uv_dense_unet_output[:,0,...].reshape(B,-1) )
            imaginary_loss = self.loss_func(vis_mat[...,1], uv_dense_unet_output[:,1,...].reshape(B,-1)) 
            loss = real_loss + imaginary_loss 
            return real_loss, imaginary_loss, 0, loss, 0.
        
        elif self.loss_type in ('unet_direct'):
            loss= (img- uv_dense_unet_output.squeeze(1)).abs().mean()
            return 0., 0., loss, loss, 0.

        else:
            raise Error(f'undefined loss_type {self.loss_type}')

            


    def _step(self, batch, batch_idx, num_zero_samples=0):
        # batch is a set of uv coords and complex visibilities
        uv_coords, uv_dense, vis_sparse, visibilities, img = batch 


        pe_uv = self.pe_encoder(uv_coords)
        inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
        z = self.context_encoder(inputs_encoder)
        
        halfspace = self._get_halfspace(uv_dense)
        uv_coords_flipped = self._flip_uv(uv_dense, halfspace)
        vis_conj = self._conjugate_vis(visibilities, halfspace)

        # now condition MLP on z #
        pred_visibilities = self(uv_coords_flipped, z)  #Bx HW x2
        vis_real = vis_conj.real.float()
        vis_imag = vis_conj.imag.float()
        
        freq_norms = torch.sqrt(torch.sum(uv_dense**2, -1))  
        abs_pred = torch.sqrt(pred_visibilities[:,:,0]**2 + pred_visibilities[:,:,1]**2)
        energy = torch.mean(freq_norms*abs_pred)
        
        real_loss = self.loss_func(vis_real, pred_visibilities[:,:,0])
        imaginary_loss = self.loss_func(vis_imag, pred_visibilities[:,:,1]) 

        loss = real_loss + imaginary_loss 

        
        return real_loss, imaginary_loss, loss, energy

    def training_step(self, batch, batch_idx, if_profile=False):

        if if_profile:
            print('start: training step')
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if self.use_unet:
            real_loss, imaginary_loss, image_loss, loss, energy= self._step_unet(batch, batch_idx)
        elif self.loss_type=='spectral':
            real_loss, imaginary_loss, loss, energy= self._step(batch, batch_idx)
        elif self.loss_type=='image' or self.loss_type=='image_spectral':
            real_loss, imaginary_loss, image_loss, loss, energy= self._step_image_loss(batch, batch_idx, loss_type=self.loss_type)
            self.log('train/image_loss', image_loss,
                     sync_dist=True if self.ngpu > 1 else False,
                     rank_zero_only=True if self.ngpu > 1 else False,)
            
        log_vars = [real_loss,
                    loss,
                    imaginary_loss,
                    energy]
        log_names = ['train/real_loss',
                     'train/total_loss',
                     'train/imaginary_loss',
                     'train_metadata/energy']
        
        for name, var in zip(log_names, log_vars):
            self.log(name, var,
                     sync_dist=True if self.ngpu > 1 else False,
                     rank_zero_only=True if self.ngpu > 1 else False)

        if if_profile:
            end.record()
            torch.cuda.synchronize()
            print(f'one training step took {start.elapsed_time(end)}')
            print('end: training step\n')

        return loss

    def test_step(self, batch, batch_idx):
        os.makedirs(self.test_fldr, exist_ok=True)

        uv_coords, uv_dense, vis_sparse, visibilities, img = batch 
        B= uv_dense.shape[0]
        vis_maps = visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
        eht_fov  = 1.4108078120287498e-09 
        max_base = 8368481300.0
        # img_res = self.hparams.input_size 
        img_res = img.shape[-1]
        nF = self.hparams.num_fourier_coeff 
        scale_ux= max_base * eht_fov/ img_res

        if not self.use_unet:
            if self.loss_type=='spectral':
                pe_uv = self.pe_encoder(uv_coords)
            else:
                pe_uv = self.pe_encoder(uv_coords*scale_ux)
            inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
            z = self.context_encoder(inputs_encoder)

        #get the query uv's
        if self.uv_coords_grid_query is None:
            uv_dense_per=uv_dense[0]
            u_dense, v_dense= uv_dense_per[:,0].unique(), uv_dense_per[:,1].unique()
            u_dense= torch.linspace( u_dense.min(), u_dense.max(), len(u_dense)//2 * 2 ).to(u_dense)
            v_dense= torch.linspace( v_dense.min(), v_dense.max(), len(v_dense)//2 * 2 ).to(u_dense)
            uv_arr= torch.cat([u_dense.unsqueeze(-1), v_dense.unsqueeze(-1)], dim=-1)
            uv_arr= ((uv_arr+.5) * 2 -1.) * scale_ux # scaled input
            U,V= torch.meshgrid(uv_arr[:,0], uv_arr[:,1])
            uv_coords_grid_query= torch.cat((U.reshape(-1,1), V.reshape(-1,1)), dim=-1).unsqueeze(0).repeat(B,1,1)
            self.uv_arr= uv_arr
            self.U, self.V= U,V
            self.uv_coords_grid_query= uv_coords_grid_query
            print('initilized self.uv_coords_grid_query')
        if self.norm_fact is None: # get the normalization factor, which is fixed given image/spectral domain dimensions
            visibilities_map= visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)
            uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
            _, _, norm_fact = make_dirtyim(uv_dense_physical,
                                   visibilities_map.detach().cpu().numpy()[ 0, :, :].reshape(-1),
                                   img_res, eht_fov, return_im=True)
            self.norm_fact= norm_fact
            print('initiliazed the norm fact')

        # reconstruct dirty image via eht-im
        # constants for our current datasets; TODO: get from metadata

        if self.use_unet:
            if self.uv_dense_sparse_index is None:
                print('getting uv_dense_sparse_index...')
                uv_coords_per= uv_coords[0] #S,2
                uv_dense_per= uv_dense[0]#N,2
                uv_dense_sparse_index= []
                for i_sparse in range(uv_coords_per.shape[0]):
                    uv_coord= uv_coords_per[i_sparse]
                    uv_dense_equal= torch.logical_and(uv_dense_per[:,0]==uv_coord[0], uv_dense_per[:,1]==uv_coord[1])
                    uv_dense_sparse_index.append( uv_dense_equal.nonzero() )
                uv_dense_sparse_index= torch.LongTensor(uv_dense_sparse_index,).to(uv_coords.device)
                print('done')
                self.uv_dense_sparse_index= uv_dense_sparse_index
            #get the sparse visibility image (input to the UNet)
            uv_dense_sparse_map= torch.zeros((uv_coords.shape[0], self.num_fourier_coeff**2, 2), ).to(uv_coords.device)
            uv_dense_sparse_map[:,self.uv_dense_sparse_index,: ]=vis_sparse
            uv_dense_sparse_map= uv_dense_sparse_map.permute(0, 2, 1).contiguous().reshape(-1, 2, self.num_fourier_coeff, self.num_fourier_coeff)
            uv_dense_unet_output= self.UNET(uv_dense_sparse_map)# B,2,H,W
            uv_dense_unet_output= torch.view_as_complex(uv_dense_unet_output.permute(0,2,3,1).contiguous()) # B,H,W
            img_recon = make_im_torch(self.uv_arr, uv_dense_unet_output, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon_gt = img
            img_recon = (img_recon.real).float() / img_recon.abs().max()
            img_recon_gt = (img_recon_gt).float() / img_recon_gt.abs().max()
            
        elif self.loss_type == 'spectral':
            pred_fft = self.inference_w_conjugate(uv_dense, z, return_np=False)
            uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
            img_recon   = make_im_torch(self.uv_arr, pred_fft, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon_gt= make_im_torch(self.uv_arr, vis_maps, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon = (img_recon.real).float()
            img_recon_gt = (img_recon_gt.real).float()

        elif self.loss_type in ('image', 'image_spectral'):
            pred_visibilities = self(self.uv_coords_grid_query, z)  #Bx (HW) x 2
            pred_visibilities_map= torch.view_as_complex(pred_visibilities).reshape(B, self.U.shape[0], self.U.shape[1])
            img_recon = make_im_torch(self.uv_arr, pred_visibilities_map, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon_gt= make_im_torch(self.uv_arr, vis_maps, img_res, eht_fov, norm_fact=self.norm_fact, return_im=True)
            img_recon = (img_recon.real).float() / img_recon.abs().max()
            img_recon_gt = (img_recon_gt.real).float() / img_recon_gt.abs().max()

        img_log= torch.cat([img_recon_gt.reshape(-1, img.shape[-1]).cpu(), img_recon.reshape(-1, img_recon.shape[-1]).cpu()], dim=-1)
        plt.imsave(f'{self.test_fldr}/img_recon_{batch_idx:05d}.png', img_log,cmap='hot')

    def validation_step(self, batch, batch_idx):
        uv_coords, uv_dense, vis_sparse, visibilities, img = batch 
        vis_maps = visibilities.reshape(-1, self.num_fourier_coeff, self.num_fourier_coeff)

        eht_fov  = 1.4108078120287498e-09 
        max_base = 8368481300.0
        img_res=img.shape[-1]
        scale_ux= max_base * eht_fov/ img_res


        if batch_idx==0 and self.ngpu<=1 and not self.use_unet: # the first batch of samples of the epoch
            if self.loss_type=='spectral':
                pe_uv = self.pe_encoder(uv_coords)
            elif self.loss_type in ('image', 'image_spectral'):
                pe_uv = self.pe_encoder(uv_coords*scale_ux)
            inputs_encoder = torch.cat([pe_uv, vis_sparse], dim=-1)
            z = self.context_encoder(inputs_encoder)
            img_res = self.hparams.input_size 
            nF = self.hparams.num_fourier_coeff 
        
            # reconstruct dirty image via eht-im
            # constants for our current datasets; TODO: get from metadata

            if self.loss_type == 'spectral':
                pred_fft = self.inference_w_conjugate(uv_dense, z)
                img_recon_raw = np.zeros((self.hparams.batch_size, img_res, img_res), dtype=np.complex128)
                img_recon_gt = np.zeros((self.hparams.batch_size, img_res, img_res), dtype=np.complex128)
                uv_dense_physical = (uv_dense.detach().cpu().numpy()[0,:,:] +0.5)*(2*max_base) - (max_base)
                for i in range(self.hparams.batch_size):
                    img_recon_raw[i,:,:] = make_dirtyim(uv_dense_physical, 
                                                        pred_fft[i,:,:].reshape(-1), 
                                                        img_res, eht_fov).imarr()
                    img_recon_gt[i,:,:] = make_dirtyim(uv_dense_physical, 
                                                       vis_maps.detach().cpu().numpy()[i,:,:].reshape(-1), 
                                                       img_res, eht_fov).imarr()
                    img_recon = torch.from_numpy(img_recon_raw).float() 

                # add diagnostic figure:
                fig = plt.figure(figsize=(14,16))
                gs = gridspec.GridSpec(6,4, hspace=0.25, wspace=0.25)
                for i in range(4):
                    gt_img = np.real(img_recon_gt[i,:,:]).astype('float32')
                    #gt_vis = np.abs(vis_maps.detach().cpu().numpy()[i,:,:])
                    gt_vis_re = np.real(vis_maps.detach().cpu().numpy()[i,:,:])
                    gt_vis_im = np.imag(vis_maps.detach().cpu().numpy()[i,:,:])
                    pred_img = img_recon.detach().cpu().numpy()[i,:,:]
                    #pred_vis = np.abs(pred_fft[i,:,:])
                    pred_vis_re = np.real(pred_fft[i,:,:])
                    pred_vis_im = np.imag(pred_fft[i,:,:])
                    vmin_img, vmax_img = np.min(gt_img), np.max(gt_img)
                    #vmin_vis, vmax_vis = np.min(gt_vis), np.max(gt_vis)
                    vmin_vis_re, vmax_vis_re = np.min(gt_vis_re), np.max(gt_vis_re)
                    vmin_vis_im, vmax_vis_im = np.min(gt_vis_im), np.max(gt_vis_im)

                    # GT spatial image
                    ax = plt.subplot(gs[0,i])
                    ax.set_title("[%i] Dense Grid Image \n$ \mathscr{F}^{-1}_{NU}[\mathcal{V}_{grid}(u,v)]$" %i, fontsize=10)
                    #im = ax.imshow(np.abs(img_recon_gt[i,:,:]))
                    im = ax.imshow(gt_img, vmin=vmin_img, vmax=vmax_img, cmap='afmhot')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    # GT spectral (real)
                    ax = plt.subplot(gs[1,i])
                    ax.set_title("[%i] Dense Grid Fourier (Re) \n Re(${\mathcal{V}_{grid}(u,v)}$)" %i, fontsize=10)
                    im = ax.imshow(gt_vis_re, vmin=vmin_vis_re, vmax=vmax_vis_re)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    # GT spectral (imaginary)
                    ax = plt.subplot(gs[2,i])
                    ax.set_title("[%i] Dense Grid Fourier (Im) \n Im(${\mathcal{V}_{grid}(u,v)}$)" %i, fontsize=10)
                    im = ax.imshow(gt_vis_im, vmin=vmin_vis_im, vmax=vmax_vis_im)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    # Predicted spectral (real)
                    ax = plt.subplot(gs[3,i])
                    ax.set_title("[%i] Predicted Fourier (Re) \n Re(${\hat{\mathcal{V}}_{pred}(u,v)}$)" %i, fontsize=10)
                    im = ax.imshow(pred_vis_re, vmin=vmin_vis_re, vmax=vmax_vis_re)
                    ax.text(0.03, 0.97, f"MSE={mse(pred_vis_re, gt_vis_re):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    ax.text(0.03, 0.92, f"SSIM={ssim(pred_vis_re, gt_vis_re):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    # Predicted spectral (imaginary)
                    ax = plt.subplot(gs[4,i])
                    ax.set_title("[%i] Predicted Fourier (Im) \n Im(${\hat{\mathcal{V}}_{pred}(u,v)}$)" %i, fontsize=10)
                    im = ax.imshow(pred_vis_im, vmin=vmin_vis_im, vmax=vmax_vis_im)
                    ax.text(0.03, 0.97, f"MSE={mse(pred_vis_im, gt_vis_im):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    ax.text(0.03, 0.92, f"SSIM={ssim(pred_vis_im, gt_vis_im):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                    # Reconstructed image from prediction
                    ax = plt.subplot(gs[5,i])
                    ax.set_title("[%i] Predicted Image \n$ \mathscr{F}^{-1}_{NU}[\hat{\mathcal{V}}_{pred}(u,v)]$" %i, fontsize=10)
                    im = ax.imshow(pred_img, vmin=vmin_img, vmax=vmax_img, cmap='afmhot')
                    ax.text(0.03, 0.97, f"MSE={mse(pred_img, gt_img):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    ax.text(0.03, 0.92, f"SSIM={ssim(pred_img, gt_img):1.3e}", color='w', fontsize=8, ha='left', va='top', transform=ax.transAxes)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig.colorbar(im, cax=cax, orientation='vertical')
                plt.savefig(f'{self.folder_val}/val{self.numPlot:04d}_pltrecon-batch_{batch_idx:04d}.png', bbox_inches='tight')
                plt.clf()

                save_image(img_recon.view((-1, 1, img_res, img_res)), f'{self.folder_val}/val%04d_recon-batch_%04d.png' % (self.numPlot, batch_idx), nrow=4)
                save_image(img.view((-1, 1, img_res, img_res)), f'{self.folder_val}/val%04d_gt-batch_%04d.png' % (self.numPlot, batch_idx), nrow=4) 
                self.numPlot += 1  
            

            elif self.loss_type in ('image', 'image_spectral'):
                img_recon_gt= img.cpu().numpy()
                img_recon = self._recon_image_rfft(uv_dense, z, imsize=[img_res, img_res], max_base=max_base, eht_fov=eht_fov).real
            
                save_image(img_recon.view((-1, 1, img_res, img_res)), f'{self.folder_val}/val%04d_recon-batch_%04d.png' % (self.numPlot, batch_idx), nrow=4)
                save_image(img.view((-1, 1, img_res, img_res)), f'{self.folder_val}/val%04d_gt-batch_%04d.png' % (self.numPlot, batch_idx), nrow=4) 
                self.numPlot += 1  

        if self.use_unet:
            return self._step_unet(batch, batch_idx)
        elif self.loss_type=='spectral':
            return self._step(batch, batch_idx)
        elif self.loss_type in ('image', 'image_spectral'):
            return self._step_image_loss(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        pass
        # avg_loss_real = torch.stack([x for x,y,z,w,s in outputs]).mean()
        # avg_loss_imag = torch.stack([y for x,y,z,w,s in outputs]).mean()
        # avg_loss = torch.stack([z for x,y,z,w,s in outputs]).mean()
        # avg_energy = torch.stack([w for x,y,z,w,s in outputs]).mean()
        # avg_kl = torch.stack([s for x,y,z,w,s in outputs]).mean()
       
        # self.log('validation/total_loss', avg_loss)
        # self.log('validation/real_loss', avg_loss_real)
        # self.log('validation/imaginary_loss', avg_loss_imag)
        # self.log('validation_metadata/energy', avg_energy)      

    def from_pretrained(self, checkpoint_name):
        return self.load_from_checkpoint(checkpoint_name, strict=False)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--L_embed', type=int, default=128)
        parser.add_argument('--input_encoding', type=str, choices=['fourier','nerf','none'], default='fourier')
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--sigma', type=float, default=5.0) #sigma=1 seems to underfit and 4 overfits/memorizes
        parser.add_argument('--model_checkpoint', type=str, default='') #, default='./vae_flow_e2e_kl0.1_epoch139.ckpt')
        parser.add_argument('--val_fldr', type=str, default=f'./val_fldr-test')

        return parser


def parse_yaml(args,):
    '''
    Parse the yaml file, the settings in the yaml file are given higher priority
    args:
    argparse.Namespace 
    '''
    import yaml

    opt=vars(args)
    opt_raw= vars(args).copy()
    args_yaml= yaml.load(open(args.yaml_file), Loader=yaml.FullLoader)
    opt.update(args_yaml,)

    opt['eval'] =opt_raw['eval']
    opt['exp_name'] =opt_raw['exp_name']
    opt['dataset']= opt_raw['dataset']
    opt['model_checkpoint'] =opt_raw['model_checkpoint']
    opt['dataset_path']= opt_raw['dataset_path']
    opt['data_path_imgs']= opt_raw['data_path_imgs']
    opt['data_path_cont']= opt_raw['data_path_cont']
    opt['loss_type']= opt_raw['loss_type']
    opt['num_fourier']= opt_raw['num_fourier']
    opt['input_size']= opt_raw['input_size']

    args= argparse.Namespace(**opt)
    return args


def cli_main():
    pl.seed_everything(42)
    
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test') #default='Galaxy10_DECals_cont_mlp8')

    parser.add_argument('--eval', action='store_true',
                        default=False, help='if evaluation mode [False]')

    parser.add_argument('--batch_size',  default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str,
                        # default='Galaxy10',
                        default='Galaxy10_DECals',
                        help='MNIST | Galaxy10 | Galaxy10_DECals')

    parser.add_argument('--dataset_path', type=str, 
            #default=f'/astroim//data/eht_grid_256FC_200im_MNIST_full.h5', 
            #default=f'/astroim//data/eht_grid_256FC_200im_Galaxy10_full.h5', 
            #default=f'/astroim/data/eht_grid_256FC_200im_Galaxy10_DECals_full.h5', 
            # default=f'/astroim/data/eht_grid_256FC_200im_Galaxy10_DECals_test100.h5', 
            default=f'../data/eht_grid_256FC_200im_Galaxy10_DECals_full.h5', 
            # default=f'../data/eht_grid_128FC_200im_Galaxy10_full.h5', 
            help='dataset path to precomputed spectral data (dense grid and sparse grid)')

    parser.add_argument('--data_path_cont', type=str, 
            #default=f'/astroim/data/eht_cont_200im_MNIST_full.h5', 
            # default=f'../data/eht_cont_200im_Galaxy10_full.h5', 
            # default=f'../data/eht_cont_200im_Galaxy10_DECals_full.h5', 
            # default=f'/astroim/data/eht_cont_200im_Galaxy10_DECals_full.h5', 
            default=None,
            help='dataset path to precomputed spectral data (continuous)')

    parser.add_argument('--data_path_imgs', type=str, 
            # default=None, 
            # default='../data/Galaxy10.h5', 
            default='../data/Galaxy10_DECals.h5', 
            help='dataset path to Galaxy10 images; for MNIST, it is by default at ./MNIST; if None, sets to 0s (faster, imgs usually not needed)')
    
    parser.add_argument('--input_size',  default=64, type=int)
    parser.add_argument('--num_fourier',  default=256, type=int)
    parser.add_argument('--loss_type', type=str, default='spectral', help='spectral | image | spectral_image [spectral]')
    parser.add_argument('--scale_loss_image', type=float, default=1., help='only valid if use spectral_image as the loss_type' )
    parser.add_argument('--mlp_layers',  default=8, type=int, help=' # of layers in mlp, this will also decide the # of tokens [8]')
    parser.add_argument('--mlp_hidden_dim',  default=256, type=int, help=' hidden dims in mlp [256]')
    parser.add_argument('--m_epochs', default=1000, type=int, help= '# of max training epochs [1000]')

    parser.add_argument('--yaml_file', default='', type=str, help ='path to yaml file')
    
    parser = pl.Trainer.add_argparse_args(parser) # get lightning-specific commandline options
    #parser = VAE.add_model_specific_args(parser) # get model-defined commandline options
    parser = ImplicitAstro_Trans.add_model_specific_args(parser) # get model-defined commandline options 
    args = parser.parse_args()


    yaml_file= args.yaml_file
    if len(yaml_file)>0:
        parse_yaml(args)
    
    print(args)
    
    numVal = 32*16 #size of mnist is 60k
    #numVal = 32 # for test100
    latent_dim = 1024

    # ------------
    # data
    # ------------    
    # load up dataset of u, v vis and images
    
    dataset = EHTIM_Dataset(dset_name = args.dataset,
                            data_path = args.dataset_path,
                            data_path_cont = args.data_path_cont,
                            data_path_imgs = args.data_path_imgs,
                            img_res = args.input_size,
                            pre_normalize = False,
                            )

    split_train, split_val = random_split(dataset, [len(dataset)-numVal, numVal])
    ngpu = torch.cuda.device_count()
    #uv_pts_normalized, uv_pts_dense, vis, vis_dense, img = dataset[10]


    train_loader = DataLoader(
            split_train, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True, drop_last=True)

    val_loader = DataLoader(
            split_val, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            drop_last=True)     

    # ------------
    # model
    # ------------
    mlp_hiddens = [args.mlp_hidden_dim for i in range(args.mlp_layers-1)]
    implicitModel = ImplicitAstro_Trans(args,
                                        learning_rate=args.learning_rate,
                                        L_embed=args.L_embed,
                                        input_encoding=args.input_encoding,
                                        sigma=args.sigma,
                                        num_fourier_coeff=args.num_fourier,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        latent_dim=latent_dim,
                                        hidden_dims=mlp_hiddens,
                                        model_checkpoint=args.model_checkpoint,
                                        ngpu=ngpu)

    if len(args.model_checkpoint)>0:
        print(f'--- loading from {args.model_checkpoint}...')
        implicitModel = implicitModel.load_from_checkpoint(args.model_checkpoint)
        implicitModel.ngpu= ngpu
    
    #checkpoint_callback = ModelCheckpoint(monitor='train/total_loss')
    #trainer = Trainer(callbacks=[checkpoint_callback])
    trainer = pl.Trainer.from_argparse_args(args, 
                                            gpus=-1,
                                            plugins=DDPPlugin(find_unused_parameters=False),
                                            replace_sampler_ddp=True,
                                            accelerator='ddp',
                                            progress_bar_refresh_rate=20, 
                                            max_epochs=args.m_epochs, 
                                            val_check_interval=0.25, 
                                            )
                                            
    # ------------
    # training
    # ------------
    if not args.eval:
        print('==Training==')
        # wandb_logger = WandbLogger(name=args.experiment_name, project='implicit_astro', log_model='all')
        trainer.fit(implicitModel, train_loader, val_loader)  
        print(implicitModel)
    else:
        print('==Testing==')
        # implicitModel.load_from_checkpoint(args.ckpt_path)
        trainer.test(implicitModel, val_loader, )   
        print(implicitModel)


if __name__ == '__main__':
    cli_main()

