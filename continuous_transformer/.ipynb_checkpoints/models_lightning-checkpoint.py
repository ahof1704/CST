import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import continuous_transformer.ContSpaceTime as ContSpaceTime
from continuous_transformer.continuous_utils import patch_sampling, plot_predictions, plot_training_curves, plot_whole_frame, plot_predictions_in_between, plot_whole_sequence, plot_predictions_in_between_pca, plot_predictions_in_between_umap, plot_predictions_curves, plot_predictions_InBetweenPoints_curves, plot_attention_weights, vn_eig_entropy
from continuous_transformer.spectral_normalization import SpectralNorm
# import torch.nn.utils.spectral_norm as SpectralNorm
from continuous_transformer.utils import SaveBestModel_CST

# from torchcubicspline import(natural_cubic_spline_coeffs, 
#                              NaturalCubicSpline)

from continuous_transformer.sobolev_loss import sobolev_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

class BERT_Model(pl.LightningModule):
    def __init__(self,
                 cfg, 
                 patch_sampling_cfg,
                 train_cfg,my_loss=None,
                 path_to_save_models=None,
                 warmup=100,
                 n_labels=1, **kwargs):
        super().__init__()
        
        # -- sanity check: sampling parameter exists
        sampling_type = patch_sampling_cfg["structure"]
        self.in_between_frame_init = patch_sampling_cfg["in_between_frame_init"]

        self.sobolev_loss_ = my_loss
        
        self.save_best_model = SaveBestModel_CST()
        self.path_to_save_models = path_to_save_models
        self.val_loss = []
        self.warmup=warmup
        self.epoch_counter=0

        if sampling_type not in ["grid", "random"]:
            print(f"sampling type '{sampling_type}' not recognized, using random instead")
            patch_sampling_cfg["structure"] = "random"

        # -- sanity check: model type exists
        model_type = cfg["model_type"]
        print('model_type: ',model_type)
        if model_type not in ["linear_encoder", "conv_encoder"]:
            print(f"model type '{sampling_type}' not recognized, using 'linea_encoder' instead")
            cfg["model_type"] = model_type = "linear_encoder"

        if model_type == "linear_encoder":
            self.model = BERT_LinearEncoder(cfg, n_labels)
#             print('cfg: ',cfg)
#             print('patch_sampling_cfg: ',patch_sampling_cfg)
        elif model_type == "conv_encoder":
            self.model = BERT_ConvEncoder(cfg, n_labels)

#         self.transformer = ContSpaceTime.Transformer(cfg)
# #         self.fc = nn.Linear(cfg["dim"], cfg["dim"])
#         self.activ1 = nn.Tanh()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#         self.leakyrelu = nn.LeakyReLU(0.1)
#         self.drop = nn.Dropout(cfg["p_drop_hidden"])
#         self.activ2 = ContSpaceTime.gelu
#         self.norm = ContSpaceTime.LayerNorm(cfg)
            
        # -- set plotting variable
        if cfg["plot_predictions"] == False:
            cfg["plotting"] = False
            
        self.Lipschitz_regularization = False #cfg["Lipschitz_regularization"]
        # -- save hyperparameters so they are accessible everywhere
        # -- access using self.hparams
        self.save_hyperparameters()

    def forward(self, latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos):
        verbose=False
            
        if self.Lipschitz_regularization:
            h, scores, embedded_patches, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2 = self.model.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
            # h, scores, embedded_patches, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2 = self.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
        else:
            h, scores, embedded_patches = self.model.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
            # h, scores, embedded_patches = self.transformer(latentVar, T, P_row, P_col, map_fn, input_mask.float())
        embedded_patches_lastLayer = h.clone().detach()
        
        if verbose:
            print('[in forward] masked_pos: ',masked_pos)
            print('[in forward] masked_pos.shape: ',masked_pos.shape)
            print('[in forward] input_mask.shape: ',input_mask.shape)
            print('[in forward] T.shape: ',T.shape)
            print('[in forward] h.shape: ',h.shape)
            
        # if self.in_between_frame_init == 'interpolation': # In this case, the dummy frames should also be predicted
        #     masked_pos_with_dummy = torch.arange(len(T)).repeat(masked_pos.shape[0],1)
        #     masked_pos_with_dummy = masked_pos_with_dummy[:, :, None].expand(-1, -1, h.size(-1))
        #     masked_pos_with_dummy = masked_pos_with_dummy.type_as(latentVar)
        #     masked_pos_with_dummy = torch.as_tensor(masked_pos_with_dummy, dtype=torch.int64)
        #     h_masked = torch.gather(h, 1, masked_pos_with_dummy)
        #     if verbose: 
        #         print('[in forward] masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)
        #         print('[in forward] h_masked [with dummy].shape: ',h_masked.shape)
        #     # h_masked = self.model.norm(self.model.activ2(self.model.linear(h_masked)))
        #     h_masked = self.model.norm(self.activ2(self.model.linear(h_masked)))
            
        # else: #Normal
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        masked_pos = masked_pos.type_as(latentVar)
        masked_pos = torch.as_tensor(masked_pos, dtype=torch.int64)
        h_masked = torch.gather(h, 1, masked_pos) # Selects the tokens specified by 'masked_pos' from 'h'
        if verbose: 
            print('[in forward] masked_pos.shape: ',masked_pos.shape)
            print('[in forward] h_masked [normal].shape: ',h_masked.shape)
        h_masked = self.model.norm(self.model.activ2(self.model.linear(h_masked)))
        # h_masked = self.norm(self.activ2(self.linear(h_masked)))
            
            
        
        
        if self.Lipschitz_regularization:
            param = self.model.decoder.weight
            sym = torch.mm(param, torch.t(param))
            sym -= torch.eye(param.shape[0]).to(device)
            ls_ort_dec = sym.pow(2.0).sum()  # Loss for orthogonality
            
            # logits_lm = self.model.decoder(h_masked)
            logits_lm = self.decoder(h_masked)
            if verbose: print('latentVar [models_lighting after Lipschitz].shape: ',latentVar.shape)
        else: 
            
            # Do the interpolation still in the latent space (ie, before decoding back to the original space)
            # Using all points
            # u, c = np.unique(T, return_counts=True)
            # dup = u[c > 1]
            # if dup.size != 0:
            #     print('[in forward] There are duplicated time coordinates: ',dup)
            #     print(aaa)
            # before_logits_lm = h_masked.detach().clone() # (batches, num_frames, pixelxpixel)

            # coeffs = natural_cubic_spline_coeffs(T.to(device),h_masked.to(device))
            # spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation on the dummy frames
            # h_masked = spline.evaluate(T[masked_pos[0,:]].to(device))

            # logits_lm_with_dummy = spline.evaluate(T.to(device)).squeeze()
            # T_clone = T.detach().clone().cpu()
            # logits_lm_with_dummy_clone = logits_lm_with_dummy.detach().clone().cpu()
            # logits_lm_clone = logits_lm.detach().clone().cpu()
            

            logits_lm = self.model.decoder(h_masked) #+ self.decoder_bias  # ORIGINAL
            
            # If I do the interpolation on the latent space, decode everything and then get the coordinates that match the real points
            # logits_lm_with_dummy = self.model.decoder(h_masked)
            # if verbose: 
            #     print('[after interpolation in latent] h_masked.shape: ',h_masked.shape)
            #     print('[after interpolation in latent] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
            #     print('[after interpolation in latent] masked_pos[0,:].shape: ', masked_pos[0,:].shape)
            #     print('[after interpolation in latent] masked_pos[0,:]: ', masked_pos[0,:])
            # logits_lm = logits_lm_with_dummy[:, masked_pos[0,:]]
            # if verbose or torch.isnan(logits_lm).any(): 
            #     print('[after decoder] logits_lm: ',logits_lm)
            #     # print(aaa)
            
            # if self.in_between_frame_init == 'interpolation': # In this case, the dummy frames should also be predicted
            #     #Now do interpolation just with the dummy frames and evaluate on the coordinates of the real frames
            #     dummy_idxs = np.argwhere(np.isin(np.arange(len(T)),masked_pos[0,:].cpu(),invert=True)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and masked_pos should be complementary
            #     # real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(masked_pos.shape[1]),invert=False)).flatten() #returns the indexes of the real frames
            #     # print('T[masked_pos[0,:]: ',T[masked_pos[0,:]])
            #     if verbose: 
            #         print('dummy_idxs: ',dummy_idxs)
            #         print('T: ',T)
            #         print('T[masked_pos[0,:]]: ',T[masked_pos[0,:]])
            #         print('T[dummy_idxs]: ',T[dummy_idxs])
            #         print('logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
                # logits_lm_with_dummy_flat = torch.flatten(logits_lm_with_dummy, start_dim=2)
                # time_interpolation = T[0,logits_lm_with_dummy]
                
                # # Using just the dummy frames 
                # before_logits_lm = logits_lm.detach().clone() # (batches, num_frames, pixelxpixel)
                # coeffs = natural_cubic_spline_coeffs(T[dummy_idxs].to(device),logits_lm[:,dummy_idxs].to(device))
                # spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation on the dummy frames
                # logits_lm = spline.evaluate(T[masked_pos[0,:]].to(device))
                
#                 # Using some real frames 
#                 before_logits_lm = logits_lm.detach().clone() # (batches, num_frames, pixelxpixel)
#                 random_pos = np.random.permutation(masked_pos[0,1:-1])[:5]
#                 all_idxs = np.sort(np.concatenate((dummy_idxs,random_pos))) #Adding the first and the last pos manually
#                 all_idxs = np.sort(np.append(np.array([masked_pos[0,0],masked_pos[0,-1]]),all_idxs))
                
#                 if verbose: 
#                     print('random_pos: ',random_pos)
#                     print('all_idxs.shape: ',all_idxs.shape)
#                     print('all_idxs: ',all_idxs)
#                     print('T[all_idxs].shape: ',T[all_idxs].shape)
#                     print('T[all_idxs]: ',T[all_idxs])
#                     print('logits_lm[:,all_idxs].shape: ',logits_lm[:,all_idxs].shape)
#                     print('logits_lm.shape: ',logits_lm.shape)
#                     print('T.shape: ',T.shape)
#                 u, c = np.unique(all_idxs, return_counts=True)
#                 dup = u[c > 1]
#                 if dup.size != 0:
#                     print('There are duplicated time coordinates: ',dup)
#                     print('T[all_idxs]: ',T[all_idxs])
#                     print(aaa)
#                 coeffs = natural_cubic_spline_coeffs(T[all_idxs].to(device),logits_lm[:,all_idxs].to(device))
#                 spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation on the dummy frames
#                 logits_lm = spline.evaluate(T[masked_pos[0,:]].to(device)).squeeze()
#                 logits_lm_with_dummy = spline.evaluate(T.to(device)).squeeze()
#                 T_clone = T.detach().clone().cpu()
#                 logits_lm_with_dummy_clone = logits_lm_with_dummy.detach().clone().cpu()
#                 logits_lm_clone = logits_lm.detach().clone().cpu()
                
#                 if verbose:
#                     fig,ax = plt.subplots(2,2, figsize=(15,10))
#                     ax=ax.ravel()
#                     ax[0].scatter(T_clone,logits_lm_with_dummy_clone[0,:,0].cpu().squeeze(), label='Dummy')
#                     ax[0].plot(T_clone[all_idxs],before_logits_lm[0,all_idxs,0].cpu().squeeze(), c='g',label='interpolated')
#                     ax[0].scatter(T_clone[random_pos],before_logits_lm[0,random_pos,0].cpu().squeeze(), c='k',label='Point for fit',s=120)
#                     ax[0].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[0,:,0].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[0].legend()
                    
#                     ax[1].scatter(T_clone,logits_lm_with_dummy_clone[0,:,1].cpu().squeeze(), label='Dummy')
#                     ax[1].plot(T_clone[all_idxs],before_logits_lm[0,all_idxs,1].cpu().squeeze(),c='g', label='interpolated')
#                     ax[1].scatter(T_clone[random_pos],before_logits_lm[0,random_pos,1].cpu().squeeze(),c='k', label='Point for fit',s=120)
#                     ax[1].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[0,:,1].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[1].legend()
                    
#                     ax[2].scatter(T_clone,logits_lm_with_dummy_clone[1,:,0].cpu().squeeze(), label='Dummy')
#                     ax[2].plot(T_clone[all_idxs],before_logits_lm[1,all_idxs,0].cpu().squeeze(),c='g',label='interpolated')
#                     ax[2].scatter(T_clone[random_pos],before_logits_lm[1,random_pos,0].cpu().squeeze(),c='k',label='Point for fit',s=120)
#                     ax[2].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[1,:,0].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[2].legend()
                    
#                     ax[3].scatter(T_clone,logits_lm_with_dummy_clone[1,:,1].cpu().squeeze(), label='Dummy')
#                     ax[3].plot(T_clone[all_idxs],before_logits_lm[1,all_idxs,1].cpu().squeeze(),c='g',label='interpolated')
#                     ax[3].scatter(T_clone[random_pos],before_logits_lm[1,random_pos,1].cpu().squeeze(),c='k',label='Point for fit',s=120)
#                     ax[3].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[1,:,1].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[3].legend()
                    
#                     plt.show()
#                     plt.close('all')
                
                # # Using all frames frames 
                # u, c = np.unique(T, return_counts=True)
                # dup = u[c > 1]
                # if dup.size != 0:
                #     print('[in forward] There are duplicated time coordinates: ',dup)
                #     print(aaa)
                # before_logits_lm = logits_lm.detach().clone() # (batches, num_frames, pixelxpixel)
                # coeffs = natural_cubic_spline_coeffs(T.to(device),logits_lm.to(device))
                # spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation on the dummy frames
                # logits_lm = spline.evaluate(T[masked_pos[0,:]].to(device))
                # logits_lm_with_dummy = spline.evaluate(T.to(device)).squeeze()
                # T_clone = T.detach().clone().cpu()
                # logits_lm_with_dummy_clone = logits_lm_with_dummy.detach().clone().cpu()
                # logits_lm_clone = logits_lm.detach().clone().cpu()
                
#                 if verbose:
#                     fig,ax = plt.subplots(2,2, figsize=(15,10))
#                     fig.suptitle('Second interpolation', fontsize=16)
#                     ax=ax.ravel()
#                     ax[0].scatter(T_clone,logits_lm_with_dummy_clone[0,:,0].cpu().squeeze(), label='Dummy')
#                     # ax[0].plot(T_clone[all_idxs],before_logits_lm[0,all_idxs,0].cpu().squeeze(), c='g',label='interpolated')
#                     # ax[0].scatter(T_clone[random_pos],before_logits_lm[0,random_pos,0].cpu().squeeze(), c='k',label='Point for fit',s=120)
#                     ax[0].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[0,:,0].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[0].legend()
                    
#                     ax[1].scatter(T_clone,logits_lm_with_dummy_clone[0,:,1].cpu().squeeze(), label='Dummy')
#                     # ax[1].plot(T_clone[all_idxs],before_logits_lm[0,all_idxs,1].cpu().squeeze(),c='g', label='interpolated')
#                     # ax[1].scatter(T_clone[random_pos],before_logits_lm[0,random_pos,1].cpu().squeeze(),c='k', label='Point for fit',s=120)
#                     ax[1].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[0,:,1].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[1].legend()
                    
#                     ax[2].scatter(T_clone,logits_lm_with_dummy_clone[1,:,0].cpu().squeeze(), label='Dummy')
#                     # ax[2].plot(T_clone[all_idxs],before_logits_lm[1,all_idxs,0].cpu().squeeze(),c='g',label='interpolated')
#                     # ax[2].scatter(T_clone[random_pos],before_logits_lm[1,random_pos,0].cpu().squeeze(),c='k',label='Point for fit',s=120)
#                     ax[2].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[1,:,0].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[2].legend()
                    
#                     ax[3].scatter(T_clone,logits_lm_with_dummy_clone[1,:,1].cpu().squeeze(), label='Dummy')
#                     # ax[3].plot(T_clone[all_idxs],before_logits_lm[1,all_idxs,1].cpu().squeeze(),c='g',label='interpolated')
#                     # ax[3].scatter(T_clone[random_pos],before_logits_lm[1,random_pos,1].cpu().squeeze(),c='k',label='Point for fit',s=120)
#                     ax[3].scatter(T_clone[masked_pos[0,:]],logits_lm_clone[1,:,1].cpu().squeeze(),c='r',label='New logits_lm')
#                     ax[3].legend()
                    
#                     plt.show()
#                     plt.close('all')
                

            if verbose or torch.isnan(logits_lm).any(): 
                # print('dummy_idxs: ',dummy_idxs)
                if self.in_between_frame_init == 'interpolation': 
                    # print('masked_pos_with_dummy[0,:]: ',masked_pos_wsith_dummy[0,:])
                    # print('T[all_idxs].shape: ',T[all_idxs].shape)
                    print('[after spline] T: ',T)
                    print('[after spline] logits_lm.shape: ',logits_lm.shape)
                    # print('[after spline]  logits_lm[:,:,all_idxs].dtype: ', logits_lm[:,all_idxs].dtype)
                    # print('[after spline] logits_lm[:,all_idxs].shape: ',logits_lm[:,all_idxs].shape)
                    # print('[after spline] logits_lm[:,all_idxs]: ',logits_lm[:,all_idxs])
                    # print('[after spline] logits_lm[0,all_idxs]: ',logits_lm[0,all_idxs])
                    print('[after spline] logits_lm.max(): ',logits_lm.max())
                    print('[after spline] logits_lm.min(): ',logits_lm.min())
                else:
                    print('masked_pos[0,:]: ',masked_pos[0,:])
                # print('[after spline] before_logits_lm: ',before_logits_lm)
                # print('[after spline] before_logits_lm.shape: ',before_logits_lm.shape)
                # print('[after spline] before_logits_lm.max(): ',before_logits_lm.max())
                # print('[after spline] before_logits_lm.min(): ',before_logits_lm.min())
                # print('spline: ',spline)
                # print('coeffs: ',coeffs)
                # print('torch.isnan(before_logits_lm).any(): ',torch.isnan(before_logits_lm).any())
                # print('[in forward] T[dummy_idxs]: ',T[dummy_idxs])
                # print('[in forward] T[dummy_idxs].shape: ',T[dummy_idxs].shape)
                # print('[in forward] logits_lm.shape: ',logits_lm.shape)
                # print('[in forward] logits_lm: ',logits_lm)
                # print(aaaa)
        # logits_lm.retain_grad()
        logits_lm = logits_lm.reshape(logits_lm.shape[0], logits_lm.shape[1], self.hparams.cfg["patch_size"][0], self.hparams.cfg["patch_size"][1])
#         logits_lm = logits_lm.squeeze()
        if self.Lipschitz_regularization:
            return logits_lm,scores, embedded_patches, embedded_patches_lastLayer, ls_ort_dec, ls_ort_pos_embed_3DLin, ls_ort_proj_q, ls_ort_proj_k, ls_ort_proj_v, ls_ort_proj, ls_ort_fc1, ls_ort_fc2
        else:
            return logits_lm,scores, embedded_patches, embedded_patches_lastLayer#, logits_lm_with_dummy

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)

    def training_step(self, batch, batch_idx):
        # self.train()
        verbose=False
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        range_imshow = self.hparams.cfg["range_imshow"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        Lipschitz_regularization = self.hparams.cfg["Lipschitz_regularization"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        in_between_frame_init = self.hparams.patch_sampling_cfg["in_between_frame_init"]
        experiment_name = self.hparams.train_cfg["experiment_name"]
        # sobolev_loss = self.hparams.train_cfg["sobolev_loss"]# Use Sobolev training loss if true
        compute_loss_whole_curve = self.hparams.train_cfg["compute_loss_whole_curve"]
        compute_loss_on_dummy_points = self.hparams.train_cfg["compute_loss_on_dummy_points"]
        if compute_loss_on_dummy_points:
            weight_loss_on_real = self.hparams.train_cfg["weight_loss_on_real"]
        # weight_vne = self.hparams.train_cfg["weight_vne"]
        std_noise_t = self.hparams.train_cfg["std_noise_t"]
        std_to_data = self.hparams.train_cfg["std_to_data"]
        # perturbation_to_t = self.hparams.train_cfg["perturbation_to_t"]
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        
        # -- batch has data on [0] and targets on [1]
        # data = Variable(batch[0])
        data = batch[0]
        # print('data.requires_grad: ',data.requires_grad)
        data.requires_grad=True # I am adding this require grad to see if I can compute the derivatives manually
        T_orig = batch[1][0].cpu() #Set T=None if you don't want to use original time coordinates
        # T_orig = batch[1][0] #Set T=None if you don't want to use original time coordinates
        # T_orig.requires_grad=True

        if verbose: print('T_orig: ',T_orig)

                

        # if verbose: print('data [models_lighting].shape: ',data.shape)
#         data = torch.reshape(data, (data.shape[0], num_frames, frame_size, frame_size))
#         print('data [models_lighting].shape: ',data.shape)
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, T_orig, self.hparams.patch_sampling_cfg, frame_size, patch_size, self.current_epoch)
        if verbose: 
            print('data[0]: ',data[0])
            print('masked_tokens[0]: ',masked_tokens[0])

        segm_frames = segm_frames.to(device).requires_grad_(True)
        if compute_loss_on_dummy_points: #Make copy of the segm_frames to compute the loss later
            segm_frames_orig = segm_frames.clone()
        # T.requires_grad=True
        # P_row.requires_grad=True
        # P_col.requires_grad=True

        if std_to_data>0:
            if verbose: print('[before perturbation_to_data] segm_frames.shape: ',segm_frames.shape)
            segm_frames_before_perturn = segm_frames.clone()
            perturb_0 = torch.normal(mean=torch.zeros_like(segm_frames),std=std_to_data)
            segm_frames = segm_frames + perturb_0
            # perturb_1 = torch.normal(mean=torch.zeros_like(data),std=std_to_data)
            if verbose: 
                print('segm_frames_before_perturn[0]: ',segm_frames_before_perturn[0])
                print('perturb_0[0]: ',perturb_0[0])
                print('segm_frames[0]: ',segm_frames[0])
                # print('perturb_0.shape: ',perturb_0.shape)
        
        if std_noise_t>0:
            if verbose: print('[before perturbation_to_t] T: ',T)
            T_before_perturn = T.clone()
            perturb = torch.normal(mean=torch.zeros_like(T),std=std_noise_t)
            T = T+perturb
            if verbose: print('[after perturbation_to_t] T: ',T)

        if plotting == True and self.current_epoch%epochs_plot == 0:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            if verbose: print('masked_tokens: ',masked_tokens)
            if verbose: print('segm_frames: ',segm_frames)
            if "Navier" in experiment_name:
                plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')
            else:
                plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train')


        # print('[train] masked_pos: ',masked_pos)
        # print('[train] masked_tokens: ',masked_tokens)
        # T.requires_grad=True
#         masked_tokens = patch_sampling(data, self.hparams, frame_size, patch_size, self.current_epoch)
#         print('masked_tokens [models_lighting].shape: ',masked_tokens.shape)

        # if verbose: 
            # print('data.shape: ',data.shape)
            # print('T: ',T)
            # print('P_row: ',P_row)
            # print('P_col: ',P_col)
            # print('T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size,patch_size)

            
        latentVar = self.model.encoder(segm_frames.type_as(data))
            # if verbose: print('self.model.encoder [models_lighting without Lipschitz]: ',self.model.encoder.weight)
#         latentVar = latentVar.reshape(1, latentVar.shape[0], latentVar.shape[1])
        if verbose: print('[train] latentVar [models_lighting].shape: ',latentVar.shape)

#         input_mask = torch.zeros(1, latentVar.shape[1])
#         input_mask[:, :segm_frames.shape[0]] = 1
        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
        if verbose: print('input_mask.shape: ',input_mask.shape)
       
        mapping_size = 256
#         scale_gauss = 1
#         print('scale_gauss: ', scale_gauss)
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
            
#             if in_between_frame_init == 'interpolation': # In this case, the dummy frames should also be predicted
#                 masked_pos_with_dummy = torch.arange(len(T)).repeat(2,1)
#                 logits_lm_with_dummy, _ , _, _ = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
#                 if verbose:
#                     print('logits_lm_with_dummy[0,0]: ',logits_lm_with_dummy[0,0])
#                     print('logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
                
#                 #Now do interpolation just with the dummy frames and evaluate on the coordinates of the real frames
#                 dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and masked_pos should be complementary
#                 logits_lm_with_dummy_flat = torch.flatten(logits_lm_with_dummy, start_dim=2)
#                 # time_interpolation = T[0,logits_lm_with_dummy]
#                 coeffs = natural_cubic_spline_coeffs(T[dummy_idxs].to(device),logits_lm_with_dummy_flat[:,dummy_idxs].to(device))
#                 spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation on the dummy frames
                
#                 # # This is causing the grad_fn to be in function of the indexing, which we don't want. So just evaluate on the points we want
#                 # logits_lm = spline.evaluate(T.to(device))
#                 # # if verbose: 
#                 # #     print('[spline] logits_lm.shape: ',logits_lm.shape)
#                 # #     print('[spline] logits_lm: ',logits_lm)
#                 # logits_lm = logits_lm[:,masked_pos[0,:]].requires_grad_() #Manually set this to 
                
#                 # causing the grad_fn to be in function of the indexing, which we don't want. So just evaluate on the points we want
#                 logits_lm = spline.evaluate(T[masked_pos[0,:]].to(device))
#                 if verbose: 
#                     # print('[spline] logits_lm.shape: ',logits_lm.shape)
#                     # print('[spline] logits_lm: ',logits_lm)
#                     # print('T[masked_pos[0,:]]: ',T[masked_pos[0,:]])
#                 # logits_lm = logits_lm[:,masked_pos[0,:]].requires_grad_() #Manually set this to 
                
#                 if verbose: 
#                     print('dummy_idxs: ',dummy_idxs)
#                     print('masked_pos: ',masked_pos)
#                     # print('T[dummy_idxs]: ',T[dummy_idxs])
#                     # print('logits_lm_with_dummy_flat[:,dummy_idxs].shape: ',logits_lm_with_dummy_flat[:,dummy_idxs].shape)
#                     print('logits_lm[0,0]: ',logits_lm[0,0])
                
#             else:
            # latentVar.requires_grad=True
            # latentVar = Variable(latentVar.data, requires_grad=true) #Adding to see if I can compute the grad
            # print('latentVar: ',latentVar)
            # logits_lm , _ , _, _, logits_lm_with_dummy = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        # if compute_loss_whole_curve:
        #     # if num_in_between_frames > 0:
        #     #     in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
        #     #     if verbose: print('[sampling_full_frames] in_between_frames: ',in_between_frames)
                
        #     #     T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
        #     #     P_row = torch.zeros_like(T,dtype=torch.int64)
        #     #     P_col = torch.zeros_like(T,dtype=torch.int64)
        #     masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
        #     # print('masked_pos_with_dummy: ',masked_pos_with_dummy)
        #     # print('data: ',data)
        #     # print(aaaa)
        #     # logits_lm , _ , _, _ = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
        #     logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
        #     loss_lm = F.mse_loss(logits_lm, data) # for masked LM
        # else: 
        #     # logits_lm , _ , _, _ = self.forward(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        #     logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        #     loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
        # if verbose: 
            # print('dummy_idxs: ',dummy_idxs)
            # print('masked_pos: ',masked_pos)
            # print('T[dummy_idxs]: ',T[dummy_idxs])
            # print('logits_lm_with_dummy_flat[:,dummy_idxs].shape: ',logits_lm_with_dummy_flat[:,dummy_idxs].shape)
            # print('[train] logits_lm[0,0]: ',logits_lm[0,0])

        if verbose:
            print('[train] masked_tokens.shape: ', masked_tokens.shape)
            # print('[train] logits_lm.shape: ',logits_lm.shape)
            
        # logits_lm = logits_lm.reshape(logits_lm.shape[0],logits_lm.shape[1],patch_size[0],patch_size[1])
        # masked_tokens = masked_tokens.reshape(masked_tokens.shape[0],masked_tokens.shape[1],patch_size[0],patch_size[1])
        
        if self.sobolev_loss_ is not None:
            
            #logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
            # ids = np.tile(np.linspace(0,data.shape[1]-1,num=10, dtype=np.int64),(data.shape[1],1))
            masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
            logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            
            if std_noise_t>0:
                    ids = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
            else:
                ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
            ids2 = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
            if verbose: 
                print('[train] ids: ',ids)
                # print('[train] logits_lm.shape: ',logits_lm.shape)
                print('[train] data.squeeze(-1).shape: ',data.squeeze(-1).shape)
                # print('[train] logits_lm.squeeze(-1).shape: ',logits_lm.squeeze(-1).shape)
                print('T: ',T)
                print('[train] logits_lm_with_dummy.squeeze(-1).shape: ',logits_lm_with_dummy.squeeze(-1).shape)
                print('[train] logits_lm_with_dummy.squeeze(-1)[:,ids[0],:].shape: ',logits_lm_with_dummy.squeeze(-1)[:,ids,:].shape)
            
            loss_lm = self.sobolev_loss_.evaluate__loss(y=logits_lm_with_dummy.squeeze(-1),
                                                        # data.squeeze(-1)[:,ids[0],:],
                                                        data=data.squeeze(-1),
                                                        x=(segm_frames),
                                                        x_fd=T.to(device),
                                                        y_0=logits_lm_with_dummy.squeeze(-1)[:,ids,:],
                                                        indexes = ids2
                                                        )
        else: #if it is not sobolev
            if compute_loss_whole_curve: 
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                # if plot_att_weights:
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                # all_att = scores2[:,:,:]#.cpu().detach().numpy() # Select just the first curve
                # mean_att_per_patch_over_time = scores2.mean(axis=(0,1,2))
                all_att = scores2.reshape(-1,scores2.shape[-1])

                # vne = vn_eig_entropy(all_att)#.item()
                if std_noise_t>0:
                    ids = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
                else: 
                    ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                if verbose: 
                    print('ids: ', ids)
                    print('T[ids]: ',T[ids])
                    print('logits_lm_with_dummy[:,ids,:]: ', logits_lm_with_dummy[:,ids,:])
                # if std_to_data>0: 
                #     loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data_before_perturn) # for masked LM
                # else:
                loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data) # for masked LM
            elif compute_loss_on_dummy_points:
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                if verbose: print('masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)
                # if plot_att_weights:
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                # print('scores2.shape: ',scores2.shape)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                # print('all_att.shape: ',all_att.shape)
                # print(aaa)
                # all_att = scores2[:,:,:]#.cpu().detach().numpy() # Select just the first curve
                # mean_att_per_patch_over_time = scores2.mean(axis=(0,1,2))
                # vne = vn_eig_entropy(all_att)#.item()

                if verbose: 
                    print('T.shape: ',T.shape)
                    print('T: ',T)
                    print('logits_lm_with_dummy.shape: ', logits_lm_with_dummy.shape)
                    print('segm_frames.shape: ', segm_frames.shape)
                
                # loss_lm = F.mse_loss(logits_lm_with_dummy.squeeze(-1), segm_frames_orig) # for masked LM
                # ids_reals = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                # vne = vn_eig_entropy(all_att)#.item()
                if std_noise_t>0:
                    ids_reals = np.argwhere(np.isin(T_before_perturn,T_orig,invert=False)).flatten()
                    ids_dummies = np.argwhere(np.isin(T_before_perturn,T_orig,invert=True)).flatten()
                else: 
                    ids_reals = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                    ids_dummies = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
                if verbose: 
                    print('ids_reals: ', ids_reals)
                    print('T[ids_reals]: ',T[ids_reals])
                    if std_noise_t>0: print('T_before_perturn[ids_reals]: ',T_before_perturn[ids_reals])
                    print('logits_lm_with_dummy[:,ids_reals,:]: ', logits_lm_with_dummy[:,ids_reals,:])
                loss_lm_real = F.mse_loss(logits_lm_with_dummy[:,ids_reals,:].squeeze(-1), segm_frames_orig[:,ids_reals,:].to(device)) # for masked LM

                if verbose: 
                    print('ids_dummies: ', ids_dummies)
                    print('T[ids_dummies]: ',T[ids_dummies])
                    print('logits_lm_with_dummy[:,ids_dummies,:]: ', logits_lm_with_dummy[:,ids_dummies,:])
                loss_lm_dummies = F.mse_loss(logits_lm_with_dummy[:,ids_dummies,:].squeeze(-1), segm_frames_orig[:,ids_dummies,:].to(device)) # for masked LM

                loss_lm = loss_lm_real*weight_loss_on_real + loss_lm_dummies*(1-weight_loss_on_real)

            else:
                logits_lm , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
                scores2 = torch.stack(scores)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
        
        
        # loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
        # print('logits_lm.shape: ',logits_lm.shape)
        # if sobolev_loss:
            # print('Computing derivative')

            # # Option #1 computes, but the output comes with several zeros and doesn't make sense
            # logits_lm.sum().backward(retain_graph=True)#, create_graph=True)
            # dev_logits_lm = data.grad.detach().cpu().numpy()
            # data.requires_grad = False
            # logits_lm = logits_lm.detach().cpu().numpy()

            # #Option 2
            # print('logits_lm.shape: ',logits_lm.shape)
            # print('data.shape: ',data.shape)
            # dev_logits_lm = torch.autograd.grad(logits_lm, data)[0]

            # # Option 3
            # print('data.shape: ',data.shape)
            # data = torch.ones_like(data).to(device)
            # print('data.shape: ',data.shape)
            # dev_logits_lm=torch.autograd.functional.jacobian(logits_lm, data)

            # # Option #4
            # T.retain_grad()
            # logits_lm[:,:,0].sum().backward(retain_graph=True)#, create_graph=True)
            # print('logits_lm.sum().shape: ',logits_lm.sum().shape)
            # # logits_lm.backward(retain_graph=True)#, create_graph=True)
            # # print('logits_lm.sum().shape: ',logits_lm.sum().shape)
            # print('data.shape: ',data.shape)
            # print('T.shape:',T.shape)
            # print('T:',T)
            # T = T.reshape(1,data.shape[1])[:,:,None].repeat(1,1,2)
            # print('T.shape: ',T.shape)
            # # to get torch.Size([5, 20, 2, 1])
            # T = T.repeat(data.shape[0],1,1,1)
            # print('T.shape: ',T.shape)
            # print('T: ',T)
            # dev_logits_lm = T.grad.detach().cpu().numpy()
            # T.requires_grad = False
            # logits_lm = logits_lm.detach().cpu().numpy()

#             #Option 5
#             a = torch.cat([torch.ones(len(T),1),torch.zeros(len(T),1)],-1).to(device)
#             print('a.shape: ',a.shape)
#             b = torch.cat([torch.zeros(len(T),1),torch.ones(len(T),1)],-1)

#             # # This works for one curve in the batch 
#             # out1 = torch.autograd.grad(
#             #     (logits_lm[0,:].squeeze()), (T),grad_outputs = (a),
#             #     allow_unused=True, retain_graph=True)[0]

#             # Attempt for multi-curves
#             out1 = torch.autograd.grad(
#                 (logits_lm.squeeze()), (T),grad_outputs = (a),is_grads_batched=True,
#                 allow_unused=True, retain_graph=True)[0]
#             # out2 = torch.autograd.grad(
#             #     (logits_lm), (T),grad_outputs = (b),
#             #     allow_unused=True, retain_graph=True)

#             print('out1: ',out1)
#             # print('out2: ',out2)
#             print('out1.shape: ',out1.shape)
#             # print('dev_logits_lm.shape: ',dev_logits_lm.shape)
# #         print('logits_lm.shape: ',logits_lm.shape)
# #         print('masked_tokens.shape: ', masked_tokens.shape)

#         if logits_lm.dim()==1:
#             logits_lm = logits_lm[None,:]
#             masked_tokens = masked_tokens[None,:]

        
        
    
        loss = loss_lm.float() #- weight_vne*vne
        # loss = -vne
        # loss = loss_lm.float() + 1 / (weight_vne*vne)
        loss = loss.type_as(data)

        tmp10 = []
        for idx_tmp in range(data.shape[0]):
            # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
            # A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() #Original Sept30th 2022
            # A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
            # A=logits_lm[idx_tmp,:].flatten() #Original
            if compute_loss_whole_curve:
                A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
                B=data[idx_tmp,:].detach().cpu().numpy().flatten()
            elif compute_loss_on_dummy_points:
                A = logits_lm_with_dummy.squeeze(-1)[idx_tmp,:].detach().cpu().numpy().flatten()
                B=segm_frames_orig[idx_tmp,:].detach().cpu().numpy().flatten()
            else: 
                A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() #Original Sept30th 2022
                B=masked_tokens[idx_tmp,:].detach().cpu().numpy().flatten()
#             if len(np.unique(A)):
#                 A=A+np.random.rand(len(A))*10e-10
#             if len(np.unique(B)):
#                 B=B+np.random.rand(len(B))*10e-10
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        if(np.isnan(tmp10).any()):
            print('A: ',A)
            print('B: ',B)
            print(aaaa)
        mean_r2 = np.nanmean(tmp10)
        
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            "train_r2": mean_r2,
            #optional for logging purposes
            # "log": logs
          }

        self.log("train_loss", loss.detach())
        self.log("train_loss_lm", loss_lm.detach())
        # self.log("train_vne", vne.detach())
        self.log("train_r2", mean_r2)
        
        #Logs
        logs={"train_loss": loss.detach(), "train_r2": mean_r2}

        if verbose:
            print('[train] plotting: ',plotting)
            print('[train] epochs_plot: ',epochs_plot)
            print('[train] self.current_epoch: ',self.current_epoch)
        
        if plotting == True and self.current_epoch%epochs_plot == 0:
            # self.eval()
            if "curve" in experiment_name or "Spirals" in experiment_name: 
                # print('compute_loss_whole_curve: ')
                if not compute_loss_whole_curve:
                    # To make plots with the dummy points ##
                    masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                    logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                #     if verbose: print('logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
                #     print(aaa)
                # else:
                #     logits_lm_with_dummy=logits_lm.clone() 

                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train', replace_by_real_frame=False)
                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train', replace_by_real_frame=True)
                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train', replace_by_real_frame=False)
                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train', replace_by_real_frame=True)
                dev_logits_lm=None
                logits_lm=None
                # if std_to_data>0: 
                #     data_to_plot  = data_before_perturn.detach()
                # if std_noise_t>0:
                #     T_to_plot = T_before_perturn.detach()
                # else:
                T_to_plot = T.clone().detach()
                if verbose: 
                    print('data[0]: ',data[:2])
                    print('masked_tokens[0]: ',masked_tokens[:2])
                if "Navier" in experiment_name:
                    plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')
                else:
                    plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='train')

            elif "grab" in experiment_name:
                # plot_predictions(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, range_imshow, save_path)
                plot_predictions(data,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='train')


        del T, P_row, P_col, segm_frames, masked_pos, masked_tokens, loss, loss_lm, tmp10, mean_r2, data, A, B #, dev_logits_lm
        
        return batch_dictionary  #It was 'loss', now it is a dict

    def on_validation_epoch_start(self):
        # -- set ploting variables
        if self.hparams.cfg["plot_predictions"] == True:
            self.hparams.cfg["plotting"] = True
        if self.hparams.cfg["plot_training_curves"] == True:
            self.hparams.cfg["plot_training_curves"] = True

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        # -- only plot for one batch, then turn off plotting
        self.hparams.cfg["plotting"] = False
        self.hparams.cfg["plot_training_curves"] = False
        
    # def on_train_epoch_start(self):
    #     # -- set ploting variables
    #     if self.hparams.cfg["plot_predictions"] == True:
    #         self.hparams.cfg["plotting"] = True
    #     if self.hparams.cfg["plot_training_curves"] == True:
    #         self.hparams.cfg["plot_training_curves"] = True
    def on_train_epoch_start(self):
        if self.hparams.cfg["plot_predictions"] == True:
            self.hparams.cfg["plotting"] = True

    def on_train_batch_end(self, outputs, batch, batch_idx): #, dataloader_idx):
        # -- only plot for one batch, then turn off plotting
        # print('[on_train_batch_end] self.trainer.num_training_batches: ',self.trainer.num_training_batches)
        if batch_idx == 0: #self.trainer.num_training_batches-4: # 
            # print('[on_train_batch_end] setting plotting=True')
            self.hparams.cfg["plotting"] = True
        else: 
            # print('[on_train_batch_end] setting plotting=False')
            self.hparams.cfg["plotting"] = False
            self.hparams.cfg["plot_training_curves"] = False
        
        
    def validation_epoch_end(self, outputs):
        self.epoch_counter+=1
        if self.epoch_counter > self.warmup:
            Lipschitz_regularization = False #self.hparams.cfg["Lipschitz_regularization"]

            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

            self.val_loss.append(avg_loss.item())

            if self.epoch_counter == self.warmup:
                print('Warmup reached')
            if self.epoch_counter > self.warmup:
                self.save_best_model(self.path_to_save_models, self.val_loss[-1], self.epoch_counter, self.model)

            if Lipschitz_regularization:
                avg_lm_loss = torch.stack([x['val_loss_lm'] for x in outputs]).mean()
                avg_ls_ort_dec = torch.stack([x['val_ls_ort_dec'] for x in outputs]).mean()
                avg_ls_ort_enc = torch.stack([x['val_ls_ort_enc'] for x in outputs]).mean()
                avg_ls_ort_pos_embed_3DLin = torch.stack([x['val_ls_ort_pos_embed_3DLin'] for x in outputs]).mean()
                avg_ls_ort_proj_q = torch.stack([x['val_ls_ort_proj_q'] for x in outputs]).mean()
                avg_ls_ort_proj_k = torch.stack([x['val_ls_ort_proj_k'] for x in outputs]).mean()
                avg_ls_ort_proj_v = torch.stack([x['val_ls_ort_proj_v'] for x in outputs]).mean()
                avg_ls_ort_proj = torch.stack([x['val_ls_ort_proj'] for x in outputs]).mean()
                avg_ls_ort_fc1 = torch.stack([x['val_ls_ort_fc1'] for x in outputs]).mean()
                avg_ls_ort_fc2 = torch.stack([x['val_ls_ort_fc2'] for x in outputs]).mean()


    #         val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
            if Lipschitz_regularization:
                tensorboard_logs = {'val_loss': avg_loss, 
                                    'val_loss_lm': avg_lm_loss,
                                    'val_ls_ort_dec': avg_ls_ort_dec,
                                    'val_ls_ort_enc': avg_ls_ort_enc,
                                    "val_ls_ort_pos_embed_3DLin" : avg_ls_ort_pos_embed_3DLin.float(),
                                    "val_ls_ort_proj_q" : avg_ls_ort_proj_q.float(),
                                    "val_ls_ort_proj_k" : avg_ls_ort_proj_k.float(),
                                    "val_ls_ort_proj_v" : avg_ls_ort_proj_v.float(),
                                    "val_ls_ort_proj": avg_ls_ort_proj.float(),
                                    "val_ls_ort_fc1" : avg_ls_ort_fc1.float(),
                                    "val_ls_ort_fc2" : avg_ls_ort_fc2.float(),
                                    'step': self.current_epoch,
                                   }
            else: 
                tensorboard_logs = {'val_loss': avg_loss, 
                                    'step': self.current_epoch,
                                   }

            # logging using tensorboard logger
            self.logger.experiment.add_scalar("Loss/Val", avg_loss, self.current_epoch)
            # self.logger.experiment.add_scalar("Loss/VNE", avg_vne, self.current_epoch)
            if Lipschitz_regularization:
                self.logger.experiment.add_scalar("Loss/Val_LM", avg_lm_loss, self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_dec", avg_ls_ort_dec, self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_enc", avg_ls_ort_enc, self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_pos_embed_3DLin",avg_ls_ort_pos_embed_3DLin,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_proj_q",avg_ls_ort_proj_q,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_proj_k",avg_ls_ort_proj_k,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_proj_v",avg_ls_ort_proj_v,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_proj",avg_ls_ort_proj,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_fc1",avg_ls_ort_fc1,self.current_epoch)
                self.logger.experiment.add_scalar("Loss/Val_ort_fc2",avg_ls_ort_fc2,self.current_epoch)


            return {'log': tensorboard_logs}
        
        else:
            print('Still warmup')
            avg_loss = float('inf')
    
    def training_epoch_end(self, outputs):
        Lipschitz_regularization = False #self.hparams.cfg["Lipschitz_regularization"]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        if Lipschitz_regularization:
            avg_lm_loss = torch.stack([x['train_loss_lm'] for x in outputs]).mean()
            avg_ls_ort_dec = torch.stack([x['train_ls_ort_dec'] for x in outputs]).mean()
            avg_ls_ort_enc = torch.stack([x['train_ls_ort_enc'] for x in outputs]).mean()
            avg_ls_ort_pos_embed_3DLin = torch.stack([x['train_ls_ort_pos_embed_3DLin'] for x in outputs]).mean()
            avg_ls_ort_proj_q = torch.stack([x['train_ls_ort_proj_q'] for x in outputs]).mean()
            avg_ls_ort_proj_k = torch.stack([x['train_ls_ort_proj_k'] for x in outputs]).mean()
            avg_ls_ort_proj_v = torch.stack([x['train_ls_ort_proj_v'] for x in outputs]).mean()
            avg_ls_ort_proj = torch.stack([x['train_ls_ort_proj'] for x in outputs]).mean()
            avg_ls_ort_fc1 = torch.stack([x['train_ls_ort_fc1'] for x in outputs]).mean()
            avg_ls_ort_fc2 = torch.stack([x['train_ls_ort_fc2'] for x in outputs]).mean()

        if Lipschitz_regularization:
            tensorboard_logs = {'train_loss': avg_loss, 
                                'train_loss_lm': avg_lm_loss,
                                'train_ls_ort_dec': avg_ls_ort_dec,
                                'train_ls_ort_enc': avg_ls_ort_enc,
                                "train_ls_ort_pos_embed_3DLin" : avg_ls_ort_pos_embed_3DLin.float(),
                                "train_ls_ort_proj_q" : avg_ls_ort_proj_q.float(),
                                "train_ls_ort_proj_k" : avg_ls_ort_proj_k.float(),
                                "train_ls_ort_proj_v" : avg_ls_ort_proj_v.float(),
                                "train_ls_ort_proj": avg_ls_ort_proj.float(),
                                "train_ls_ort_fc1" : avg_ls_ort_fc1.float(),
                                "train_ls_ort_fc2" : avg_ls_ort_fc2.float(),
                                'step': self.current_epoch,
                               }
        else: 
            tensorboard_logs = {'train_loss': avg_loss, 
                                'step': self.current_epoch,
                               }
        # logging histograms
        self.custom_histogram_adder()
    
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",avg_loss,self.current_epoch)
        if Lipschitz_regularization:
            self.logger.experiment.add_scalar("Loss/Train_LM",avg_lm_loss,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_dec",avg_ls_ort_dec,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_enc",avg_ls_ort_enc,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_pos_embed_3DLin",avg_ls_ort_pos_embed_3DLin,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_q",avg_ls_ort_proj_q,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_k",avg_ls_ort_proj_k,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj_v",avg_ls_ort_proj_v,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_proj",avg_ls_ort_proj,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_fc1",avg_ls_ort_fc1,self.current_epoch)
            self.logger.experiment.add_scalar("Loss/Train_ort_fc2",avg_ls_ort_fc2,self.current_epoch)
        
    
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs}

        
    def validation_step(self, batch, batch_idx):

        # torch.set_grad_enabled(True)
        verbose=False
        if self.sobolev_loss_ is not None:
            torch.set_grad_enabled(True)
        # -- load variables from cfg dict
        n_pred = self.hparams.patch_sampling_cfg["num_patches_to_hide"]
        num_patches = self.hparams.patch_sampling_cfg["num_patches"]
        num_frames = self.hparams.patch_sampling_cfg["num_frames"]
        frame_size = self.hparams.cfg["frame_size"]
        patch_size = self.hparams.cfg["patch_size"]
        range_imshow = self.hparams.cfg["range_imshow"]
        model_type = self.hparams.cfg["model_type"]
        scale_gauss = self.hparams.cfg["scale_gauss"]
        Lipschitz_regularization = False #self.hparams.cfg["Lipschitz_regularization"]
        penalty_orthogonality = self.hparams.cfg["penalty_orthogonality"]
        in_between_frame_init = self.hparams.patch_sampling_cfg["in_between_frame_init"]
        experiment_name = self.hparams.train_cfg["experiment_name"]
        # sobolev_loss = self.hparams.train_cfg["sobolev_loss"]# Use Sobolev training loss if true
        compute_loss_whole_curve = self.hparams.train_cfg["compute_loss_whole_curve"]
        compute_loss_on_dummy_points = self.hparams.train_cfg["compute_loss_on_dummy_points"]
        if compute_loss_on_dummy_points:
            weight_loss_on_real = self.hparams.train_cfg["weight_loss_on_real"]
        plot_att_weights = self.hparams.cfg["plot_att_weights"]
        # weight_vne = self.hparams.train_cfg["weight_vne"]
        std_noise_t = self.hparams.train_cfg["std_noise_t"]
        # perturbation_to_t = self.hparams.train_cfg["perturbation_to_t"]
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        inference=False
        
        
        # -- batch has data on [0] and targets on [1]
        data = batch[0]
        # data = Variable(batch[0])
        # data.requires_grad=True # I am adding this require grad to see if I can compute the derivatives manually
        # data.requires_grad=True # I am adding this require grad to see if I can compute the derivatives manually
        T_orig = batch[1][0].cpu() #Just the first item 
        if verbose: 
            print('[validation_step] data.shape: ',data.shape)
            print('[validation_step] T_orig: ',T_orig)

#         data = torch.reshape(data, (data.shape[0], num_frames, frame_size["rows"], frame_size["cols"]))
#         if verbose: print('data [models_lighting].shape: ',data.shape)
        
        # -- sample patch in time and space
        # -- returns patch coordinates, segment with patches, and masked positions
        T, \
        P_row, \
        P_col, \
        segm_frames, \
        masked_pos, \
        masked_tokens = patch_sampling(data, T_orig, self.hparams.patch_sampling_cfg, frame_size, patch_size)
#         print('[models_light] T.shape: {}, P_row.shape: {}, P_col.shape: {}, segm_frames.shape: {}, masked_pos.shape: {}, masked_tokens.shape: {}'.format(T.shape, P_row.shape, P_col.shape, segm_frames.shape, masked_pos.shape, masked_tokens.shape))
        if self.sobolev_loss_ is not None:
            segm_frames = segm_frames.to(device).requires_grad_(True)
        if verbose: print('[val] masked_pos: ',masked_pos)
        if verbose: print('[val] masked_tokens: ',masked_tokens)
        if verbose: print('[val] masked_pos.shape: ',masked_pos.shape)
        if verbose: print('[val] T[masked_pos[0,:].long()]: ',T[masked_pos[0,:]])
        
        if model_type=='conv_encoder':
            segm_frames = segm_frames.reshape(segm_frames.shape[0],1,patch_size[0],patch_size[1])

        if verbose: print('[val] segm_frames.shape: ',segm_frames.shape)

        # segm_frames = segm_frames.to(device).requires_grad_(True)
        if plotting == True and self.current_epoch%epochs_plot == 0:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            if "Navier" in experiment_name:
                plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val_segFram')
            else:
                plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, None, logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val_segFram')
        if inference:
            dev_logits_lm=None
            logits_lm=None
            loss=None
            mean_r2=None
            plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, segm_frames, patch_size,self.current_epoch, loss, mean_r2, None, mode='val_segFram_inference')

        # if perturbation_to_t:
        #     if verbose: print('[before perturbation_to_t] T: ',T)
        #     T_before_perturn = T.clone()
        #     perturb = torch.normal(mean=torch.zeros_like(T),std=std_noise_t)
        #     T = T+perturb
        #     if verbose: print('[after perturbation_to_t] T: ',T)
        
            
        latentVar = self.model.encoder(segm_frames.type_as(data))
#         latentVar = latentVar.reshape(1, latentVar.shape[0], latentVar.shape[1])
            # if verbose: print('latentVar [models_lighting].shape: ',latentVar.shape)

        # latentVar = self.model.encoder(segm_frames.type_as(data))
        if verbose: print('[val] latentVar.shape: ',latentVar.shape)
#         latentVar = latentVar.reshape(1, latentVar.shape[0], latentVar.shape[1])
#         print('latentVar [models_lighting - validation step].shape: ',latentVar.shape)

#         # original 
#         input_mask = torch.zeros(1, latentVar.shape[1])
#         input_mask[:, :segm_frames.shape[0]] = 1

#         x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
#         mask : (B(batch_size) x S(seq_len))
        input_mask = torch.ones(latentVar.shape[0], latentVar.shape[1])
        # if verbose: print('input_mask.shape: ',input_mask.shape)
#         input_mask[:, :segm_frames.shape[0]] = 1
       
        mapping_size = 256
#         scale_gauss = 1
#         print('scale_gauss: ', scale_gauss)
        if scale_gauss is None: 
            map_fn = None 
        else: 
            map_fn = np.random.normal(0, scale=1, size=(mapping_size, 3)) * scale_gauss
            
        # logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
            
        # if verbose:
        #     print('[val] logits_lm.shape: ',logits_lm.shape)
        #     # print('[val] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        #     print('[val] masked_tokens.shape: ', masked_tokens.shape)

        if self.sobolev_loss_ is not None:
            
            #logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
            # ids = np.tile(np.linspace(0,data.shape[1]-1,num=10, dtype=np.int64),(data.shape[1],1))
            masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
            logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
            all_att = scores2.reshape(-1,scores2.shape[-1])
            # if plot_att_weights:
            #     logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            #     scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
            #     # all_att = scores2[:,:,:]#.cpu().detach().numpy() # Select just the first curve
            #     mean_att_per_patch_over_time = scores2.mean(axis=(0,1,2))
            #     vne = vn_eig_entropy(mean_att_per_patch_over_time).item()
            # else: 
            #     logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
        #     
            # logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
            
            ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
            ids2 = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
            if verbose: 
                print('[train] ids: ',ids)
                # print('[train] logits_lm.shape: ',logits_lm.shape)
                print('[train] data.squeeze(-1).shape: ',data.squeeze(-1).shape)
                # print('[train] logits_lm.squeeze(-1).shape: ',logits_lm.squeeze(-1).shape)
                print('T: ',T)
                print('[train] logits_lm_with_dummy.squeeze(-1).shape: ',logits_lm_with_dummy.squeeze(-1).shape)
                print('[train] logits_lm_with_dummy.squeeze(-1)[:,ids[0],:].shape: ',logits_lm_with_dummy.squeeze(-1)[:,ids,:].shape)
            
            loss_lm = self.sobolev_loss_.evaluate__loss(y=logits_lm_with_dummy.squeeze(-1),
                                                        # data.squeeze(-1)[:,ids[0],:],
                                                        data=data.squeeze(-1),
                                                        x=(segm_frames),
                                                        x_fd=T.to(device),
                                                        y_0=logits_lm_with_dummy.squeeze(-1)[:,ids,:],
                                                        indexes = ids2
                                                        )
            

        else: #if it is not sobolev
            if compute_loss_whole_curve: 
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                if verbose: print('masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)
                # if plot_att_weights:
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                # print('scores2.shape: ',scores2.shape)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                # print('all_att.shape: ',all_att.shape)
                # print(aaa)
                # all_att = scores2[:,:,:]#.cpu().detach().numpy() # Select just the first curve
                # mean_att_per_patch_over_time = scores2.mean(axis=(0,1,2))
                # vne = vn_eig_entropy(all_att)#.item()

                ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                if verbose: 
                    print('ids: ', ids)
                    print('T[ids]: ',T[ids])
                    print('logits_lm_with_dummy[:,ids,:]: ', logits_lm_with_dummy[:,ids,:])
                
                loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data) # for masked LM
            elif compute_loss_on_dummy_points:
                masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                if verbose: print('masked_pos_with_dummy.shape: ',masked_pos_with_dummy.shape)
                # if plot_att_weights:
                logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                scores2 = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
                # print('scores2.shape: ',scores2.shape)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                # print('all_att.shape: ',all_att.shape)
                # print(aaa)
                # all_att = scores2[:,:,:]#.cpu().detach().numpy() # Select just the first curve
                # mean_att_per_patch_over_time = scores2.mean(axis=(0,1,2))
                # vne = vn_eig_entropy(all_att)#.item()

                if verbose: 
                    print('T.shape: ',T.shape)
                    print('T: ',T)
                    print('logits_lm_with_dummy.shape: ', logits_lm_with_dummy.shape)
                    print('segm_frames.shape: ', segm_frames.shape)
                
                # loss_lm = F.mse_loss(logits_lm_with_dummy.squeeze(-1), segm_frames.to(device)) # for masked LM
                ids_reals = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                if verbose: 
                    print('ids_reals: ', ids_reals)
                    print('T[ids_reals]: ',T[ids_reals])
                    print('logits_lm_with_dummy[:,ids_reals,:]: ', logits_lm_with_dummy[:,ids_reals,:])
                loss_lm_real = F.mse_loss(logits_lm_with_dummy[:,ids_reals,:].squeeze(-1), segm_frames[:,ids_reals,:].to(device)) # for masked LM

                ids_dummies = np.argwhere(np.isin(T,T_orig,invert=True)).flatten()
                if verbose: 
                    print('ids_dummies: ', ids_dummies)
                    print('T[ids_dummies]: ',T[ids_dummies])
                    print('logits_lm_with_dummy[:,ids_dummies,:]: ', logits_lm_with_dummy[:,ids_dummies,:])
                loss_lm_dummies = F.mse_loss(logits_lm_with_dummy[:,ids_dummies,:].squeeze(-1), segm_frames[:,ids_dummies,:].to(device)) # for masked LM

                loss_lm = loss_lm_real*weight_loss_on_real + loss_lm_dummies*(1-weight_loss_on_real)

            else: 
                logits_lm , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
                scores2 = torch.stack(scores)
                all_att = scores2.reshape(-1,scores2.shape[-1])
                loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM
        # logits_lm = logits_lm.reshape(logits_lm.shape[0],logits_lm.shape[1],patch_size[0],patch_size[1])
        # logits_lm_with_dummy = logits_lm_with_dummy.reshape(logits_lm_with_dummy.shape[0],logits_lm_with_dummy.shape[1],patch_size[0],patch_size[1])
        # masked_tokens = masked_tokens.reshape(masked_tokens.shape[0],masked_tokens.shape[1],patch_size[0],patch_size[1])

        # if compute_loss_whole_curve:
        #     # print('T.shape: ',T.shape)
        #     # print('T_orig.shape: ',T_orig.shape)
        #     ids = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
        #     masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
        #     if plot_att_weights:
        #         logits_lm_with_dummy , scores , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
        #     else: 
        #         logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
        #     # print('logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        #     # print('logits_lm_with_dummy.squeeze(-1)[:,ids,:].shape: ',logits_lm_with_dummy.squeeze(-1)[:,ids,:].shape)
        #     # print('data.shape: ',data.shape)      
        #     loss_lm = F.mse_loss(logits_lm_with_dummy[:,ids,:], data) # for masked LM
        # else: 
        #     logits_lm , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos)
        #     loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM

        # if verbose:
        #     print('[val] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        #     # print('[val] logits_lm: ',logits_lm)
        #     print('[val] masked_tokens.shape: ', masked_tokens.shape)

        # loss_lm = F.mse_loss(logits_lm, masked_tokens) # for masked LM

        # if sobolev_loss:
        #     print('Computing derivative')
        #     logits_lm.sum().backward(retain_graph=True, create_graph=True)
        #     dev_logits_lm = data.grad#.detach().numpy()
        #     print('dev_logits_lm: ',dev_logits_lm)
        #     print('dev_logits_lm.shape: ',dev_logits_lm.shape)

#         if logits_lm.dim()==1:
#             logits_lm = logits_lm[None,:]
#             masked_tokens = masked_tokens[None,:]

        # tmp10 = []
        # for idx_tmp in range(logits_lm.shape[0]):
        #      # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
        #     A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten()
        #     if compute_loss_whole_curve:
        #         B=data[idx_tmp,:].detach().cpu().numpy().flatten()
        #     else: 
        #         B=masked_tokens[idx_tmp,:].detach().cpu().numpy().flatten()

        #     r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
        #     tmp10.append(r2_sq)
        # mean_r2 = np.mean(tmp10)

        # if Lipschitz_regularization:
        #     loss = loss_lm.float() + penalty_orthogonality*(ls_ort_dec.float() + ls_ort_enc.float() + ls_ort_pos_embed_3DLin.float() + ls_ort_proj_q.float() + 
        #                                                     ls_ort_proj_k.float() + ls_ort_proj_v.float() + ls_ort_proj.float() + ls_ort_fc1.float() + ls_ort_fc2.float())
        # else:
            # print('loss_lm.float(): ',loss_lm.float())
            # print('vne: ',vne)
        loss = loss_lm.float() #- weight_vne*vne
        # loss = -vne
        # loss = loss_lm.float() + 1 / (weight_vne*vne)
        loss = loss.type_as(data)

        tmp10 = []
        for idx_tmp in range(data.shape[0]):
            # Check if they are all the same value. If yes, add a small jitter just to be able to compute R2
            # A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() #Original Sept30th 2022
            # A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
            # A=logits_lm[idx_tmp,:].flatten() #Original
            if compute_loss_whole_curve:
                A = logits_lm_with_dummy.squeeze(-1)[:,ids,:][idx_tmp,:].detach().cpu().numpy().flatten()
                B=data[idx_tmp,:].detach().cpu().numpy().flatten()
            elif compute_loss_on_dummy_points:
                A = logits_lm_with_dummy.squeeze(-1)[idx_tmp,:].detach().cpu().numpy().flatten()
                B=segm_frames[idx_tmp,:].detach().cpu().numpy().flatten()
            else: 
                A=logits_lm[idx_tmp,:].detach().cpu().numpy().flatten() #Original Sept30th 2022
                B=masked_tokens[idx_tmp,:].detach().cpu().numpy().flatten()
#             if len(np.unique(A)):
#                 A=A+np.random.rand(len(A))*10e-10
#             if len(np.unique(B)):
#                 B=B+np.random.rand(len(B))*10e-10
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2
        
            tmp10.append(r2_sq)
        if(np.isnan(tmp10).any()):
            print('A: ',A)
            print('B: ',B)
            print(aaaa)
        mean_r2 = np.nanmean(tmp10)

        self.log("val_loss", loss)
        self.log("val_loss_lm", loss_lm.detach())
        self.log("val_r2", mean_r2)
        # self.log("val_vne", vne)
        
        #Logs
        logs={"val_loss": loss, "val_r2": mean_r2}
        
        if Lipschitz_regularization:
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "val_loss": loss,
                "val_ls_ort_dec" : ls_ort_dec.float(),
                "val_ls_ort_enc" : ls_ort_enc.float(),
                "val_ls_ort_pos_embed_3DLin" : ls_ort_pos_embed_3DLin.float(),
                "val_ls_ort_proj_q" : ls_ort_proj_q.float(),
                "val_ls_ort_proj_k" : ls_ort_proj_k.float(),
                "val_ls_ort_proj_v" : ls_ort_proj_v.float(),
                "val_ls_ort_proj": ls_ort_proj.float(),
                "val_ls_ort_fc1" : ls_ort_fc1.float(),
                "val_ls_ort_fc2" : ls_ort_fc2.float(),
                "val_loss_lm" : loss_lm.float(),
                "val_r2": mean_r2,
                #optional for logging purposes
                "log": logs
              }
        else: 
            batch_dictionary={
                #REQUIRED: It ie required for us to return "loss"
                "val_loss": loss,
                "val_r2": mean_r2,
                # "vne": vne,
                #optional for logging purposes
                "log": logs
              }
        
        plotting = self.hparams.cfg["plotting"]
        epochs_plot = self.hparams.cfg["plot_every_epochs"]
        if verbose:
            print('[val] plotting: ',plotting)
            print('[val] epochs_plot: ',epochs_plot)            
        
        if plotting == True and self.current_epoch%epochs_plot == 0:
            if "curve" in experiment_name or "Spirals" in experiment_name: 
                # print('compute_loss_whole_curve: ', compute_loss_whole_curve)
                if not compute_loss_whole_curve:
                #     # To make plots with the dummy points ##
                #     # T_tmp = torch.linspace(0,1,100)
                #     # P_row_tmp = torch.zeros(100)
                #     # P_col_tmp = torch.zeros(100)
                    masked_pos_with_dummy = torch.arange(len(T)).repeat(data.shape[0],1)
                    logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                #     logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                #     if verbose: print('logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)

                # else:
                #     logits_lm_with_dummy=logits_lm.clone() 
    #             logits_lm_with_dummy , _ , _, _ = self(latentVar, T, P_row, P_col, map_fn, input_mask, masked_pos_with_dummy)
                # if verbose: print('logits_lm.shape: ',logits_lm.shape)
                # logits_lm_with_dummy=logits_lm.clone()
                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val', replace_by_real_frame=False)
                # plot_predictions_InBetweenPoints_curves(data,T, P_row, P_col, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val', replace_by_real_frame=True)
                # plot_predictions_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir,mode='val')
                dev_logits_lm=None
                logits_lm=None
                if "Navier" in experiment_name:
                    plot_predictions(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val')
                else:
                    plot_predictions_curves(data,T, T_orig, P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val')

                if plot_att_weights:
                    plot_attention_weights(scores,self.current_epoch, self.logger.log_dir, mode='val')
                # plot_predictions_curves(segm_frames,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, mode='val_seg_frames')

    #             plot_predictions_in_between(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=True)
    #             plot_predictions_in_between(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=False)

    #             plot_predictions_in_between_pca(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=True)
    #             plot_predictions_in_between_pca(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=False)

    #             # del logits_lm_with_dummy, masked_pos_with_dummy
    #             ## \To make plots with the dummy points ##
            elif "grab" in experiment_name:
                # plot_whole_sequence(data, self.current_epoch, range_imshow, self.logger.log_dir)
                plot_predictions(data,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir, mode='val')
    #             plot_whole_frame(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size, frame_size,
    #                              self.current_epoch, loss, mean_r2, range_imshow, self.logger.log_dir)
            
            # # To get prediction for the intermediary frames, I need to pass more info to masked_pos
            # # Start by selecting the indexes of the in-between frames
            # masked_pos_in_between=[]
            # for idx in range(len(T)):
            #     if not int(T[idx])==T[idx]: #This checkes if the time point is integer. If it is, then sample from the real data, ow just mask
            #         if verbose: print('Masking idx: {}, T[idx]: {}'.format(idx,T[idx]))
            #         masked_pos_in_between.append(idx)
            # masked_pos_in_between = torch.from_numpy(np.array(masked_pos_in_between, dtype=np.int64))#.reshape(batch_size_segments,n_pred)
            # masked_pos_in_between = masked_pos_in_between.repeat(2,1) #Hardcoded for two batches at this point
            
        del T, P_row, P_col, segm_frames, masked_pos, masked_tokens, loss, latentVar
            # plot_predictions_in_between_umap(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=False)
            # plot_predictions_in_between_umap(data,T, P_row, P_col, masked_pos_with_dummy, masked_pos, logits_lm_with_dummy, patch_size,self.current_epoch, loss, mean_r2, self.logger.log_dir, replace_by_real_frame=True)
            
            
            
        if self.hparams.cfg["plot_training_curves"] == True: 
            plot_training_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size, 
                     self.current_epoch, loss, mean_r2, self.logger.log_dir)
            
        return batch_dictionary  #It was 'loss', now it is a dict
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.train_cfg["learning_rate"],
                         weight_decay=self.hparams.train_cfg["weight_decay"])
        return optimizer
    
    
    
'''
Experiment with BERT model from this github:
https://github.com/dhlee347/pytorchic-bert
'''
class CNN_arc(nn.Module):     
    def __init__(self, image_size):
        super(CNN_arc, self).__init__()
        self.conv0 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # [(image_size−K+2P)/S]+1 = [(image_size−4+2*1)/2]+1
        self.batch0 =  nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  #[([[(image_size−4+2*1)/2]+1]−4+2*1)/2]+1
        self.batch1 =  nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 
        self.batch2 = nn.BatchNorm2d(128)
#         in_features = int((((((((((((image_size-4+2*1)/2)+1)-4+2*1)/2)+1)-4+2*1)/2)+1)-4+2*1)/2)+1)
        in_features = int(((image_size/8)**2)*128)
#         print('image_size: ',image_size)
#         print('in_features: ',in_features)
        self.fc1 = nn.Linear(in_features=in_features, out_features=768)   #For the 16x16 patch
        self.batch3 = nn.BatchNorm1d(768)

        nn.init.kaiming_normal_(self.conv0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv0.bias, 0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.batch0.weight, 1)
        nn.init.constant_(self.batch0.bias, 0)
        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)
        nn.init.constant_(self.batch2.weight, 1)
        nn.init.constant_(self.batch2.bias, 0)
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.batch3.weight, 1)
        nn.init.constant_(self.batch3.bias, 0)
            
    def forward(self, images):
#         print('\nimages.shape: ',images.shape)
        x = F.leaky_relu(self.conv0(images)) # [64,64,1], K=4, P=1, S=2 -> W2 =(W−K+2P)/S+1 = (64-3+2*1)/2+1= 86
#         print('x.shape: ',x.shape)
        x = self.batch0(x)
        x = F.leaky_relu(self.conv1(x)) # [256,256,1], K=3, P=1, S=3 -> W2 =(W−K+2P)/S+1 = (256-3+2*1)/3+1= 86
#         print('x1.shape: ',x.shape)
        x = self.batch1(x)
        x = F.leaky_relu(self.conv2(x)) #[43,43,16] -> W2 = (43-3+2*1)/2+1 = 22
#         print('x2.shape: ',x.shape)
        x = self.batch2(x)
        x = x.view([images.size(0), -1])
#         print('x3.shape: ',x.shape)
        x = F.leaky_relu(self.fc1(x)) #size mismatch, m1: [128 x 200], m2: [3528 x 500] at 
        x = self.batch3(x)
        return x #, p1,self_att

class BERT_LinearEncoder(nn.Module):
    "Bert Model : Masked LM and next sentence classification"
    def __init__(self, cfg, n_labels=1):
        super(BERT_LinearEncoder, self).__init__()
        self.transformer = ContSpaceTime.Transformer(cfg)
#         self.fc = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ1 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.activ2 = ContSpaceTime.gelu
        self.norm = ContSpaceTime.LayerNorm(cfg)
        self.beta_Lipschitz=0.8
        self.gamma_Lipschitz=0.01
        self.Spectral_normalization =  False #cfg["Spectral_normalization"]
        self.Lipschitz_regularization = False #.cfg["Lipschitz_regularization"]
        # self.I = torch.eye(cfg["patch_size"]**2) #For the encoder
#         n_dim = cfg["dim"]
        # print('cfg["patch_size"]: ',cfg["patch_size"])
        # if cfg["patch_size"]=='frame_size':
        #     frame_size = cfg["frame_size"]
        #     print('frame_size: ',frame_size)
        #     # dim_images = cfg[frame_size['rows']] * frame_size['cols']
        # else: 
        # dim_images = cfg["patch_size"]**2 #In this case, the patches are square, so one dimension is enough to determine the final image dim_image
        dim_images = cfg["patch_size"][0]* cfg["patch_size"][1] #In this case, the patches are square, so one dimension is enough to determine the final image dim_image
    
        if self.Lipschitz_regularization:
            self.linear = nn.Linear(cfg["dim"], cfg["dim"])
            self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
            nn.init.orthogonal_(self.decoder.weight)
            nn.init.orthogonal_(self.encoder.weight)
            nn.init.orthogonal_(self.linear.weight)
            
#               #### Using SVD #####
# #             self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
# #             U_dec, S_dec, Vh_dec = torch.linalg.svd(self.decoder.weight) #Do this just to get the right dimensions for USV. 
            
# #             self.U_dec = nn.Parameter(U_dec)
# #             self.S_dec = nn.Parameter(S_dec)
# #             nn.init.normal_(self.S_dec)
# #             self.Vh_dec = nn.Parameter(Vh_dec)
# #             del self.decoder
# #             self.W_dec = self.U_dec @ torch.diag(self.S_dec) @ self.Vh_dec[:self.U_dec.shape[0],:]
            
# #             if cfg["operation_with_pos_encoding"] == "concatenate":
# #                 self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
# #             elif cfg["operation_with_pos_encoding"] == "sum":  
# #                 self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
                
# #             U_enc, S_enc, Vh_enc = torch.linalg.svd(self.encoder.weight) #Do this just to get the right dimensions for USV. 
# #             self.U_enc = nn.Parameter(U_enc)
# #             self.S_enc = nn.Parameter(S_enc)
# #             nn.init.normal_(self.S_enc)
# #             self.Vh_enc = nn.Parameter(Vh_enc)
# #             del self.encoder
# #             self.W_enc = self.U_enc @ torch.diag(self.S_enc) @ self.Vh_enc[:self.U_enc.shape[0],:]
# #             #### \Using SVD #####
            
        elif self.Spectral_normalization: #To use spectral normalization
            self.linear = SpectralNorm(nn.Linear(cfg["dim"], cfg["dim"]))
            self.decoder = SpectralNorm(nn.Linear(cfg["dim"], dim_images, bias=False))
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = SpectralNorm(nn.Linear(dim_images, int(cfg["dim"]/2), bias=False))
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = SpectralNorm(nn.Linear(dim_images, cfg["dim"], bias=False))
            
            # self.linear = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], cfg["dim"]))
            # self.decoder = nn.utils.parametrizations.spectral_norm(nn.Linear(cfg["dim"], dim_images, bias=False))
            # if cfg["operation_with_pos_encoding"] == "concatenate":
            #     self.encoder = nn.utils.parametrizations.spectral_norm(nn.Linear(dim_images, int(cfg["dim"]/2), bias=False))
            # elif cfg["operation_with_pos_encoding"] == "sum":  
            #     self.encoder = nn.utils.parametrizations.spectral_norm(nn.Linear(dim_images, cfg["dim"], bias=False))
                
#             nn.init.orthogonal_(self.decoder.weight)
#             nn.init.orthogonal_(self.encoder.weight)
#             nn.init.orthogonal_(self.linear.weight)
            
        else: #If not Lipschitz constrained, just use linear layers
            self.linear = nn.Linear(cfg["dim"], cfg["dim"])
            self.decoder = nn.Linear(cfg["dim"], dim_images, bias=False)
            # print("cfg[""operation_with_pos_encoding""]: ", cfg["operation_with_pos_encoding"])
            if cfg["operation_with_pos_encoding"] == "concatenate":
                self.encoder = nn.Linear(dim_images, int(cfg["dim"]/2), bias=False)
            elif cfg["operation_with_pos_encoding"] == "sum":  
                self.encoder = nn.Linear(dim_images, cfg["dim"], bias=False)
            nn.init.xavier_normal_(self.decoder.weight, gain=nn.init.calculate_gain('tanh'))


class BERT_ConvEncoder(nn.Module):
    "Bert Model : Masked LM and next sentence classification"
    def __init__(self, cfg, n_labels=1):
        super().__init__()
#         print(f"BERT_ConvEncoder IS NOT IMPLEMENTED")
        self.transformer = ContSpaceTime.Transformer(cfg)
        self.fc = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ1 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(cfg["p_drop_hidden"])
        self.linear = nn.Linear(cfg["dim"], cfg["dim"])
        self.activ2 = ContSpaceTime.gelu
        self.norm = ContSpaceTime.LayerNorm(cfg)
        n_dim = cfg["dim"]
        dim_images = cfg["patch_size"]**2
        self.decoder = nn.Linear(n_dim, dim_images, bias=False)
#         self.encoder = nn.Linear(dim_images, n_dim, bias=False)
        self.encoder = CNN_arc(cfg["patch_size"])
#         nn.init.kaiming_normal_(self.decoder.weight, mode='fan_out', nonlinearity='relu')