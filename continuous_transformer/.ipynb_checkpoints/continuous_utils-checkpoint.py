import os, random, torch, h5py
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
from skimage.transform import rescale, resize
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from scipy import interpolate
# from scipy import linalg as la
# import math as m

# import umap
import pickle

from continuous_transformer.utils import Train_val_split, Dynamics_Dataset3, Dynamics_Dataset4

# from torchcubicspline import(natural_cubic_spline_coeffs, 
#                              NaturalCubicSpline)
#from torchinterp1d import interp1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_nvidia_smi():
    bashCommand = "nvidia-smi"
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output.decode('ascii'))
    
def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# def vn_eig_entropy(rho):
#     verbose=False
#     #Symmetrize matrix
#     rho = rho*rho.T
#     # rho = rho/np.sum(np.diag(rho))
#     rho = rho/torch.diagonal(rho).sum()
    
#     if verbose: print('rho as density: ',rho)
#     # print('np.diag(rho).sum(): ',np.diag(rho).sum())
#     # print('check_symmetric(rho): ',check_symmetric(rho))
#     # EV = la.eigvals(rho)
#     # EV = torch.linalg.svdvals(torch.from_numpy(rho))**2 #torch.view_as_real
#     EV = torch.linalg.svdvals(rho)**2 #torch.view_as_real
#     EV = EV/EV.sum()
#     if verbose:print('EV: ',EV)

#     # Drop zero eigenvalues so that log2 is defined
#     my_list = [x for x in EV.tolist() if x]
#     if verbose:print('my_list: ',my_list)
#     # EV = np.array(my_list)
#     EV = torch.Tensor(my_list)
#     if verbose:print('EV: ',EV)

#     # log2_EV = np.matrix(np.log(EV))
#     if verbose:print('np.matrix(np.log(EV)).H: ',np.matrix(np.log(EV)).H)
#     log2_EV = torch.log(EV)
#     if verbose:print('torch.log(EV): ',torch.log(EV))
#     # print('log2_EV.shape: ',log2_EV.shape)
#     if verbose:print('log2_EV: ',log2_EV)
#     # EV = np.matrix(EV)
#     # print('EV.shape: ',EV.shape)
#     # S = -torch.dot(EV, log2_EV.H)
#     S = -torch.dot(EV, log2_EV)
#     S.requires_grad=True
#     if verbose:print('S: ',S)
#     return(S)

def vn_eig_entropy(rho):
    
    rho = rho + 1e-10 # Just to solve the issue with values being 0.
    # print('rho.shape: ',rho.shape)
    # print('rho*torch.log(rho).shape: ',rho*torch.log(rho).shape)
    S= torch.mean(-torch.sum(rho*torch.log(rho),dim=1))
    # print('S.requires_grad: ',S.requires_grad)
        
    return S

    
    
# def patch_sampling(data, T_orig, patch_sampling_cfg, frame_size, patch_size, current_epoch=None):
#     sampling_type = patch_sampling_cfg["structure"]
    
#     if frame_size['rows']==patch_size[0] and frame_size['cols']==patch_size[1]:
#         T, P_row, P_col = sampling_full_frames(data, T_orig, patch_sampling_cfg, frame_size, patch_size)
#     # elif sampling_type == "random":
#     #     T, P_row, P_col = sampling_random(data, patch_sampling_cfg, frame_size, patch_size)
#     # else: # sampling_type == "grid"
#     #     if patch_sampling_cfg["mode"]=="inference":
#     #         T, P_row, P_col = sampling_grid(data, patch_sampling_cfg, frame_size, patch_size, current_epoch)
#     #     else:
#     #         T, P_row, P_col = sampling_grid(data, patch_sampling_cfg, frame_size, patch_size)

#     return sampling_segments(data, T, T_orig, P_row, P_col, patch_sampling_cfg, frame_size, patch_size)

def patch_sampling(data, T_orig, patch_sampling_cfg, frame_size, patch_size, current_epoch=None, T_fixed=None):
    sampling_type = patch_sampling_cfg["structure"]
    
    # if frame_size['rows']==patch_size[0] and frame_size['cols']==patch_size[1]: # Sampling whole frame (just temporal attention)
    T, P_row, P_col = sampling_full_frames(data, T_orig, patch_sampling_cfg, frame_size, patch_size, T_fixed)
    # elif sampling_type == "random":
    #     T, P_row, P_col = sampling_random(data, patch_sampling_cfg, frame_size, patch_size)
    # elif sampling_type == "grid":
    #     if patch_sampling_cfg["mode"]=="inference":
    #         T, P_row, P_col = sampling_grid(data, T_orig, patch_sampling_cfg, frame_size, patch_size)
    #     else:
    #         T, P_row, P_col = sampling_grid(data, T_orig, patch_sampling_cfg, frame_size, patch_size, T_fixed)

    return sampling_segments(data, T, T_orig, P_row, P_col, patch_sampling_cfg, frame_size, patch_size)

# def sampling_full_frames(data, T, patch_sampling_cfg, frame_size, patch_size):
#     # -- select T samples with replacement for patches
#     verbose=False
#     num_patches = patch_sampling_cfg["num_patches"]
#     num_in_between_frames = patch_sampling_cfg["num_in_between_frames"]
#     if num_in_between_frames>0:
#         sampling_type = patch_sampling_cfg["sampling_type"]
#     if data.shape[0]> num_patches:
#         rep_param = False
#     else:
#         rep_param = True
        
#     # T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param)))
#     if T is None: 
#         T = torch.arange(0,data.shape[1]-1) # Select T samples with 
    
#     if verbose: print('[sampling_full_frames] T: ',T)
    
#     #Add in-between frames
#     if num_in_between_frames > 0:
#         if sampling_type == 'random':
#             in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
#             T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort

#         else: 
#             # just get the original coordinates of the data
#             T_tmp = torch.linspace(T[0], T[-1], steps=160) # just get the original coordinates of the data
#             ids_downsampled = np.linspace(0,len(T_tmp)-1,num=num_in_between_frames+len(T), dtype=np.int64)
#             T_tmp = T_tmp[ids_downsampled]
#             # print('T_tmp: ',T_tmp)
#             # print('T_tmp.shape: ',T_tmp.shape)
#             idx_overlapping = [np.argmin(np.abs(T_tmp.numpy()-T[idx].numpy())) for idx in range(len(T))] #Select the indexes that match the first 5 points used before
#             idx_non_overlapping = np.argwhere(np.isin(np.arange(len(T_tmp)),idx_overlapping,invert=True)).flatten()
#             in_between_frames = T_tmp[idx_non_overlapping]
#             T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
#             # print('idx_non_overlapping: ',idx_non_overlapping)
#             # print('T_tmp: ',T_tmp)
#             # print('T_tmp.shape: ',T_tmp.shape)
#             # in_between_frames = torch.linspace(T[0], T[-1], steps=num_in_between_frames+len(T)) #Assuming this will create duplicates 
#         if verbose: print('[sampling_full_frames] in_between_frames: ',in_between_frames)
        
        
        
#         #Check if there are duplicates and resample if there are
#         dup=np.array([0])
#         while dup.size != 0:
#             u, c = np.unique(T_tmp, return_counts=True)
#             dup = u[c > 1]
#             if dup.size != 0:
#                 print('[sampling_full_frames] There are duplicated time coordinates: ',dup)
#                 print('Resampling')
#                 # print(aaaa)
                
#                  #Add in-between frames
#                 in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
#                 T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
                
        
#         T = T_tmp
#         del T_tmp
    
#     # T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
    
#     P_row = torch.zeros_like(T,dtype=torch.int64)
#     P_col = torch.zeros_like(T,dtype=torch.int64)
    
#     return T, P_row, P_col

def sampling_full_frames(data, T, patch_sampling_cfg, frame_size, patch_size, T_fixed):
    # -- select T samples with replacement for patches
    verbose=False
    # num_patches = patch_sampling_cfg["num_patches"]
    num_in_between_frames = patch_sampling_cfg["num_in_between_frames"]
    if num_in_between_frames>0:
        sampling_type = patch_sampling_cfg["sampling_type"]
    # if data.shape[0]> num_patches:
    #     rep_param = False
    # else:
    #     rep_param = True
        
    # T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param)))
    if T is None: 
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
    
    if verbose: print('[sampling_full_frames] T: ',T)
    
    #Add in-between frames
    if num_in_between_frames > 0:
        if sampling_type == 'random':
            if T_fixed is None:
                in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
                T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
            else: # In this case, T_fixed already has some coordinates that we will include, so just sample the difference
                in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames-T_fixed.size(0)).uniform_(T[0], T[-1]))
                T_tmp,_ = torch.sort(torch.cat((T,T_fixed,in_between_frames))) # append the in-between and sort


        else: 
            # just get the original coordinates of the data
            T_tmp = torch.linspace(T[0], T[-1], steps=160) # just get the original coordinates of the data
            ids_downsampled = np.linspace(0,len(T_tmp)-1,num=num_in_between_frames+len(T), dtype=np.int64)
            T_tmp = T_tmp[ids_downsampled]
            # print('T_tmp: ',T_tmp)
            # print('T_tmp.shape: ',T_tmp.shape)
            idx_overlapping = [np.argmin(np.abs(T_tmp.numpy()-T[idx].numpy())) for idx in range(len(T))] #Select the indexes that match the first 5 points used before
            idx_non_overlapping = np.argwhere(np.isin(np.arange(len(T_tmp)),idx_overlapping,invert=True)).flatten()
            in_between_frames = T_tmp[idx_non_overlapping]
            T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
            # print('idx_non_overlapping: ',idx_non_overlapping)
            # print('T_tmp: ',T_tmp)
            # print('T_tmp.shape: ',T_tmp.shape)
            # in_between_frames = torch.linspace(T[0], T[-1], steps=num_in_between_frames+len(T)) #Assuming this will create duplicates 
        if verbose: 
            print('[sampling_full_frames] in_between_frames: ',in_between_frames)
            print('[sampling_full_frames] T_fixed: ',T_fixed)
            print('[sampling_full_frames] T: ',T)
        
        
        
        #Check if there are duplicates and resample if there are
        dup=np.array([0])
        while dup.size != 0:
            u, c = np.unique(T_tmp, return_counts=True)
            dup = u[c > 1]
            if dup.size != 0:
                print('[sampling_full_frames] There are duplicated time coordinates: ',dup)
                print('Resampling')
                # print(aaaa)
                
                #Add in-between frames
                if T_fixed is None:
                    in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames).uniform_(T[0], T[-1]))
                    T_tmp,_ = torch.sort(torch.cat((T,in_between_frames))) # append the in-between and sort
                else:
                    in_between_frames,_ = torch.sort(torch.FloatTensor(num_in_between_frames-T_fixed.size(0)).uniform_(T[0], T[-1]))
                    T_tmp,_ = torch.sort(torch.cat((T,T_fixed,in_between_frames))) # append the in-between and sort
                
        
        T = T_tmp
        del T_tmp
    
    # T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
    
    P_row = torch.zeros_like(T,dtype=torch.int64)
    P_col = torch.zeros_like(T,dtype=torch.int64)
    if verbose:
        print('[sampling_full_frames] final T: ',T)
        print('[sampling_full_frames] final P_row: ',P_row)
        print('[sampling_full_frames] final P_col: ',P_col)

    
    return T, P_row, P_col
    
def sampling_random(data, patch_sampling_cfg, frame_size, patch_size):
    # -- sample random patches
    num_patches = patch_sampling_cfg["num_patches"]
    n_pred = patch_sampling_cfg["num_patches_to_hide"]
#     num_frames = patch_sampling_cfg["num_frames"]

    # -- select T coordinates (one for each patch)
    if data.shape[1]> num_patches:
        rep_param = False
    else:
        rep_param = True
        
#     # Original
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]), num_patches, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches)))

##################################################################
#     # Sample random patches from the sequence, but get non-overlapping pathes for the last frame of the segment, to make sure we can make reconstruction
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
#     T_lastFrame = torch.tensor([data.shape[0]-1])
#     P_row_lastFrame = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
#     P_col_lastFrame = torch.linspace(0,frame_size-patch_size-1,steps=int(frame_size/patch_size),dtype=torch.int64)
    
#     T_lastFrame,P_row_lastFrame,P_col_lastFrame = torch.meshgrid(T_lastFrame,P_row_lastFrame,P_col_lastFrame)
#     T_lastFrame,P_row_lastFrame,P_col_lastFrame = T_lastFrame.flatten(),P_row_lastFrame.flatten(),P_col_lastFrame.flatten()
    
#     T = torch.cat((T,T_lastFrame))
#     P_row = torch.cat((P_row,P_row_lastFrame))
#     P_col = torch.cat((P_col,P_col_lastFrame))

# ##################################################################
#     # Sample random patches from the sequence, but sample n_pred from the last frame only (no need to make whole frame reconstruction for this configuration)
#     T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[0]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
#     P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
#     P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
#     T_lastFrame = torch.ones(n_pred,dtype=torch.int64)*(data.shape[0]-1)
#     P_row_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
#     P_col_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    
#     T = torch.cat((T,T_lastFrame))
#     P_row = torch.cat((P_row,P_row_lastFrame))
#     P_col = torch.cat((P_col,P_col_lastFrame))


# ##################################################################
#     # Sample random patches from the sequence, but sample n_pred from the last frame only (no need to make whole frame reconstruction for this configuration)
#     # Also, adding implementation to work on batches
    T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[1]-1), num_patches-n_pred, replace=rep_param))) # Select T samples with 
    
    P_row = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    P_col = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=num_patches-n_pred)))
    
    T_lastFrame = torch.ones(n_pred,dtype=torch.int64)*(data.shape[1]-1)
    P_row_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    P_col_lastFrame = torch.from_numpy(np.asarray(random.choices(np.arange(frame_size-patch_size), k=n_pred)))
    
    T = torch.cat((T,T_lastFrame))
    P_row = torch.cat((P_row,P_row_lastFrame))
    P_col = torch.cat((P_col,P_col_lastFrame))

    return T, P_row, P_col

def sampling_grid(data, patch_sampling_cfg, frame_size, patch_size,current_epoch=None):
# def sampling_grid(data, hparams, frame_size, patch_size):
    # -- sample patches in grid-like form
    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"]
    mode = patch_sampling_cfg["mode"]
#     grid_rows = frame_size//patch_size

    if data.shape[1]> num_patches:
        rep_param = False
    else:
        rep_param = True

    if mode=="inference":
#         if current_epoch%2==0:
            # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.arange(0,data.shape[1]-1) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
            
        
    else: 
        # To sample a grid from the frame and make sure that last frame is always sampled, because we want to predict that one everytime
        T = torch.from_numpy(np.sort(np.random.choice(np.arange(data.shape[1]-1), num_patches-1, replace=rep_param))) # Select T samples with 
        T = torch.cat((T,torch.tensor([data.shape[1]-1]))) # Append the last frame manually 
        P_row = torch.linspace(0,frame_size["rows"]-patch_size-1,steps=int(frame_size["rows"]/patch_size),dtype=torch.int64)
        P_col = torch.linspace(0,frame_size["cols"]-patch_size-1,steps=int(frame_size["cols"]/patch_size),dtype=torch.int64)
    #     T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Original, but potentially wrong
        T,P_row,P_col = torch.meshgrid(T,P_row,P_col) #Needs to invert x and y to match actual 
        T,P_row,P_col = T.flatten(),P_row.flatten(),P_col.flatten()
    
    return T, P_row, P_col

def sampling_segments(data, T, T_orig, P_row, P_col, patch_sampling_cfg, frame_size, patch_size):
    verbose=False
    if verbose: 
        print('[sampling_segments] data.shape: ',data.shape)
        print('[sampling_segments] T: ',T)
        print('[sampling_segments] P_row: ',P_row)
        print('[sampling_segments] P_col: ',P_col)
        print('[sampling_segments] T: {}, P_row: {}, P_col: {}'.format(T.shape, P_row.shape, P_col.shape))
        print('[sampling_segments] T_orig: ',T_orig)
        print('[sampling_segments] T_orig.shape: ',T_orig.shape)

    num_patches = patch_sampling_cfg["num_patches"]
    num_frames = patch_sampling_cfg["num_frames"] # NUmber of frames in teh segment
    n_pred = patch_sampling_cfg["num_patches_to_hide"] #Number of patches to hide, 
    n_frames_to_hide = patch_sampling_cfg["n_frames_to_hide"] #Number of frames to hide, 
    in_between_frame_init = patch_sampling_cfg["in_between_frame_init"]
    num_in_between_frames = patch_sampling_cfg["num_in_between_frames"]
    prob_replace_masked_token = patch_sampling_cfg["prob_replace_masked_token"] #prob of replacing a masked token. Previously it was 0.8.
    masking_type = patch_sampling_cfg["masking_type"]
    mode = patch_sampling_cfg["mode"]
    batch_size_segments = data.shape[0]
    if in_between_frame_init == 'interpolation':
        interpolation_kind = patch_sampling_cfg["interpolation_kind"]
#     print('num_patches: ',num_patches)
    
#     segm_frames = torch.zeros((num_patches, patch_size**2), dtype=torch.float32) # Using patches. This would work for patch=frame size
    
    #Orignal
#     segm_frames = torch.zeros((T.shape[0],patch_size**2),dtype=torch.float32) # Using patches 
    # segm_frames = torch.zeros((batch_size_segments,T.shape[0],patch_size**2),dtype=torch.float32) # Using patches. Original for square patches
    segm_frames = torch.zeros((batch_size_segments,T.shape[0],patch_size[0]*patch_size[1]),dtype=torch.float32) # Using patches. Original for square patches
    
    dict_special_tokens = {
        "MASK": 0.5
    }
    # MASK_FRAME = torch.from_numpy(np.ones(patch_size[0]*patch_size[1])*dict_special_tokens["MASK"])
    MASK_FRAME = torch.tensor(np.ones(patch_size[0]*patch_size[1])*dict_special_tokens["MASK"], requires_grad=True)
    
    if verbose: print('segm_frames.shape: ',segm_frames.shape)
    if in_between_frame_init == 'interpolation':
        # Compute interpolation between all the frames
        # data[idx1, int(T[idx]), P_row[idx]:P_row[idx]+patch_size[0], P_col[idx]:P_col[idx]+patch_size[1]]
        data_flat = torch.flatten(data, start_dim=2)
        
        if verbose: 
            print('[sampling_segments] data_flat.shape: ',data_flat.shape)
            print('[sampling_segments] segm_frames.shape: ',segm_frames.shape)
        # coeffs = natural_cubic_spline_coeffs(torch.arange(data.shape[1],dtype=torch.float).to(device),data_flat)
        # tmp_time = torch.FloatTensor([1,4,5]).to(device) #Change this to be randomly sampled indexes from the original data
        if verbose: print('n_frames_to_hide: ',n_frames_to_hide)
        tmp_time = np.sort(np.random.choice(np.arange(T_orig.shape[0]), n_frames_to_hide, replace=False)) # Select 'n_pred' frames that will be used to compute the interpolating function
        # tmp_time = np.sort(np.random.choice(np.arange(1,T_orig.shape[0]-1), n_frames_to_hide-2, replace=False)) # Select 'n_pred' frames that will be used to compute the interpolating function
        # tmp_time = np.array([0,tmp_time[0],T_orig.shape[0]-1])
        if verbose: 
            print('[sampling_segments] tmp_time.shape: ',tmp_time.shape)
            print('[sampling_segments] tmp_time: ',tmp_time)
            print('[sampling_segments] T_orig[tmp_time]: ',T_orig[tmp_time])
        # idxs = np.array([1,4,5])
        if verbose: 
            print('[sampling_segments] data_flat[:,tmp_time].shape: ',data_flat[:,tmp_time].shape)
            print('T[tmp_time].to(device): ',T_orig[tmp_time].to(device))
            print('data_flat[:,tmp_time].to(device): ',data_flat[:,tmp_time].to(device))
        
        # #For cubic spline 
        # coeffs = natural_cubic_spline_coeffs(T_orig[tmp_time].to(device),data_flat[:,tmp_time].to(device))
        # spline = NaturalCubicSpline(coeffs) #This is the function resulting from the interpolation
        # segm_frames = spline.evaluate(T.to(device))
        # if in_between_frame_init == 'interpolation':
    #     del spline, coeffs, data_flat, tmp_time

        # batch_values = torch.zeros(self.y.size(0),x.size(0),self.y.size(2)) #[batch_size, number_points, dim]
        # if verbose: 
        #     print('batch_values.shape: ',batch_values.shape)

        # # For linear interpolation 
        # # t_lin = self.points.repeat(self.y.size(2),1)
        # t_lin= T_orig[tmp_time].repeat(data_flat.size(2),1).to(device)
        # t_lin.requires_grad = False
        # if verbose: 
        #     print('t_lin.shape: ',t_lin.shape)
        #     print('t_lin.requires_grad: ',t_lin.requires_grad)

        # for idx_batch in range(segm_frames.size(0)):
        #     # x_lin = self.y[idx_batch,:].squeeze().T
        #     x_lin = data_flat[idx_batch,tmp_time].T.to(device)
        #     # y = y_orig[:,:-2]
        #     if verbose: 
        #         print('x_lin.shape: ',x_lin.shape)
        #         print('x_lin.requires_grad: ',x_lin.requires_grad)

        #     t_in_lin = T.repeat(data_flat.size(2),1).to(device)
        #     t_in_lin.requires_grad = False
        #     if verbose: 
        #         print('t_in_lin.shape: ',t_in_lin.shape)
        #         print('t_in_lin.requires_grad: ',t_in_lin.requires_grad)

        #     yq_cpu = interp1d(t_lin, x_lin, t_in_lin, None)
        #     yq_cpu = yq_cpu.detach()
        #     if verbose: 
        #         print('yq_cpu.T.shape: ',yq_cpu.T.shape)
        #         print('yq_cpu.requires_grad: ',yq_cpu.requires_grad)
        #     segm_frames[idx_batch,:] = yq_cpu.T  
        # del t_lin,t_in_lin, x_lin, yq_cpu

        # Using scioy for the interpolation
        t_lin= T_orig[tmp_time]#.repeat(data_flat.size(2),1).to(device)
        if verbose: print('t_lin: ',t_lin)
        x_lin = data_flat[:,tmp_time]
        if verbose: print('x_lin: ',x_lin)
        f = interpolate.interp1d(t_lin, x_lin.cpu().detach().numpy(),axis=1,kind=str(interpolation_kind)) #Time is on the axis=1
        
        segm_frames = torch.Tensor(f(T)).to(device)
        if verbose: print('segm_frames.shape: ',segm_frames.shape)  
        del t_lin, x_lin, tmp_time


        # tmp_out = spline.evaluate(T.to(device))
        # if verbose: 
            # print('tmp_out.shape: ',tmp_out.shape)
        
    elif in_between_frame_init != 'interpolation':
        # print('in_between_frame_init: ',in_between_frame_init)
        for idx1 in range(batch_size_segments):
            for idx in range(len(T)):
                # if verbose: print('patch_size: ',patch_size)
                # Only replace the coordinate in segm_frames if the T coordinate is in T_orig. Otherwise, already replace it with MASK_FRAME
                if T[idx] in T_orig:
                    if verbose: print('using the real coordinates: {}, T[idx]: {}'.format(idx,T[idx]))
                    tmp_idx = (T_orig == T[idx]).nonzero(as_tuple=True)[0]
                    segm_frames[idx1,idx, :] = data[idx1,tmp_idx, P_row[tmp_idx]:P_row[tmp_idx]+patch_size[0], P_col[tmp_idx]:P_col[tmp_idx]+patch_size[1]].flatten()
                else:
                    if in_between_frame_init == 'mask':
                        if verbose: print('Masking idx with cte: {}, T[idx]: {}'.format(idx,T[idx]))
                        segm_frames[idx1, idx, :] = MASK_FRAME
                    elif in_between_frame_init == 'random':
                        if verbose: print('Masking idx with random: {}, T[idx]: {}'.format(idx,T[idx]))
                        segm_frames[idx1, idx, :] = torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(-1.0, 1.0).squeeze()

                # if verbose: print('Not masking idx: {}, T[idx]: {}'.format(idx,T[idx]))
     
    # if verbose and in_between_frame_init=='interpolation': 
    #     fig,ax = plt.subplots(2,2)
    #     fig.suptitle('First interpolation', fontsize=16)
    #     ax=ax.ravel()
    #     ax[0].plot(T,tmp_out[0,:,0].cpu().squeeze())
    #     ax[0].scatter(T_orig,data_flat[0,:,0].cpu().squeeze())
    #     ax[0].scatter(T_orig[tmp_time],data_flat[0,tmp_time,0].cpu().squeeze())
    #     ax[1].plot(T,tmp_out[0,:,1].cpu().squeeze())
    #     ax[1].scatter(T_orig,data_flat[0,:,1].cpu().squeeze())
    #     ax[1].scatter(T_orig[tmp_time],data_flat[0,tmp_time,1].cpu().squeeze())
    #     ax[2].plot(T,tmp_out[1,:,0].cpu().squeeze())
    #     ax[2].scatter(T_orig,data_flat[1,:,0].cpu().squeeze())
    #     ax[2].scatter(T_orig[tmp_time],data_flat[1,tmp_time,0].cpu().squeeze())
    #     ax[3].plot(T,tmp_out[1,:,1].cpu().squeeze())
    #     ax[3].scatter(T_orig,data_flat[1,:,1].cpu().squeeze())
    #     ax[3].scatter(T_orig[tmp_time],data_flat[1,tmp_time,1].cpu().squeeze())
    #     plt.show()
    #     plt.close('all')
        

#     # Original
#     cand_pos = np.arange(1, segm_frames.shape[0]-1) # Do not replace the first and last tokens
#     if frame_size == patch_size:
#         random.shuffle(cand_pos)
        
    # Mask just the last patches (ie, the patches of the last frames)
    # Original 
    # if verbose: print('n_pred: ',n_pred)
    if in_between_frame_init == 'interpolation':
        # # Use this if the the candidates to masked can only be real frames (ie, integers)
        # n_pred = data.shape[1]
        # cand_pos = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]))).flatten() 
        # tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
        
        # Use this if the to get the coordinates of the real points by checking T_orig
        n_pred = data.shape[1]
        cand_pos = np.argwhere(np.isin(T.numpy(),T_orig)).flatten() 
        tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
     
        
        # # Use this to "mask" all frames (real and dummies). This just means all frames will be predicted
        # n_pred = len(T)
        # cand_pos = np.arange(len(T))
        # tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 

    else: 

        if n_pred==1 and n_frames_to_hide==1: # mask just the entire whole frame (ie, no patches)
            cand_pos = np.arange(segm_frames.shape[1]-n_pred, segm_frames.shape[1]) #This only masks the last "n_pred" frames
            tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
        elif n_pred==1 and n_frames_to_hide>1: # Mask the randomly "n_frames_to_hide" frames, but the whole frame
            if masking_type=='last_n_points': #For last n frames 
                # if mode=='inference': #manually remove 
                #     idx_used_for_train = np.array([0, 11, 22, 33, 44, 55, 66]) #The last 3 points were masked anyways
                #     idx_used_for_inference = np.argwhere(np.isin(np.arange(len(T_orig)),idx_used_for_train,invert=True))
                #     cand_pos = idx_used_for_inference.flatten() #np.concatenate((idx_used_for_inference.flatten(),idx_used_for_train[:3])) #Now manually add some back
                # else: 
                cand_pos = np.arange(segm_frames.shape[1]-n_frames_to_hide, segm_frames.shape[1]) #This only masks the last "n_pred" frames
                # tmp_masked_tokens = torch.empty(batch_size_segments, n_frames_to_hide, segm_frames.shape[2]) 
                # n_pred=n_frames_to_hide
            elif masking_type=='equal_spaced': # For equally spaced seen points
                n_frames_to_keep = segm_frames.shape[1]-n_frames_to_hide
                cand_pos = np.argwhere(np.isin(np.arange(len(T)),np.floor(np.linspace(0,len(T),n_frames_to_keep)),invert=False)).flatten()
                # cand_pos = np.argwhere(np.isin(np.arange(len(T)),np.arange(0,len(T))[::int(n_frames_to_hide/segm_frames.shape[1])],invert=False)).flatten()
                # cand_pos = np.argwhere(np.isin(np.arange(len(T)),np.arange(0,len(T))[::int(segm_frames.shape[1]/n_frames_to_hide)],invert=False)).flatten()
            elif masking_type=='random_masking':
                 # # For random point selection 
                # cand_pos = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]))).flatten() # The candidates to masked can only be real frames (ie, integers)

                if mode=='inference': #manually remove 
                    # # For my tests without the Sobolev loss
                    # idx_used_for_train = np.array([0, 11, 22, 33, 44, 55, 66, 77, 88, 99])
                    # idx_used_for_inference = np.argwhere(np.isin(np.arange(len(T_orig)),idx_used_for_train,invert=True))
                    # # print('random.shuffle(idx_used_for_train): ',np.random.shuffle(idx_used_for_train))
                    # np.random.shuffle(idx_used_for_train)
                    # print('idx_used_for_train: ',idx_used_for_train)
                    # print('idx_used_for_train[:3]: ',idx_used_for_train[:3])
                    # cand_pos = np.concatenate((idx_used_for_inference.flatten(),idx_used_for_train[:3])) #Now manually add some back

                    # # For the tests with Sobolev (double check if the coordinates selected in idx_used_for_train give the corrent time stamps)
                    # print('T_orig.shape: ',T_orig.shape)
                    # print('T_orig: ',T_orig)
                    # idx_used_for_train = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0]
                    # # print('train-tile: ',idx_used_for_train)
                    # # print('T[idx_used_for_train]: ',T[idx_used_for_train])
                    # # idx_used_for_train = np.array([0, 16, 32, 49, 65, 87, 104, 120, 137, 159]) # When using 160 points
                    # # idx_used_for_train = np.array([0, 6, 12, 18, 24, 33, 39, 45, 51, 59]) # When using 60 points  
                    # # idx_used_for_train = np.array([0, 12, 24, 37, 49, 66, 78, 90, 103, 119]) # When using 120 points  
                    # # idx_used_for_train = np.array([0, 59]) # When using 160 points  
                    # print('train-manual: ',idx_used_for_train)
                    # print('T[idx_used_for_train]: ',T[idx_used_for_train])
                    # idx_used_for_inference = np.argwhere(np.isin(np.arange(len(T_orig)),idx_used_for_train,invert=True))
                    # #print('inference: ',idx_used_for_inference)
                    # #cand_pos = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0]
                    # random.shuffle(idx_used_for_train)
                    # cand_pos = np.concatenate((idx_used_for_inference.flatten(),idx_used_for_train[:3]))
                    # # cand_pos = idx_used_for_inference.flatten()

                    #For the tests using random time points
                    idx_real_coordinates = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                    random.shuffle(idx_real_coordinates)
                    cand_pos = idx_real_coordinates[:3]

                else: 
                    # cand_pos = np.arange(0,len(T)) #Original
                    # cand_pos = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0] #For the sobolev on the 2D spirals
                    # random.shuffle(cand_pos)


                    if num_in_between_frames>0: #In this case, we need to make sure only the real coordinates are selected for masking
                        idx_real_coordinates = np.argwhere(np.isin(T,T_orig,invert=False)).flatten()
                        random.shuffle(idx_real_coordinates)
                        cand_pos = idx_real_coordinates[:n_frames_to_hide]

                    else: #in this case there are no dummy points
                        # idx_used_for_train = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0]
                        # #print('train: ',idx_used_for_train)
                        # idx_used_for_inference = np.argwhere(np.isin(np.arange(len(T_orig)),idx_used_for_train,invert=True))
                        # #print('inference: ',idx_used_for_inference)
                        # #cand_pos = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0]
                        # random.shuffle(idx_used_for_train)
                        # cand_pos = np.concatenate((idx_used_for_inference.flatten(),idx_used_for_train[:3]))

                        cand_pos = np.arange(0,len(T)) #Original
                        # cand_pos = np.tile(np.linspace(0,len(T_orig)-1,num=10, dtype=np.int64),(len(T_orig)-1,1))[0] #For the sobolev on the 2D spirals
                        random.shuffle(cand_pos)
                        if verbose: print('cand_pos: ',cand_pos)


            tmp_masked_tokens = torch.empty(batch_size_segments, n_frames_to_hide, segm_frames.shape[2]) 
            n_pred=n_frames_to_hide
            # cand_pos = np.sort(cand_pos) #original
            cand_pos = np.sort(cand_pos[:n_pred]) #original

        
    # if in_between_frame_init == 'interpolation':
    #     # # Use this if the the candidates to masked can only be real frames (ie, integers)
    #     # n_pred = data.shape[1]
    #     # cand_pos = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]))).flatten() 
    #     # tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
        
    #     # Use this if the to get the coordinates of the real points by checking T_orig
    #     n_pred = data.shape[1]
    #     cand_pos = np.argwhere(np.isin(T.numpy(),T_orig)).flatten() 
    #     tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
     
        
    #     # # Use this to "mask" all frames (real and dummies). This just means all frames will be predicted
    #     # n_pred = len(T)
    #     # cand_pos = np.arange(len(T))
    #     # tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 
    
    if verbose: 
        print('[sampling_segments] n_pred: ',n_pred)
        print('[sampling_segments] cand_pos: ',cand_pos)
    #     random.shuffle(cand_pos)
    #     print('[after shuffle] cand_pos: ',cand_pos)
    #     print(aaa)

    tmp_masked_pos = []
    
#    #Original
#     tmp_masked_tokens = torch.empty(n_pred, segm_frames.shape[1]) # replace this constant by a variable
    # tmp_masked_tokens = torch.empty(batch_size_segments, n_pred, segm_frames.shape[2]) 


    
#     #Original 
#     for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
#         tmp_masked_tokens[idx_pos, :] = segm_frames[pos, :]    
#         tmp_masked_pos.append(pos)
#         if random.random() < 1: # 0.8
#             segm_frames[pos, :] = MASK_FRAME

    if in_between_frame_init == 'interpolation': #If interpolation is being used, we want to predict all the frames. So copy them to 'masked_tokens'
        for idx1 in range(batch_size_segments):
            for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
                tmp_masked_tokens[idx1, idx_pos, :] = data[idx1, idx_pos, :].flatten() #Get the tokens from the input data (ie, before inteporlation), since this is what we are trying to predict   
                tmp_masked_pos.append(pos)
    else: 
        # if in_between_frame_init=='random':
        #     min_val = segm_frames.min().detach().numpy().astype(np.float)
        #     max_val = segm_frames.max().detach().numpy().astype(np.float)
        if verbose: print('Using masking. The is in_between_frame_init = ', in_between_frame_init)
        # Function to mask some of the real frames for prediction
        for idx1 in range(batch_size_segments):
            for pos, idx_pos in zip(cand_pos[:n_pred], range(n_pred)):
                if verbose: print('tmp_masked_tokens[{}, {}, :] = segm_frames[{}, {}, :]'.format(idx1, idx_pos,idx1,pos))
                tmp_masked_tokens[idx1, idx_pos, :] = segm_frames[idx1, pos, :]    
                tmp_masked_pos.append(pos)
                prob = random.random()
                if prob < prob_replace_masked_token: # 80% randomly change token to mask token
                    segm_frames[idx1, pos, :] = MASK_FRAME
                elif prob < 0.9: # 10% randomly change token to random token of the data
                    rand_curve = np.random.randint(batch_size_segments)
                    if verbose: print('rand_curve: ',rand_curve)
                    # rand_token = np.random.randint(segm_frames.shape[1]) #Before
                    rand_token = np.random.randint(data.shape[1]) #Before
                    if verbose: print('rand_token: ',rand_token)
                    segm_frames[idx1, pos, :] = data[rand_curve, rand_token, :].squeeze(-1)
                # 10% randomly change token to current token, which in this case is the default

                    # if in_between_frame_init=='mask':
                    #     segm_frames[idx1, pos, :] = MASK_FRAME
                    # elif in_between_frame_init=='random':
                    #     if verbose: print('replacing elem ' + str(pos) + ' by rand value')
                    #     # print('segm_frames.shape: ',segm_frames.shape)
                    #     # print('patch_size: ',patch_size)
                    #     # print('min_val, max_val: ',min_val, max_val)
                    #     # print('torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(min_val, max_val): ',torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(min_val, max_val))
                    #     # print('torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(min_val, max_val).shape: ',torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(min_val, max_val).shape)
                    #     # print('segm_frames[idx1, pos, :].shape: ',segm_frames[idx1, pos, :].shape)
                    #     segm_frames[idx1, pos, :] = torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(-1.0, 1.0).squeeze()
                    #     # segm_frames[idx1, pos, :] = torch.FloatTensor(patch_size[0], patch_size[1]).uniform_(min_val, max_val).squeeze()

#     # Original
#     masked_pos = torch.zeros((1, n_pred), dtype=torch.int64)
#     masked_pos[0, :len(tmp_masked_pos)] = torch.tensor(np.array(tmp_masked_pos, dtype=np.int64))# dtype=torch.int64)
#     masked_pos = torch.from_numpy(np.array(masked_pos))
#     masked_tokens = []
#     masked_tokens.append(tmp_masked_tokens)
#     masked_tokens = torch.stack(masked_tokens, dim=0)
#     masked_tokens = torch.reshape(
#         masked_tokens,
#         (masked_tokens.shape[0]*masked_tokens.shape[1], masked_tokens.shape[2])
#     )
#     masked_tokens = masked_tokens.squeeze().type_as(data)

#     masked_pos = torch.zeros((batch_size_segments, n_pred), dtype=torch.int64)
    masked_pos = torch.tensor(np.array(tmp_masked_pos, dtype=np.int64)).reshape(batch_size_segments,n_pred)
    masked_pos = torch.from_numpy(np.array(masked_pos))
    
    if verbose: 
        # print('len(tmp_masked_pos): ',len(tmp_masked_pos))
        print('[sampling_segments] tmp_masked_pos: ',tmp_masked_pos)
        print('[sampling_segments] tmp_masked_tokens.shape: ',tmp_masked_tokens.shape)
        print('[sampling_segments] masked_pos.shape: ',masked_pos.shape)
        print('[sampling_segments] masked_pos: ',masked_pos)
#     masked_tokens = []
#     masked_tokens.append(tmp_masked_tokens)
#     masked_tokens = torch.stack(masked_tokens, dim=0)
#     masked_tokens = torch.reshape(
#         masked_tokens,
#         (masked_tokens.shape[0]*masked_tokens.shape[1], masked_tokens.shape[2])
#     )
#     masked_tokens = tmp_masked_tokens.reshape(batch_size_segments,num_patches, patch_size, patch_size) #What was using for test 402
    masked_tokens = tmp_masked_tokens.reshape(batch_size_segments,n_pred, patch_size[0], patch_size[1])
    if verbose: print('[sampling_segments] masked_tokens.shape: ', masked_tokens.shape)
    masked_tokens = masked_tokens.type_as(data)

    return T, P_row, P_col, segm_frames, masked_pos, masked_tokens


def plot_training_curves(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path):
    # block displaying plots during training
    plt.ioff()

    tmp_size = np.minimum(masked_tokens.shape[0], 10)
    fig, ax = plt.subplots(tmp_size, 2, figsize=(10,50))

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos.detach().cpu().numpy().flatten()
    
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')
#     ax.plot(train_loss_epoch)
    ax.plot(val_loss.detach().cpu())
    ax.set_yscale('log')
    ax.legend(['Validation'], fontsize=15)
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15) 
    ax.set_title('Total losses (Contrast labels)', fontsize=15)
    ax.set_aspect('auto')

#     ax2.plot(r2_landmarks_train)
#     ax2.plot(r2_landmarks_val)
#     ax2.plot(r2_reg_train)
    ax2.plot(val_r2)
    #             ax2.legend(['Rsquared'], fontsize=15)
    ax2.legend(['Val_LM'], fontsize=15)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('R^2', fontsize=15) 
    ax2.set_title('R^2', fontsize=15)
    ax2.set_aspect('auto')
    plt.show()
        
    plt.savefig(os.path.join(save_path, "plots", f"training-curves-epoch-{epoch:05d}.png"))
    plt.close('all')

def plot_attention_weights(scores,epoch, save_path, mode): 
    from matplotlib.gridspec import GridSpec

    verbose=False
    # print('\n\nbatch_idx: ',batch_idx)
    # scores = torch.tensor(scores, device = 'cpu')
    scores = torch.stack(scores) # [num_layers, batch, num_heads, tokens, tokens]
    # print('scores.shape: ',scores.shape)
    # all_att = np.concatenate(scores,axis=0) #.cpu().detach().numpy()
    all_att = scores[:,0,:]#.cpu().detach().numpy() # Select just the first curve
    if verbose: print('all_att.shape: ',all_att.shape)
    del scores
    
    mean_att_per_patch_over_time = all_att.mean(axis=(0))
    # fig,ax = plt.subplots(2,2,facecolor='w')
    # create objects
    fig = plt.figure(figsize=(10, 5), facecolor='w')
    gs = GridSpec(nrows=2, ncols=4)

    # ax = ax.ravel()
    # for idx_head in range(4):
    idx_head=0
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax0.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=1
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax1.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=2
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax2.set_title('head ' + str(idx_head)+',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    idx_head=3
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(mean_att_per_patch_over_time[idx_head,:].cpu().detach().numpy())
    ax3.set_title('head ' + str(idx_head) +',ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time[idx_head,:]).cpu().detach().numpy()))

    ax4 = fig.add_subplot(gs[:, 2:])
    mean_att_per_patch_over_time = all_att.mean(axis=(0,1))
    ax4.imshow(mean_att_per_patch_over_time.cpu().detach().numpy())
    ax4.set_title('Mean head&layers, ent: ' + str(vn_eig_entropy(mean_att_per_patch_over_time).cpu().detach().numpy()) )
    # ax4.plot(time, score)

    # fig.tight_layout()

    # plt.show()
    
    # mean_att_per_patch_over_time = all_att.mean(axis=(0,1))
    # print('mean_att_per_patch_over_time.shape: ',mean_att_per_patch_over_time.shape)
    # fig,ax = plt.subplots(facecolor='w')
    # ax.imshow(mean_att_per_patch_over_time)
    # ax.set_title('Mean head&layers')
    fig.tight_layout()
    # plt.show()

    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"att-weight-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"att-weight-epoch-{epoch:05d}-{mode}.png"))
        plt.close('all')
    else: 
        plt.show()
    
def plot_predictions_curves(data,T, T_orig,P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,
                     epoch, val_loss, val_r2, save_path, mode):
    # block displaying plots during training
    verbose=False
    data = data.squeeze(-1)
    if masked_tokens is not None: masked_tokens = masked_tokens.squeeze(-1)
    # logits_lm = logits_lm.squeeze()
    # if logits_lm_with_dummy is not None:
    #     logits_lm_with_dummy = logits_lm_with_dummy.squeeze(-1)
     
    plt.ioff()
    if verbose:
        print('[plot_predictions_curves] masked_tokens.shape: ',masked_tokens.shape)

    patches_to_plot = 32 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]
    #     masked_tokens = masked_tokens[None,:]
        
    u, c = np.unique(T, return_counts=True)
    dup = u[c > 1]
    if dup.size != 0:
        print('[plot_predictions_curves] There are duplicated time coordinates: ',dup)
        print(aaa)
        
        
    # if masked_tokens.shape[-1]==1 : 
    #     masked_tokens = torch.transpose(masked_tokens, 1, 2)
    #     logits_lm = torch.transpose(logits_lm, 1, 2)
    if masked_tokens is None:
        ref_data = data
    else:
        ref_data = masked_tokens
    tmp_size = np.minimum(ref_data.shape[-1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        n_rows = int(np.floor(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        if n_cols*n_rows<ref_data.shape[1]: n_rows+=1
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(10,5), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    if ref_data.shape[-1] > patches_to_plot:
        list_idx = np.sort(np.random.choice(np.arange(ref_data.shape[-1]), tmp_size, replace=True))
    else: 
        list_idx = np.arange(ref_data.shape[-1])
    # print('list_idx: ',list_idx)
    if verbose:
        print('[plot_predictions_curves] list_idx: ',list_idx)
        print('[plot_predictions_curves] masked_pos: ',masked_pos)
        print('[plot_predictions_curves] T[masked_pos[0,:]: ',T[masked_pos[0,:]])
        print('[plot_predictions_curves] logits_lm.shape: ',logits_lm.shape)
        print('[plot_predictions_curves] T.shape: ',T.shape)
        if logits_lm_with_dummy is not None:
            print('[plot_predictions_curves] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        # print('[plot_predictions_curves] dev_logits_lm.shape: ',dev_logits_lm.shape)

    for idx_tmp in list_idx:
        # real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(masked_tokens.shape[2]),invert=False)).flatten() #returns the indexes of the dummy frames
        ax[c].scatter(T_orig, data[0,:,idx_tmp].detach().cpu(), label='Data')
        if masked_tokens is not None:
            ax[c].scatter(T[masked_pos[0,:]], masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data') # For regular Transformer training
            # ax[c].scatter(T_orig, masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data') # For inference and training with dummy points 
        ax[c].set_title("Dim " + str(idx_tmp))
        # ax[c].set_xticks([])
        # ax[c].set_yticks([])
        # c+=1

        # A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
        # # A = logits_lm[0,idx_tmp,:].flatten()
        # B = masked_tokens[0,idx_tmp,:].detach().cpu().numpy().flatten()
        # r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


        # ax[c].plot(T, logits_lm[0,:,idx_tmp].detach().cpu(),c='r', label='model')
        if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
            # ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
            if mode=='inference':
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
                ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='r', label='model')
            else: 
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
        else: 
            ax[c].plot(T[masked_pos[0,:]], logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
            # ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
        if dev_logits_lm is not None:
            ax[c].plot(T, dev_logits_lm[0,:,idx_tmp].squeeze(), label='dy/dt')
        # ax[c].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
        # ax[c].set_xticks([])
        # ax[c].set_yticks([])
        fig.tight_layout()
        ax[c].legend()
        c+=1


    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
        plt.close('all')
    else: 
        plt.show()
    del dev_logits_lm

def plot_predictions_curves2(data,T, T_orig,data_all,T_all,P_row, P_col, masked_tokens, masked_pos, logits_lm, dev_logits_lm, logits_lm_with_dummy, patch_size,
                     epoch, val_loss, val_r2, save_path, in_between_frame_init, mode):
    # block displaying plots during training
    verbose=True
    data = data.squeeze(-1)
    if masked_tokens is not None: masked_tokens = masked_tokens.squeeze(-1)
    # logits_lm = logits_lm.squeeze()
    if logits_lm_with_dummy is not None:
        logits_lm_with_dummy = logits_lm_with_dummy.squeeze(-1)
     
    plt.ioff()
    if verbose:
        print('[plot_predictions_curves] masked_tokens.shape: ',masked_tokens.shape)

    patches_to_plot = 32 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]
    #     masked_tokens = masked_tokens[None,:]
        
    u, c = np.unique(T, return_counts=True)
    dup = u[c > 1]
    if dup.size != 0:
        print('[plot_predictions_curves] There are duplicated time coordinates: ',dup)
        print(aaa)
        
        
    # if masked_tokens.shape[-1]==1 : 
    #     masked_tokens = torch.transpose(masked_tokens, 1, 2)
    #     logits_lm = torch.transpose(logits_lm, 1, 2)
    if masked_tokens is None:
        ref_data = data
    else:
        ref_data = masked_tokens
    tmp_size = np.minimum(ref_data.shape[-1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        n_rows = int(np.floor(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
        if n_cols*n_rows<ref_data.shape[1]: n_rows+=1
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(10,5), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    if ref_data.shape[-1] > patches_to_plot:
        list_idx = np.sort(np.random.choice(np.arange(ref_data.shape[-1]), tmp_size, replace=True))
    else: 
        list_idx = np.arange(ref_data.shape[-1])
    # print('list_idx: ',list_idx)
    if verbose:
        print('[plot_predictions_curves] list_idx: ',list_idx)
        print('[plot_predictions_curves] masked_pos: ',masked_pos)
        print('[plot_predictions_curves] T[masked_pos[0,:]: ',T[masked_pos[0,:]])
        # print('[plot_predictions_curves] logits_lm.shape: ',logits_lm.shape)
        print('[plot_predictions_curves] T.shape: ',T.shape)
        if logits_lm_with_dummy is not None:
            print('[plot_predictions_curves] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        # print('[plot_predictions_curves] dev_logits_lm.shape: ',dev_logits_lm.shape)

    for idx_tmp in list_idx:
        # real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(masked_tokens.shape[2]),invert=False)).flatten() #returns the indexes of the dummy frames
        # ax[c].scatter(T_orig, data[0,:,idx_tmp].detach().cpu(), label='Data')
        ax[c].scatter(T_all, data_all[0,:,idx_tmp].detach().cpu(),c='g', label='Data',alpha=0.2) 
        if masked_tokens is not None:
            # ax[c].scatter(T[masked_pos[0,:]], masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data')
            # ax[c].scatter(T_orig, masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data')
            if in_between_frame_init=='mask':
                ax[c].scatter(T_orig, data[0,:,idx_tmp].detach().cpu(), label='Sampled') # For the BERT framework
                ax[c].scatter(T[masked_pos[0,:]], masked_tokens[0,:,idx_tmp].detach().cpu(), label='Hidden Data') #For the BERT framework
            elif in_between_frame_init=='interpolation':
                ax[c].scatter(T_orig, masked_tokens[0,:,idx_tmp].detach().cpu(), label='Sampled') # For the CST framework
        ax[c].set_title("Dim " + str(idx_tmp))
        # ax[c].set_xticks([])
        # ax[c].set_yticks([])
        # c+=1

        # A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
        # # A = logits_lm[0,idx_tmp,:].flatten()
        # B = masked_tokens[0,idx_tmp,:].detach().cpu().numpy().flatten()
        # r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


        # ax[c].plot(T, logits_lm[0,:,idx_tmp].detach().cpu(),c='r', label='model')
        if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
            # ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
            if mode=='inference':
                # ax[c].scatter(T_all, data_all[0,:,idx_tmp].detach().cpu(),c='g', label='Data',alpha=0.2)
                # ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
                ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='r', label='model')
            else: 
                ax[c].scatter(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model',alpha=0.2)
        else: 
            ax[c].plot(T[masked_pos[0,:]], logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
            # ax[c].plot(T, logits_lm_with_dummy[0,:,idx_tmp].detach().cpu(),c='g', label='model')
        if dev_logits_lm is not None:
            ax[c].plot(T, dev_logits_lm[0,:,idx_tmp].squeeze(), label='dy/dt')
        # ax[c].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
        # ax[c].set_xticks([])
        # ax[c].set_yticks([])
        fig.tight_layout()
        ax[c].legend()
        c+=1


    if save_path is not None: 
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
        plt.close('all')
    else: 
        plt.show()
    del dev_logits_lm


def plot_predictions_InBetweenPoints_curves(data, T, P_row, P_col, masked_pos, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, mode, replace_by_real_frame=False):
    # block displaying plots during training
    verbose=False
    plt.ioff()
    pca = PCA(n_components=2)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy().copy().squeeze()
    if verbose:
        print('T.shape: ',T.shape)
        print('T: ',T)
        print('data.shape: ',data.shape)
        print('masked_pos.shape: ',masked_pos.shape)
        # print('masked_pos_real.shape: ',masked_pos_real.shape)
        print('logits_lm.shape: ',logits_lm.shape)
        
    u, c = np.unique(T, return_counts=True)
    dup = u[c > 1]
    if dup.size != 0:
        print('[plot_predictions_InBetweenPoints_curves] There are duplicated time coordinates: ',dup)
        print(aaa)

    patches_to_plot = 36 
    fig, ax = plt.subplots(1,2, figsize=(10,5),facecolor='w', sharey=True)

    ax = ax.ravel()

    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    # print('logits_lm[0].shape: ',logits_lm[0].shape)
    # print('logits_lm_pca.shape: ',logits_lm_pca.shape)
    
    
    # dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    dummy_idxs = np.argwhere(np.isin(np.arange(len(T)),masked_pos[0,:].cpu(),invert=True)).flatten() #returns the indexes of the dummy frames. So dummy_idxs and masked_pos should be complementary
    real_idxs = masked_pos[0,:].cpu().flatten() #returns the indexes of the real frames
    # pred_idx = masked_pos_real[0]
    
    # print('dummy_idxs: ',dummy_idxs)
    # print('real_idxs: ',real_idxs)
    # print('pred_idx: ',pred_idx)
    # real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()] #returns the indexes of the real frames that were not used for prediction
    if verbose:
        print('[plot_predictions_InBetweenPoints_curves] dummy_idxs.shape: ',dummy_idxs.shape)
        print('[plot_predictions_InBetweenPoints_curves] dummy_idxs: ',dummy_idxs)
        print('[plot_predictions_InBetweenPoints_curves] real_idxs: ',real_idxs)
        print('[plot_predictions_InBetweenPoints_curves] T[real_idxs]: ',T[real_idxs])
        
    # print('real_idxs2.size: ',real_idxs2.size)
    # if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        for idx_tmp,idx_tmp2 in zip(real_idxs, range(len(real_idxs))):
            if verbose: 
                print('T[idx_tmp]: ',T[idx_tmp])
                print('idx_tmp2: ',idx_tmp2)
            # print('logits_lm[0,{}] = data[0,{},:]'.format(idx_tmp,int(T[idx_tmp])))
            # logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy().squeeze()
            logits_lm_copy[0,idx_tmp] = data[0,idx_tmp2,:].detach().cpu().numpy().squeeze()
    else:
        fig.suptitle('Reconst. Frames', fontsize=16)
    # else:
        # print('not replacing reconstructed')
    logits_lm_copy = logits_lm_copy[0,:] # Just the first curve of the batch
    logits_lm = logits_lm[0,:].detach().cpu().numpy()#.copy()
    # logits_lm_pca = pca.fit_transform(logits_lm_copy[0].reshape(logits_lm[0].shape[0],-1))
    # print('logits_lm_copy.shape: ',logits_lm_copy.shape)
    # print('logits_lm.shape: ',logits_lm.shape)
    #     
    all_idxs = np.sort(np.concatenate((dummy_idxs,real_idxs)))
    if dummy_idxs.size != 0: ax[0].scatter(logits_lm_copy[dummy_idxs,0], logits_lm_copy[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5) #If there are no dummy, don't plot it
    # if np.array_equal(real_idxs, pred_idx) and replace_by_real_frame: #If real=pred and points were replaced, just plot Data
    #     ax[0].scatter(logits_lm_copy[real_idxs2,0], logits_lm_copy[real_idxs2,1], label='Data', c='green') 
        
    # elif np.array_equal(real_idxs, pred_idx) and not replace_by_real_frame: #If real=pred and points were NOT replaced, just plot Pred
    #     ax[0].scatter(logits_lm_copy[pred_idx,0], logits_lm_copy[pred_idx,1], label='Pred', c='red')
        
    # elif not np.array_equal(real_idxs, pred_idx): #Plot both If real!=pred 
    if replace_by_real_frame:
        ax[0].scatter(logits_lm_copy[real_idxs,0], logits_lm_copy[real_idxs,1], label='Data', c='green') 
    else: 
        ax[0].scatter(logits_lm_copy[all_idxs,0], logits_lm_copy[all_idxs,1], label='Pred', c='red')
        
    ax[0].plot(logits_lm[:,0], logits_lm[:,1], label='Model', c='red')
    ax[0].set_title('All points')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    fig.tight_layout()
    ax[0].legend()
        # c+=1
    
    # # for idx_tmp in range(len(logits_lm_pca)): #masked_pos.shape[1]):
    #     # color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
    #     # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    # ax[2].scatter(logits_lm_pca[:,0], logits_lm_pca[:,1], c=T, alpha=0.5)
    # # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    # ax[2].set_title('All points')
    # ax[2].set_xticks([])
    # ax[2].set_yticks([])

    # plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca-all-points_epoch-{epoch:05d}.png"))
    
    #Now plot just the predicted frames and the dummy ones
    # print('T: ',T)
    
    # print('dummy_idxs: ',dummy_idxs)
    # all_idxs = np.concatenate((dummy_idxs,masked_pos_real[0]))
    # print('all_idxs: ',all_idxs)
    
    # for idx_tmp in all_idxs: #masked_pos.shape[1]):
    #     color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
    #     # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    #     if int(T[idx_tmp])==T[idx_tmp]: ax[1].scatter(logits_lm_pca[idx_tmp,1], logits_lm_pca[idx_tmp,0], label='Data', c='red', alpha=0.5)
    #     else: ax[1].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,1], label='Dummy', c='blue', alpha=0.5)
    #     # ax[1].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], c=color_point, alpha=0.5)
    #     # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    # ax[1].scatter(logits_lm_copy[dummy_idxs,0], logits_lm_copy[dummy_idxs,1], label='Dummy', c=T[dummy_idxs])#, alpha=0.5)
    # ax[1].scatter(logits_lm_copy[masked_pos_real[0],0], logits_lm_copy[masked_pos_real[0],1], label='Data', c=T[masked_pos_real[0]])
    ax[1].scatter(logits_lm_copy[:,0], logits_lm_copy[:,1], label='Dummy', c=T)#, alpha=0.5)
    ax[1].plot(logits_lm[:,0], logits_lm[:,1], label='Model', c='red')
    ax[1].set_title('Dummy and Pred')
    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    ax[1].legend()
        # c+=1

    # ax[3].scatter(logits_lm_pca[all_idxs,0], logits_lm_pca[all_idxs,1], c=T[all_idxs], alpha=0.5)
    # # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    # ax[3].set_title('Dummy and Pred')
    # ax[3].set_xticks([])
    # ax[3].set_yticks([])
    # # plt.show()
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_{mode}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_{mode}_ReconsFrames.png"))
        plt.close('all')
    

    
# def plot_predictions(data,T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,
#                      epoch, val_loss, val_r2, range_imshow, save_path, mode):
#     verbose=False
#     # block displaying plots during training
#     plt.ioff()

#     if verbose:
#         print('[plot_predictions] T.shape: ',T.shape)
#         print('[plot_predictions] T: ',T)
#         print('[plot_predictions] P_row.shape: ',P_row.shape)
#         print('[plot_predictions] masked_tokens.shape: ',masked_tokens.shape)
#         print('[plot_predictions] masked_pos.shape: ',masked_pos.shape)
#         print('[plot_predictions] logits_lm.shape: ',logits_lm.shape)

#     patches_to_plot = 36 
#     if logits_lm.dim()==1:
#         logits_lm = logits_lm[None,:]
#         masked_tokens = masked_tokens[None,:]
        
#     if masked_tokens.shape[-1]==1 : 
#         masked_tokens = torch.transpose(masked_tokens, 1, 2)
#         logits_lm = torch.transpose(logits_lm, 1, 2)
        
#     if data.shape[1]==logits_lm.shape[1]: #In this case, we are plotting all the points of the input sequence
#         fig, ax = plt.subplots(2,data.shape[1], figsize=(30,5), facecolor='w')
#         for idx_tmp in range(data.shape[1]):
#             ax[0,idx_tmp].imshow(data[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
#             if idx_tmp in masked_pos:
#                 ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}",color= 'r')#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
#             else: 
#                 ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}")#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
#             ax[0,idx_tmp].set_xticks([])
#             ax[0,idx_tmp].set_yticks([])

#             A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
#             B = data[0,idx_tmp,:].detach().cpu().numpy().flatten()
#             r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


#             ax[1,idx_tmp].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
#             ax[1,idx_tmp].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
#             ax[1,idx_tmp].set_xticks([])
#             ax[1,idx_tmp].set_yticks([])

#     else: 
#         tmp_size = np.minimum(2*masked_tokens.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
#         if verbose: print('[plot_predictions] tmp_size: ',tmp_size)
#         if tmp_size > 2:
#             n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
#             n_rows = int(np.floor(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
#             if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
#             if n_cols*n_rows<2*masked_tokens.shape[1]: 
#                 n_rows+=1
#             if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
#             fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w')
#         else:
#             fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

#         ax = ax.ravel()
#         c=0
#         tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
        
#         if masked_tokens.shape[1] > patches_to_plot:
#             list_idx = np.sort(np.random.choice(np.arange(masked_tokens.shape[1]), tmp_size, replace=True))
#         else: 
#             list_idx = np.arange(masked_tokens.shape[1])
#         # print('list_idx: ',list_idx)
        
#         for idx_tmp in list_idx:
#             ax[c].imshow(masked_tokens[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
#             ax[c].set_title(f"masked_tokens[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}, row: {P_row[tmp_maskpos[idx_tmp]]}, col: {P_col[tmp_maskpos[idx_tmp]]}, masked_pos[{idx_tmp}]: {tmp_maskpos[idx_tmp]}")
#             ax[c].set_xticks([])
#             ax[c].set_yticks([])
#             c+=1

#             A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
#             B = masked_tokens[0,idx_tmp,:].detach().cpu().numpy().flatten()
#             r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


#             ax[c].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
#             ax[c].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
#             ax[c].set_xticks([])
#             ax[c].set_yticks([])
#             c+=1
#         # plt.show()

#     if save_path is not None: 
#         plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
#     plt.close('all')


def plot_predictions(data,T,T_orig, P_row, P_col, masked_tokens, masked_pos,logits_lm, 
    logits_lm_with_dummy, patch_size,epoch, val_loss, val_r2, range_imshow, save_path, mode):
    verbose=False
    # block displaying plots during training
    plt.ioff()
    
    #verbose=True
    if verbose:
        print('[plot_predictions] T.shape: ',T.shape)
        print('[plot_predictions] T: ',T)
        print('[plot_predictions] T_orig: ',T_orig)
        print('[plot_predictions] data.shape: ',data.shape)
        print('[plot_predictions] P_row.shape: ',P_row.shape)
        print('[plot_predictions] masked_tokens.shape: ',masked_tokens.shape)
        print('[plot_predictions] masked_pos.shape: ',masked_pos.shape)
        # print('[plot_predictions] logits_lm.shape: ',logits_lm.shape)
        if logits_lm_with_dummy is not None:
            print('[plot_predictions_curves] logits_lm_with_dummy.shape: ',logits_lm_with_dummy.shape)
        

    patches_to_plot = 36 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]
    #     masked_tokens = masked_tokens[None,:]
        
    # if masked_tokens.shape[-1]==1 : 
    #     masked_tokens = torch.transpose(masked_tokens, 1, 2)
    #     logits_lm = torch.transpose(logits_lm, 1, 2)

    if logits_lm_with_dummy is None:
        ref_data = data
    else:
        ref_data = logits_lm_with_dummy
        
    if data.shape[1]==logits_lm_with_dummy.shape[1]: #In this case, we are plotting all the points of the input sequence. No dummies
        fig, ax = plt.subplots(2,data.shape[1], figsize=(30,5), facecolor='w')
        for idx_tmp in range(data.shape[1]):
            ax[0,idx_tmp].imshow(data[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            if idx_tmp in masked_pos:
                ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}",color= 'r')#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
            else: 
                ax[0,idx_tmp].set_title(f"data[{idx_tmp}], t: {T[idx_tmp]:.2f}")#, row: {P_row[idx_tmp]}, col: {P_col[idx_tmp]}")
            ax[0,idx_tmp].set_xticks([])
            ax[0,idx_tmp].set_yticks([])

            A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
            B = data[0,idx_tmp,:].detach().cpu().numpy().flatten()
            r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


            ax[1,idx_tmp].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[1,idx_tmp].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
            ax[1,idx_tmp].set_xticks([])
            ax[1,idx_tmp].set_yticks([])

    else: 
        tmp_size = np.minimum(2*ref_data.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if verbose: print('[plot_predictions] tmp_size: ',tmp_size)
        if tmp_size > 2:
            n_cols = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
            n_rows = int(np.ceil(np.sqrt(tmp_size))) # 2x because I want to plot masked patch and prediction 
            if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
            if n_cols*n_rows<2*ref_data.shape[1]: 
                n_rows+=1
            if verbose: print('[plot_predictions] n_cols: {}, n_rows: {}'.format(n_cols,n_rows))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w')
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')

        ax = ax.ravel()
        c=0
        tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
        
        if ref_data.shape[1] > patches_to_plot:
            list_idx = np.sort(np.random.choice(np.arange(masked_tokens.shape[1]), tmp_size, replace=True))
        else: 
            list_idx = np.arange(ref_data.shape[1])
        if verbose: print('list_idx: ',list_idx)
        
        for idx_tmp in list_idx:
            if logits_lm_with_dummy is not None and len(T)==len(logits_lm_with_dummy[0,:]):
                ax[c].imshow(logits_lm_with_dummy[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
                ax[c].set_title("logit[{}], t: {:.3f}, row: {}, col: {}".format(idx_tmp,T[idx_tmp],P_row[idx_tmp],P_col[idx_tmp]),fontsize=18)
                ax[c].set_xticks([])
                ax[c].set_yticks([])
                c+=1
#                 if T[idx_tmp] in T_orig: #Mark as a real point and plot the of the real frame right below
#                     idx_real = np.argwhere(T[idx_tmp]==T_orig).numpy()
#                     ax[c].imshow(data[0,idx_real,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
#                     ax[c].set_title("data{}, t: {:.3f}".format(idx_real[0],T[idx_tmp]),color= 'red', fontsize=18)
#                     ax[c].set_xticks([])
#                     ax[c].set_yticks([])
#                     c+=1

            # ax[c].imshow(masked_tokens[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            # ax[c].set_title(f"masked_tokens[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}, row: {P_row[tmp_maskpos[idx_tmp]]}, col: {P_col[tmp_maskpos[idx_tmp]]}, masked_pos[{idx_tmp}]: {tmp_maskpos[idx_tmp]}")
            # ax[c].set_xticks([])
            # ax[c].set_yticks([])
            # c+=1

            # A = logits_lm[0,idx_tmp,:].detach().cpu().numpy().flatten()
            # B = masked_tokens[0,idx_tmp,:].detach().cpu().numpy().flatten()
            # r2_sq = (np.corrcoef(A, B, rowvar=False)[0,1])**2


            # ax[c].imshow(logits_lm[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]).detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            # ax[c].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
            # ax[c].set_xticks([])
            # ax[c].set_yticks([])
            # c+=1
        # plt.show()
    # print(aaa)
    if save_path is not None: 
    #     plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
        if mode=='inference':
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        elif val_loss is None:
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"patch-samples-epoch-{epoch:05d}-{mode}-loss-{val_loss:.6f}-{mode}-r2-{val_r2:.2f}.png"))
    plt.close('all')
    
def plot_predictions_in_between(data,T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
    # if logits_lm_copy.dim()==1:
    #     logits_lm_copy = logits_lm_copy[None,:]
        # masked_tokens = masked_tokens[None,:]
        
#     if  masked_tokens.dim()==1: # In this case, we are plotting the whole frame instead of patches
#         fig, ax = plt.subplots(1,2, figsize=(10,5))
#         idx_tmp =0
#         tmp_maskpos = masked_pos.detach().cpu().numpy().flatten()
#         ax[0].imshow(masked_tokens.reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
#         ax[0].set_title(f"masked_tokens[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}, x: {P_row[tmp_maskpos[idx_tmp]]}, y: {P_col[tmp_maskpos[idx_tmp]]}")
#         ax[0].set_xticks([])
#         ax[0].set_yticks([])

#         ax[1].imshow(logits_lm.reshape(patch_size,patch_size).detach().cpu(), vmin=-0.5, vmax=0.5)
#         r2_sq = (np.corrcoef(logits_lm.detach().cpu().numpy().flatten(), masked_tokens.detach().cpu().numpy().flatten(), rowvar=False)[0,1])**2
#         ax[1].set_title(f"logits_lm[{idx_tmp}], r2: {r2_sq:.2f}")
#         ax[1].set_xticks([])
#         ax[1].set_yticks([])
            
#     else:
    # print('logits_lm_copy.shape: ',logits_lm_copy.shape)
    if logits_lm_copy.shape[-1]==1 : 
        logits_lm_copy = np.transpose(logits_lm_copy, (0, 2, 1, 3))
    # print('logits_lm_copy.shape: ',logits_lm_copy.shape)
    # print('data.shape: ',data.shape)
        
    tmp_size = np.minimum(logits_lm_copy.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    # print('tmp_size: ',tmp_size)
    if tmp_size > 2:
        n_rows = int(np.ceil(np.sqrt(tmp_size)))
        n_cols = int(np.floor(np.sqrt(tmp_size)))
        # print('n_rows, n_cols: ', n_rows, n_cols)
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w')
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w')
        n_rows,n_cols=2,1
    tmp_size = np.minimum(tmp_size, n_rows*n_cols) 

    ax = ax.ravel()
    c=0
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the dummy frames
    pred_idx = masked_pos_real[0]
    
    # print('dummy_idxs: ',dummy_idxs)
    # print('real_idxs: ',real_idxs)
    # print('pred_idx: ',pred_idx)
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()]
    # print('real_idxs2.size: ',real_idxs2.size)
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        # print('Replacing reconstructed')
        # print('real_idxs2: ',real_idxs2)
        # print('T[real_idxs2]: ',T[real_idxs2])
        for idx_tmp in real_idxs2:
            # print('logits_lm[0,{}] = data[0,{},:]'.format(idx_tmp,int(T[idx_tmp])))
            if logits_lm_copy.shape[-1]==1 : 
                logits_lm_copy[0,:,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
            else:     
                logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else: 
        fig.suptitle('Reconst. Frames', fontsize=16)
    # else:
        # print('Not Replacing reconstructed')
    # print('logits_lm.shape: ',logits_lm.shape)
            
#     for idx_tmp in range(masked_tokens.shape[1]-1, masked_tokens.shape[1]-tmp_size-1,-1):
    # list_idx = np.sort(np.random.choice(np.arange(masked_pos.shape[1]), tmp_size, replace=True))
    # for idx_tmp in list_idx:
    if logits_lm_copy.shape[-1]==1 : 
        for idx_tmp in range(tmp_size): #masked_pos.shape[1]):
            ax[c].scatter(T, logits_lm_copy[0,idx_tmp,:])
            ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
           
    else:
        for idx_tmp in range(tmp_size): #masked_pos.shape[1]):
            ax[c].imshow(logits_lm_copy[0,idx_tmp,:].reshape(patch_size[0],patch_size[1]), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
        # plt.show()
    
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-epoch-{epoch:05d}_TrueFrames.png"))
        else:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    
# #     # Plot the histogram of the real idxs
#     logits_lm_np = logits_lm_copy[0].copy()
#     fig, ax = plt.subplots()

#     # the histogram of the data
#     ax.hist(logits_lm_np.flatten(), 100, density=True)
    # plt.show()
    # if save_path is not None: 
    #     if replace_by_real_frame:
    #         plt.savefig(os.path.join(save_path, "plots", f"in-between-samples_hist-epoch-{epoch:05d}_TrueFrames.png"))
    #     else:
    #         plt.savefig(os.path.join(save_path, "plots", f"in-between-samples_hist-epoch-{epoch:05d}_ReconsFrames.png"))
    #     plt.close('all')
        
    # return logits_lm_np[real_idxs2,:]
    
def plot_predictions_in_between_pca(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    pca = PCA(n_components=2)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]

    # tmp_size = np.minimum(masked_pos.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    # if tmp_size > 2:
    #     n_rows = int(np.ceil(np.sqrt(tmp_size)))
    #     n_cols = int(np.floor(np.sqrt(tmp_size)))
    #     # print('n_rows, n_cols: ', n_rows, n_cols)
    #     fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50))
    # else:
    fig, ax = plt.subplots(2,2, figsize=(10,10),facecolor='w')

    ax = ax.ravel()
    # c=0
    # print('masked_pos_real: ',masked_pos_real)
    # print('masked_pos: ',masked_pos)
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    # print('logits_lm[0].shape: ',logits_lm[0].shape)
    
    # print('logits_lm_pca.shape: ',logits_lm_pca.shape)
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the real frames
    pred_idx = masked_pos_real[0]
    
    # print('dummy_idxs: ',dummy_idxs)
    # print('real_idxs: ',real_idxs)
    # print('pred_idx: ',pred_idx)
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()] #returns the indexes of the real frames that were not used for prediction
    # print('real_idxs2.size: ',real_idxs2.size)
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        # print('Replacing reconstructed')
        # print('real_idxs2: ',real_idxs2)
        # print('T[real_idxs2]: ',T[real_idxs2])
        for idx_tmp in real_idxs2:
            # print('logits_lm[0,{}] = data[0,{},:]'.format(idx_tmp,int(T[idx_tmp])))
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else:
        fig.suptitle('Reconst. Frames', fontsize=16)
    # else:
        # print('not replacing reconstructed')
        
    logits_lm_pca = pca.fit_transform(logits_lm_copy[0].reshape(logits_lm[0].shape[0],-1))

    # for idx_tmp in range(len(logits_lm_pca)): #masked_pos.shape[1]):
    #     color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
    #     # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    #     if int(T[idx_tmp])==T[idx_tmp]: ax[0].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], label='Data', c='red', alpha=0.5)
    #     else: ax[0].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], label='Dummy', c='blue', alpha=0.5)
    #     # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    #     
    ax[0].scatter(logits_lm_pca[dummy_idxs,0], logits_lm_pca[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[0].scatter(logits_lm_pca[real_idxs2,0], logits_lm_pca[real_idxs2,1], label='Data', c='green')
    ax[0].scatter(logits_lm_pca[pred_idx,0], logits_lm_pca[pred_idx,1], label='Pred', c='red')
    ax[0].set_title('All points')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].legend()
        # c+=1
    
    # for idx_tmp in range(len(logits_lm_pca)): #masked_pos.shape[1]):
        # color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
        # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    ax[2].scatter(logits_lm_pca[:,0], logits_lm_pca[:,1], c=T, alpha=0.5)
    # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    ax[2].set_title('All points')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    # plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca-all-points_epoch-{epoch:05d}.png"))
    
    #Now plot just the predicted frames and the dummy ones
    # print('T: ',T)
    
    # print('dummy_idxs: ',dummy_idxs)
    all_idxs = np.concatenate((dummy_idxs,masked_pos_real[0]))
    # print('all_idxs: ',all_idxs)
    
    # for idx_tmp in all_idxs: #masked_pos.shape[1]):
    #     color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
    #     # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    #     if int(T[idx_tmp])==T[idx_tmp]: ax[1].scatter(logits_lm_pca[idx_tmp,1], logits_lm_pca[idx_tmp,0], label='Data', c='red', alpha=0.5)
    #     else: ax[1].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,1], label='Dummy', c='blue', alpha=0.5)
    #     # ax[1].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], c=color_point, alpha=0.5)
    #     # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    ax[1].scatter(logits_lm_pca[dummy_idxs,0], logits_lm_pca[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[1].scatter(logits_lm_pca[masked_pos_real[0],0], logits_lm_pca[masked_pos_real[0],1], label='Data', c='red')
    ax[1].set_title('Dummy and Pred')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].legend()
        # c+=1

    ax[3].scatter(logits_lm_pca[all_idxs,0], logits_lm_pca[all_idxs,1], c=T[all_idxs], alpha=0.5)
    # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    ax[3].set_title('Dummy and Pred')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    # plt.show()
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    
    del pca, logits_lm_pca, all_idxs
    
def plot_predictions_in_between_pca_vs_time(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, num_components, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    if num_components is not None:
        pca = PCA(n_components=num_components)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]

    tmp_size = np.minimum(num_components, patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
    if tmp_size > 2:
        n_rows = int(np.ceil(np.sqrt(tmp_size)))
        n_cols = int(np.floor(np.sqrt(tmp_size)))
        # print('n_rows, n_cols: ', n_rows, n_cols)
        fig, ax = plt.subplots(n_rows,n_cols, figsize=(20,20), facecolor='w', sharex=True)
    else:
        fig, ax = plt.subplots(6,6, figsize=(20,20),facecolor='w',sharex=True)

    ax = ax.ravel()
    # c=0
    # print('masked_pos_real: ',masked_pos_real)
    # print('masked_pos: ',masked_pos)
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    # print('logits_lm[0].shape: ',logits_lm[0].shape)
    
    # print('logits_lm_pca.shape: ',logits_lm_pca.shape)
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the real frames
    pred_idx = masked_pos_real[0]
    
    # print('dummy_idxs: ',dummy_idxs)
    # print('real_idxs: ',real_idxs)
    # print('pred_idx: ',pred_idx)
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()] #returns the indexes of the real frames that were not used for prediction
    # print('real_idxs2.size: ',real_idxs2.size)
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        fig.suptitle('With True Frames', fontsize=16)
        # print('Replacing reconstructed')
        # print('real_idxs2: ',real_idxs2)
        # print('T[real_idxs2]: ',T[real_idxs2])
        for idx_tmp in real_idxs2:
            # print('logits_lm[0,{}] = data[0,{},:]'.format(idx_tmp,int(T[idx_tmp])))
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    else:
        fig.suptitle('Reconst. Frames', fontsize=16)
    # else:
        # print('not replacing reconstructed')
    if num_components is not None:
        logits_lm_pca = pca.fit_transform(logits_lm_copy[0].reshape(logits_lm[0].shape[0],-1))

    # for idx_tmp in range(len(logits_lm_pca)): #masked_pos.shape[1]):
    #     color_point = 'r' if int(T[idx_tmp])==T[idx_tmp] else 'b'
    #     # print('T[idx_tmp]:{}, color_point: {}'.format(T[idx_tmp], color_point))
    #     if int(T[idx_tmp])==T[idx_tmp]: ax[0].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], label='Data', c='red', alpha=0.5)
    #     else: ax[0].scatter(logits_lm_pca[idx_tmp,0], logits_lm_pca[idx_tmp,0], label='Dummy', c='blue', alpha=0.5)
    #     # ax[c].set_title(f"logits_lm[{idx_tmp}], t: {T[tmp_maskpos[idx_tmp]]}")
    for idx in range(tmp_size):
    #     
        ax[idx].scatter(T[dummy_idxs], logits_lm_pca[dummy_idxs,idx], label='Dummy', c='blue', alpha=0.5)
        ax[idx].scatter(T[real_idxs2], logits_lm_pca[real_idxs2,idx], label='Data', c='green')
        ax[idx].scatter(T[pred_idx], logits_lm_pca[pred_idx,idx], label='Pred', c='red')
        ax[idx].set_title('PC' + str(idx))
        ax[idx].set_xlabel('T')
        # ax[0].set_xticks([])
        # ax[0].set_yticks([])
    # ax[0].legend()
        # c+=1

    # plt.show()
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_vs_time_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-pca_vs_time_epoch-{epoch:05d}_ReconsFrames.png"))
        plt.close('all')
    fig.tight_layout()
    
    del pca, logits_lm_pca
    
def plot_predictions_in_between_umap(data, T, P_row, P_col, masked_pos, masked_pos_real, logits_lm, patch_size,
                     epoch, val_loss, val_r2, save_path, replace_by_real_frame=False):
    # block displaying plots during training
    plt.ioff()
    # pca = PCA(n_components=2)
    reducer = umap.UMAP(n_components=2)
    
    logits_lm_copy = logits_lm.detach().cpu().numpy()#.copy()

    patches_to_plot = 36 
    # if logits_lm_copy.dim()==1:
    #     logits_lm_copy = logits_lm_copy[None,:]

    fig, ax = plt.subplots(2,2, figsize=(10,10),facecolor='w')

    ax = ax.ravel()
    tmp_maskpos = masked_pos[0,:].detach().cpu().numpy().flatten()
    
    dummy_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=True)).flatten() #returns the indexes of the dummy frames
    real_idxs = np.argwhere(np.isin(T.numpy(),np.arange(data.shape[1]),invert=False)).flatten() #returns the indexes of the dummy frames
    pred_idx = masked_pos_real[0]
    
    real_idxs2 = real_idxs[np.argwhere(np.isin(real_idxs,pred_idx,invert=True)).flatten()]
    # print('real_idxs2.size: ',real_idxs2.size)
    if real_idxs2.size == 0: real_idxs2 = pred_idx # In this case, all the frames are being predicted
    
    if replace_by_real_frame:
        for idx_tmp in real_idxs2:
            logits_lm_copy[0,idx_tmp] = data[0,int(T[idx_tmp]),:].detach().cpu().numpy()
    # else:
    #     print('Not replacing reconstructed')
        
    logits_lm_umap = reducer.fit_transform(logits_lm_copy[0].reshape(logits_lm_copy[0].shape[0],-1))
   
    ax[0].scatter(logits_lm_umap[dummy_idxs,0], logits_lm_umap[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[0].scatter(logits_lm_umap[real_idxs2,0], logits_lm_umap[real_idxs2,1], label='Data', c='green')
    ax[0].scatter(logits_lm_umap[pred_idx,0], logits_lm_umap[pred_idx,1], label='Pred', c='red')
    ax[0].set_title('All points')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].legend()

    ax[2].scatter(logits_lm_umap[:,0], logits_lm_umap[:,1], c=T, alpha=0.5)
    ax[2].set_title('All points')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    all_idxs = np.concatenate((dummy_idxs,masked_pos_real[0]))

    ax[1].scatter(logits_lm_umap[dummy_idxs,0], logits_lm_umap[dummy_idxs,1], label='Dummy', c='blue', alpha=0.5)
    ax[1].scatter(logits_lm_umap[masked_pos_real[0],0], logits_lm_umap[masked_pos_real[0],1], label='Data', c='red')
    ax[1].set_title('Dummy and Pred')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].legend()

    ax[3].scatter(logits_lm_umap[all_idxs,0], logits_lm_umap[all_idxs,1], c=T[all_idxs], alpha=0.5)
    ax[3].set_title('Dummy and Pred')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    # plt.show()
        
    if save_path is not None: 
        if replace_by_real_frame:
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-umap_epoch-{epoch:05d}_TrueFrames.png"))
        else: 
            plt.savefig(os.path.join(save_path, "plots", f"in-between-samples-umap_epoch-{epoch:05d}_ReconsFrame.png"))
        plt.close('all')
        
def plot_whole_sequence(data,epoch, range_imshow, save_path): ## (data, self.current_epoch, self.logger.log_dir)
    # print('data.shape: ',data.shape)
    # block displaying plots during training
    plt.ioff()

    patches_to_plot = 32 
    # if logits_lm.dim()==1:
    #     logits_lm = logits_lm[None,:]
        # masked_tokens = masked_tokens[None,:]
    
    if data.shape[-1]==1: #In this case, juts plot as a time series
        tmp_size = np.minimum(data.shape[2], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if tmp_size > 2:
            n_rows = int(np.ceil(np.sqrt(tmp_size)))
            n_cols = int(np.floor(np.sqrt(tmp_size)))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w',sharex=True)
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w',sharex=True)
    else: 
        tmp_size = np.minimum(data.shape[1], patches_to_plot) #Plotting 16 so we can reconstruct a whole frame
        if tmp_size > 2:
            n_rows = int(np.ceil(np.sqrt(tmp_size)))
            n_cols = int(np.floor(np.sqrt(tmp_size)))
            fig, ax = plt.subplots(n_rows,n_cols, figsize=(50,50), facecolor='w',sharex=True)
        else:
            fig, ax = plt.subplots(1,2, figsize=(10,5), facecolor='w',sharex=True)

    ax = ax.ravel()
    c=0
    tmp_maskpos = data[0,:].detach().cpu().numpy().flatten()

    if data.shape[-1]==1: #In this case, juts plot as a time series
        for idx_tmp in range(data.shape[2]):
            ax[c].scatter(np.arange(len(data[0,:,idx_tmp,:])), data[0,:,idx_tmp,:].detach().cpu())
            ax[c].set_title(f"data[{idx_tmp}]")#, t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1
        
    else: 
        for idx_tmp in range(data.shape[1]):
            ax[c].imshow(data[0,idx_tmp,:].detach().cpu(), vmin=range_imshow[0], vmax=range_imshow[1])
            ax[c].set_title(f"data[{idx_tmp}]")#, t: {T[tmp_maskpos[idx_tmp]]}")
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            c+=1

    if save_path is not None: 
        plt.savefig(os.path.join(save_path, "plots", f"whole-seq-samples-epoch-{epoch:05d}.png"))
        plt.close('all')
    
def plot_whole_frame(T, P_row, P_col, masked_tokens, masked_pos, logits_lm, patch_size,frame_size,
                     epoch, val_loss, val_r2, range_imshow, save_path):
    
    fig,ax = plt.subplots(1,2, facecolor='w',figsize=(10,6))
    
    tmp_img1 = torch.zeros(frame_size["rows"],frame_size["cols"])
    count=-int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size[0],steps=int(frame_size["rows"]/patch_size[0]),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size[1],steps=int(frame_size["cols"]/patch_size[1]), dtype=torch.int64):
            tmp_img1[idx_patch1:idx_patch1+patch_size[0], idx_patch2:idx_patch2+patch_size[1]] = logits_lm[0,count,:].reshape(patch_size[0],patch_size[1])
            count+=1

    ax[1].imshow(tmp_img1,vmin=range_imshow[0], vmax=range_imshow[1])
    count=-int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))

    ax[1].set_title('Reconstructed (Frame: {})'.format(T[-1]))
    
    tmp_img = torch.zeros(frame_size["rows"],frame_size["cols"])
    count= -int((frame_size["rows"]/patch_size[0])*(frame_size["cols"]/patch_size[1]))
    for idx_patch1 in torch.linspace(0,frame_size["rows"]-patch_size[0],steps=int(frame_size["rows"]/patch_size[0]),dtype=torch.int64):
        for idx_patch2 in torch.linspace(0,frame_size["cols"]-patch_size[1],steps=int(frame_size["cols"]/patch_size[1]), dtype=torch.int64):
            tmp_img[idx_patch1:idx_patch1+patch_size[0], idx_patch2:idx_patch2+patch_size[1]] = masked_tokens[0,count,:].reshape(patch_size[0],patch_size[1])
            count+=1

    ax[0].imshow(tmp_img, vmin=range_imshow[0], vmax=range_imshow[1])
    ax[0].set_title('Original')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    r_square_frames = (np.corrcoef(tmp_img1.flatten(), 
                                   tmp_img.flatten(), rowvar=False)[0,1])**2
    fig.suptitle('R2 = {:.4f}'.format(r_square_frames))
#     plt.show()

    if save_path is not None: plt.savefig(os.path.join(save_path, "plots", f"whole_frame-epoch-{epoch:05d}-val-loss-{val_loss:.6f}-val-r2-{r_square_frames:.3f}.png"))
#     plt.close('all')
    
    reconstrcuted_frame = tmp_img1.clone()
    
    return reconstrcuted_frame



def create_dataloaders(path_file, experiment_name, use_first_n_frames, frame_size=64, margem_to_crop = [32,40,24,24], 
    fast_start=True, show_plots=False, batch_size_segments=10, segment_size = 32, behavior_variable=None, verbose=True, args=None):
    # everything from bellow in one cell
    # Defining the dataset to be used
    dataSource = 'mouse' # '2dWave' is a toy dataset
    use_SimCLR=False
    orig_video=True
    window = 50 # Number of frames in the MMD window 
    loadCNMF = False # if true, load the factorization results from the cNMF method
    add_time_to_origVideo = True # If true and orig_video==True, then add time as channels in the original video.
    label_last_frame = False # if True, the last frame of the MMD window is used as the label for the window (the idea of make future predictions based on the window information)
    use_SetTransformer = False # if True, uses the Set Transformer to encode the windows

    if args.load_preprocessed:
        if verbose: print('Loading ',os.path.join('./datasets',experiment_name + '_preprocessed_' + str(args.use_first_n_frames) + '.p'))
        # path_to_preprocessed = torch.load(os.path.join(os.path.join('./datasets',experiment_name + '_preprocessed.pt')))
        # with open('file.pkl', 'rb') as file:
        with open(os.path.join('./datasets',experiment_name + '_preprocessed_' + str(args.use_first_n_frames) + '.p'), 'rb') as file:
      
            # Call load method to deserialze
            path_to_preprocessed = pickle.load(file)

        data = path_to_preprocessed['data']
        labels1 = path_to_preprocessed['labels']
    else: 
        if dataSource == 'mouse': # To load the Ca imaging video
    #         path_file = os.path.join(path_file, experiment_name)
            output_path = os.path.join(path_file,experiment_name, "output")
            output_preprocessed = os.path.join(path_file,experiment_name, "output_preprocessed")
            # experiment = '11232019_grabAM07_vis_stim'
            fullpath = os.path.join(path_file,experiment_name, "final_dFoF.mat")
    #         print(f"stim full path: {fullpath}")

        #     path_file = "/home/antonio2/data/2pmeso/imaging_with_575_excitation/Cardin/GRABS_Data_March/Analyzed_SVDMethodPatch14/DualMice"
        #     output_path = "/home/antonio2/data/2pmeso/output"
        #     output_preprocessed = '/home/antonio2/data/2pmeso/preprocessed'
        #     experiment = '11222019_grabAM05_spont'
        #     fullpath = os.path.join(path_file,experiment,"final_dFoF.mat")
        #     #print(fullpath)
        #     fast_start=True #True if we just want to load the saved variables (much faster)
        #     orig_video = False # True to work with original video. False to work with WFT
        #     single_freq = False # True to perform classification using single bandwidths from the fourier for test
        #     add_time_to_origVideo = True # If true, add time as channels in the original video.
        #     use_SimCLR=False

        if not fast_start:
            if verbose: print('Loading ',fullpath)
            f2 = h5py.File(fullpath,'r')
            #print(f2.keys())
            x = f2["dFoF"]
            #print(list(x))
            video = np.array(x['green'])
            if verbose: print('video loaded shape: ',video.shape)
                
            frame_width = int(np.sqrt(video.shape[1]))

            video = np.transpose(video.reshape(video.shape[0],frame_width, frame_width),(0,2,1)) #Turn video in frames, width, height
            if frame_size != video.shape[2]: 
                video = resize(video.T, (frame_size, frame_size, video.shape[0])).T
                if verbose: print('new video shape: ',video.shape)
                    
    #         data3 = video.reshape(1,video.shape[0],video.shape[1],video.shape[2]) # Original
            data3 = video.reshape(video.shape[0],video.shape[1],video.shape[2]) # Original
            if use_first_n_frames is not None:
                print('Using the first ' + str(use_first_n_frames) + ' frames')
                data3 = data3[:use_first_n_frames,:, :]
    #         data3 = np.transpose(data3,(0,1,3,2))
            if verbose: print('video original shape: ',data3.shape)
            
            data3 = data3[:,margem_to_crop[0]:video.shape[1]-margem_to_crop[1],margem_to_crop[2]:video.shape[2]-margem_to_crop[3]]
            if verbose: print('video after crop shape: ',data3.shape)
            data3[np.isnan(data3)] = 0
            
            tmp_range_imshow = np.array([np.quantile(data3.flatten(), 0.01), np.quantile(data3.flatten(), 0.99)])
            print('tmp_range_imshow: ',tmp_range_imshow)
            
            
                    
    #         video2 = np.resize(video[:,:,:], ((video.shape[0], frame_size, frame_size)))
        if not fast_start and show_plots:
            fig,ax = plt.subplots()
            ax.imshow(video[1000,:,:], cmap='viridis_r',vmin=tmp_range_imshow[0], vmax=tmp_range_imshow[1])
            ax.set_title('Sample frame: 1000')
            plt.show()

            # #print('video: ',video)
            #print('data3.shape: ',data3.shape)
            #print('data3.min(): ',data3.min())
            fig,ax = plt.subplots()
            ax.imshow(data3[1000,:,:], cmap='viridis_r',vmin=tmp_range_imshow[0], vmax=tmp_range_imshow[1])
            ax.set_title('Replacing NaN by zeros')
            
            # Create a Rectangle patch
            rect = patches.Rectangle((20, 40), 50, 50, linewidth=1,
                                     edgecolor='r', facecolor="none")

            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.show()
            print('Rectangle for reference: xy:{},{}, width: {}, height: {}'.format(20,40,50,50))
            
            fig,ax = plt.subplots()
    #         ax.imshow(data3[0,1000,40:40+50, 20:20+50], cmap='viridis_r')
            ax.imshow(data3[1000,40:40+50, 20:20+50], cmap='viridis_r',vmin=tmp_range_imshow[0], vmax=tmp_range_imshow[1])
            ax.set_title('data3[0,1000,40:40+50, 20:20+50]')
            plt.show()
            
        

    #         np.save(output_preprocessed + "/video_orig.npy", data3)
        '''
        Loading the dataset (fast start mode)
        '''
        if dataSource == 'mouse':
            if fast_start==True:
                if orig_video==False:
                    data3 = np.load(os.path.join(output_path,experiment_name,"output/video_WFT.npy"),allow_pickle=True)
                else:
                    data3 = np.load(os.path.join(path_file,experiment_name,"output/video_orig.npy"),allow_pickle=True)
    #             vis_stim_all = np.load(os.path.join(path_file,experiment_name,"output/vis_stim_all.npy"))
    #             time = np.load(os.path.join(path_file,experiment_name,"output/time.npy"))
    #             bins = np.load(os.path.join(path_file,experiment_name,"output/bins.npy"))
    #             freqs = np.load(os.path.join(path_file,experiment_name,"output/freqs.npy"))
                #print('data3.shape: {}'.format(data3.shape))
                #print(f"vis_stim_all.shape: {vis_stim_all.shape}")
                #print('bins.shape: {}'.format(bins.shape))
                #print('freqs.shape: {}'.format(freqs.shape))


        if dataSource == '2dWave':
            #print('fast_start: {}, orig_video: {}'.format(fast_start, orig_video))
            if fast_start==True and orig_video==False:
                P3 = np.load(os.path.join(output_path,experiment_name,"output/freq_decomp_" + video_name + "64.npy"),allow_pickle=True)
            #     vis_stim_all = np.load(os.path.join(output_path,experiment,"output/vis_stim_all.npy"))
            #     time = np.load(os.path.join(output_path,experiment,"output/time.npy"))
                bins = np.load(os.path.join(output_path,experiment_name,"output/bins_" + video_name + "64.npy"))
                freqs = np.load(os.path.join(output_path,experiment_name,"output/freqs_" + video_name + "64.npy"))
                #print('P3.shape: {}'.format(P3.shape))
                #print('bins.shape: {}'.format(bins.shape))
                #print('freqs.shape: {}'.format(freqs.shape))
        '''
        Loading behavior variables for the mouse dataset
        '''

        if dataSource == 'mouse' and behavior_variable is not None:
            from scipy.io import loadmat
            spike2_vars = loadmat(os.path.join(path_file,experiment_name,behavior_variable+".mat"))

            vis_stim = spike2_vars['timestamps']['contrasts_bin_100'][0][0] * 100
            vis_stim2 = spike2_vars['timestamps']['contrasts_bin_50'][0][0] *50 #This is already the visual stim matching frames
            vis_stim3 = spike2_vars['timestamps']['contrasts_bin_20'][0][0] * 20
            vis_stim4 = spike2_vars['timestamps']['contrasts_bin_10'][0][0] * 10
            vis_stim5 = spike2_vars['timestamps']['contrasts_bin_5'][0][0] * 5
            vis_stim6 = spike2_vars['timestamps']['contrasts_bin_2'][0][0] * 2
            vis_stim_all = vis_stim + vis_stim2 + vis_stim3 + vis_stim4 + vis_stim5 + vis_stim6
            print('vis_stim_all.shape: ',vis_stim_all.shape)
            time = spike2_vars['timestamps']['timaging'][0][0]
            print('time.shape: ',time.shape)
            fs_imaging = round(time[1][0]-time[0][0],2)
            print('fs_imaging: ',fs_imaging)

            if show_plots:
                fig, ax = plt.subplots(1,1,figsize=(16, 4), dpi= 100, facecolor='w', edgecolor='k')
                fig.suptitle('Stimulus')
                tmp1 = vis_stim_all[:1000]
                tmp2 = time[:1000]
                ax.plot(time[:1000], vis_stim_all[:1000])
                ax.set_xticks(np.arange(60, max(time[:1000])+1, 5.0))

            # Behavior variables
            allwhellon = spike2_vars['timing']['allwheelon'][0][0]
            allwhelloff = spike2_vars['timing']['allwheeloff'][0][0]
            allwheel = np.concatenate((allwhellon,allwhelloff), axis=1)

            allwheel_analog = spike2_vars['channels_data']['wheelspeed'][0][0].squeeze()
            allwheel_analog_time = np.arange(0,allwheel_analog.shape[0])
            allwheel_analog_time = allwheel_analog_time/5000 #Sampled at 5kHz
            print('[1] allwheel_analog_time.shape: {}'.format(allwheel_analog_time.shape))
            fs_wheel = allwheel_analog_time[1]-allwheel_analog_time[0]
            print('[1] fs_wheel = ', fs_wheel)
            ratio_imaging_wheel = int(fs_imaging/fs_wheel)
            print('ratio_imaging_wheel: ',ratio_imaging_wheel)
            allwheel_analog = allwheel_analog[::ratio_imaging_wheel]
            allwheel_analog_time = allwheel_analog_time[::ratio_imaging_wheel]
            allwheel_analog = allwheel_analog[np.where((allwheel_analog_time>=time[0]) & (allwheel_analog_time<=time[-1]))]
            allwheel_analog_time = allwheel_analog_time[np.where((allwheel_analog_time>=time[0]) & (allwheel_analog_time<=time[-1]))]
            print('allwheel_analog.shape: {}'.format(allwheel_analog.shape))
            print('allwheel_analog_time.shape: {}'.format(allwheel_analog_time.shape))
            print('allwheel_analog_time: {},{}',allwheel_analog_time[0], allwheel_analog_time[-1])
            print('time: ',time[0],time[-1])
            allwheel_time = np.arange(time[0],time[-1],0.1)
            allwheel_tmp = np.zeros_like(allwheel_time)

            for i in range(0,len(allwheel)):
                allwheel_tmp[(allwheel_time>allwheel[i,0]) & (allwheel_time<allwheel[i,1])]=1
            allwheel = allwheel_tmp[:]
            if show_plots:
                fig, (ax,ax1) = plt.subplots(2,1,figsize=(16, 4), dpi= 100, facecolor='w', edgecolor='k')
                fig.suptitle('Wheel')
                ax.plot(allwheel_time[:1000], allwheel[:1000])
                print('allwheel.shape: {}'.format(allwheel.shape))

                ax1.plot(allwheel_analog_time[:1000], allwheel_analog[:1000])
                ax.set_xticks(np.arange(60, max(time[:1000])+1, 5.0))

            #Make sure both arrays have the same size
            if np.abs(time[0]-allwheel_analog_time[0])<np.abs(time[-1]-allwheel_analog_time[-1]):
                time = time[:len(allwheel_analog_time)]
                vis_stim_all = vis_stim_all[:len(allwheel_analog_time)]
    #             data3 =  data3[:,:len(allwheel_analog_time),:,:]
                data3 =  data3[:len(allwheel_analog_time),:,:]
            else:
                time = time[len(allwheel_analog_time):]   
                vis_stim_all = vis_stim_all[len(allwheel_analog_time):]
    #             data3 =  data3[:,len(allwheel_analog_time):,:,:]
                data3 =  data3[len(allwheel_analog_time):,:,:]

            print('allwheel_analog_time: {},{}, {}'.format(allwheel_analog_time[0], allwheel_analog_time[-1],allwheel_analog_time.shape))
            print('time: {},{}, {}'.format(time[0],time[-1],time.shape))
            print('data3.shape: ',data3.shape)
            
    #     # Replace every frame in the sequence by the mean frame
    #     mean_frame = np.nanmean(data3, axis=1).squeeze()
    #     print('\nmean_frame.shape: ',mean_frame.shape)
    #     data3 = np.repeat(mean_frame[None,...],data3.shape[1],axis=0)
    #     del mean_frame
    #     data3 = data3[None,:]
    #     print('[after replace by mean frame] data3.shape: ',data3.shape, data3.min(), data3.max())
            
    #     '''
    #     Matching sampling of behavior variables and video
    #     '''

    #     if dataSource == 'mouse':
    #         #print(bins.shape)
    #         #print(vis_stim_all.shape)
    #         #print(time.shape)
    #         #print('time[0]: {}, time[-1]: {}'.format(time[0], time[-1]))
    #         time2 = time - time[0]
    #         #print('time2: {}'.format(time2))
    #         #print(bins)

    #         bin_idx = []
    #         for i in bins:
    #             idx = np.argmin(abs(i-time2[:]))
    #             bin_idx.append(idx)
    #         #print('len(bin_idx): {}'.format(len(bin_idx)))
    #         time_down = time2[bin_idx[:]]
    #         #print(time_down[:10])

    #     if dataSource == '2dWave':
    #         time = np.arange(0,duration,1/fps)
    #         #print('bins.shape: {}'.format(bins.shape))
    #         #print('time.shape: {}'.format(time.shape))
    #         #print('time[0]: {}, time[-1]: {}'.format(time[0], time[-1]))
    #         time2 = time - time[0]
    #         #print('time2: {}'.format(time2))
    #         #print('bins: {}'.format(bins))

    #         bin_idx = []
    #         for i in bins:
    #             idx = np.argmin(abs(i-time2[:]))
    #             bin_idx.append(idx)
    #         #print('len(bin_idx): {}'.format(len(bin_idx)))
    #         time_down = time2[bin_idx[:]]


        '''
        Normalization
        '''

        if not loadCNMF:
            if use_SimCLR==True or use_SetTransformer:
                data_normalized = np.zeros((data3.shape[1], data3.shape[0], data3.shape[2], data3.shape[3]))

                for i in range(0, data_normalized.shape[0]): #Replace this by permute(1,0,2,3). Much simpler
                    data_normalized[i,:,:,:] = data3[:,i,:,:]
                #print('data_normalized.shape: {}'.format(data_normalized.shape))
                # dataset = MyDataset(data, labels)

            else: #Use this option to transform the 4D to 2D 
                if orig_video:
                    #print('using orig_video')
                    orig_shape = data3.shape
    #                 data3[0,:,:,:] = data3[0,:,:,:]/np.nanstd(data3[0,:,:,:])  
    #                 data3[0,:,:,:] = ((data3[0,:,:,:] - np.nanmin(data3[0,:,:,:]))/(np.nanmax(data3[0,:,:,:]) - np.nanmin(data3[0,:,:,:])))*2-1
                    data3[:,:,:] = data3[:,:,:]/np.nanstd(data3[:,:,:])  
                    data3[:,:,:] = ((data3[:,:,:] - np.nanmin(data3[:,:,:]))/(np.nanmax(data3[:,:,:]) - np.nanmin(data3[:,:,:])))*2-1

    #                 data_normalized = np.zeros((data3.shape[1], data3.shape[0] * data3.shape[2] * data3.shape[3]))
    #                 for i in range(0, data_normalized.shape[0]): 
    #                     data_normalized[i,:] = data3[:,i,:,:].reshape(1,data3.shape[0] * data3.shape[2] * data3.shape[3]) #Flattening everything

    #                 data_normalized = data3.copy()
                else:
                    data_normalized = np.zeros((data3.shape[1], data3.shape[0] * data3.shape[2] * data3.shape[3]))
                    for i in range(0, data_normalized.shape[0]): 
                        data_normalized[i,:] = data3[:,i,:,:].reshape(1,data3.shape[0] * data3.shape[2] * data3.shape[3]) #Flattening everything
                if verbose: print('data3.shape: {}'.format(data3.shape))
                    
        
        
        '''
        Normalize labels as well
        '''
        if behavior_variable is not None: 
            #     labels_tmp = allwheel_analog[bin_idx[:]]
            # labels = vis_stim_all[bin_idx[:]]
        #     del data3
            labels1 = vis_stim_all
            labels2 = allwheel_analog
            labels1 = np.sqrt(labels1)
            # labels = allwheel[bin_idx[:]]
        #      labels = ((labels - labels.min())/(labels.max() - labels.min()))*2-1
            labels1 = ((labels1 - labels1.min())/(labels1.max() - labels1.min()))*2-1

        #     labels = {"vis_stim": labels1, "analog_wheel": labels2}
            # data_normalized = ((data_normalized - data_normalized.min())/(data_normalized.max() - data_normalized.min()))*2-1
            print('data3.shape: {}'.format(data3.shape))
            print('data3.min: {}'.format(data3.min()))
            print('data3.max: {}'.format(data3.max()))
            print('allwheel.shape: {}'.format(allwheel.shape))
            # print('vis_stim_all[bin_idx[:]].shape: {}'.format(vis_stim_all[bin_idx[:]].shape))
            print('vis_stim_all.shape: {}'.format(vis_stim_all.shape))

            if show_plots:
                print('\nplotting frame sample...')
                fig, (ax2, ax3) = plt.subplots(2,1,figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
                if orig_video:
        #             print('np.nanmax(data_normalized,axis=1),np.nanmin(data_normalized,axis=1): ',np.nanmax(data_normalized,axis=1),np.nanmin(data_normalized,axis=1))
        #             #Original
        #             ax2.plot(np.nanmax(data_normalized,axis=1))
        #             ax3.plot(np.nanmin(data_normalized,axis=1))

                    ax2.plot(data3[:,:,:].max(axis=(1,2)))
                    ax3.plot(data3[:,:,:].min(axis=(1,2)))

                else:
                    ax2.plot(data3[:,:,:,:].max(axis=(0,2,3)))
                    ax3.plot(data3[:,:,:,:].min(axis=(0,2,3)))
        else:
            labels1 = np.ones(len(data3))

        if orig_video:
            print('\ncopy data_normalized')
        #     data2 = data3[:,bin_idx[:],:,:] #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
            data = data3.copy() #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
            del data3
            print('finished copying')
        #     data = data_normalized[:,:] #For the 18D version, this operation is done in the concatenation step. Since there is no concat for orig_video, it has to be done here
        else:
            #print('Just copying data3')
            data2 = data3.copy()  
            del data3
            # data = data2.reshape(data2.shape[1], data2.shape[0], data2.shape[2], data2.shape[3]) #data2.shape: (17, 20477, 64, 64)
            data = np.zeros((data2.shape[1], data2.shape[0], data2.shape[2], data2.shape[3]))

            for i in range(0, data.shape[0]): #Replace this by permute(1,0,2,3). Much simpler
                data[i,:,:,:] = data2[:,i,:,:]
            del data2

        # Save pre-processed data to speed process
        if args.save_preprocessed:
            outputs_to_save = {
                    'data': data,
                    'labels': labels1,
            }
            # torch.save(outputs_to_save, os.path.join('./datasets',experiment_name + '_preprocessed.pt'))
            # pickle.dump(d, open("file", 'w'), protocol=4)
            # pickle.dump(outputs_to_save, open(os.path.join('./datasets',experiment_name + '_preprocessed.pt'), 'w'), protocol=4)
            with open(os.path.join('./datasets',experiment_name + '_preprocessed_' + str(args.use_first_n_frames) + '.p'), 'wb') as file:
      
                # A new file will be created
                pickle.dump(outputs_to_save, file)

    class MyDataset(Dataset):
        def __init__(self, data, times, target, compression_factor, segment_size):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(target).float()
            self.segment_size = segment_size
            self.times = times.float()

        def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
            x = self.data[index:index+self.segment_size]
            # y = self.target[index:index+self.segment_size]
            t = self.times
            return x, t

        def __len__(self):
            return len(self.data)

        
    class MyDataset_test(Dataset):
        def __init__(self, data, target1, target2, compression_factor, segment_size):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = torch.from_numpy(data).float()
            self.target1 = torch.from_numpy(target1).float()
            self.target2 = torch.from_numpy(target2).float()
            self.segment_size = segment_size

        def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
#             x = self.data[1000:2000]
#             y1 = self.target1[1000:2000]
#             y2 = self.target2[1000:2000]

            x = self.data[:10000]
            y1 = self.target1[:10000]
            y2 = self.target2[:10000]
            return x, y1, y2

        def __len__(self):
            return len(self.data)
    #print('max_labels {}, min {}'.format(np.amax(labels),np.amin(labels)))

    # t_max = 1 #frames.shape[0]
    print('data.shape: ',data.shape)
    # t_max=1
    t_max=segment_size-1
    t_min=0
    n_points = segment_size

    times = torch.linspace(t_min, t_max, n_points) #Original
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    # times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    # times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    # times = times.flatten()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    compression_factor=1
    range_imshow = np.array([np.quantile(data.flatten(), 0.01), np.quantile(data.flatten(), 0.99)])
    print('range_imshow: ',range_imshow)
        
    dataset = MyDataset(data, times, labels1, compression_factor, segment_size)
    print('created dataset')
    dataset_size  = len(dataset)
    print('dataset_size: {}'.format(dataset_size))
    validation_split=args.validation_split

    # Number of frames in the sequence (in this case, same as number of tokens). Maybe I can make this number much bigger, like 4 times bigger, and then do the batches of batches...
    # For example, when classifying, I can test if the first and the second chunk are sequence vs the first and third
    # big_batch_size=300 # Number of frames in the minibatch [initially]
    #batch_size=32 #256 #How many frames are actually going to be selected
    #REMINDER: The batch size still needs to be adjusted with the same value in train_mrpc.json and bert_base.json files

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    #print('train/val split: {}'.format(split))
    # np.random.shuffle(indices) # Randomizing the indices is not a good idea if you want to model the sequence
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices, val_indices = train_indices[:-segment_size], val_indices[:-segment_size] #remove the indices at the end, since we want continuous windows of length segment_size
    print('train_indices[:50]: {}'.format(train_indices[:50]))
    print('val_indices[:50]: {}'.format(val_indices[:50]))
    print('len(val_indices): ',len(val_indices))

    # -- create dataloaders
    if args.mode=='train':
        #Original
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(indices[0])

    #     # Use sequential instead of random to make sure things are still in the right order
    #     train_sampler = SequentialSampler(train_indices)
    #     valid_sampler = SequentialSampler(val_indices)

        # # To make batch of batches, in which the minibatches are sequential
        # train_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_indices),batch_size=3, drop_last=True)
        # valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_indices),batch_size=3, drop_last=True)
        
        # dataset_test = MyDataset_test(data, labels1, labels2, compression_factor, segment_size)
        
        num_workers = 6
        dataloaders   = {
            'train': DataLoader(dataset, batch_size=batch_size_segments, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(dataset, batch_size=batch_size_segments, sampler=valid_sampler, num_workers=num_workers),
            # 'test': DataLoader(dataset_test,  batch_size=batch_size_segments, shuffle=False, num_workers=num_workers),
            }
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(val_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(dataset, batch_size=batch_size_segments, sampler=valid_sampler, num_workers=num_workers),
            }

# #     # Check if normalization worked
#     dataiter = iter(dataloaders['test'])
#     images_tmp, labels_tmp = dataiter.next()

#     print('Dim of sample batch sample: {}\n'.format(images_tmp.shape))
#     print('Dim of labels: {}\n'.format(labels_tmp.shape))
#     #print('max {}, min {}'.format(torch.max(images_tmp),torch.min(images_tmp)))
#     #print('max_labels {}, min {}'.format(torch.max(labels_tmp),torch.min(labels_tmp)))

#     fig, ax = plt.subplots(1,5,figsize=(20,5),facecolor='w', edgecolor='k')
#     ax =ax.ravel()
#     for idx in range(5):
#         ax[idx].plot(labels_tmp[idx,:].squeeze())
#     plt.show()
# #     plt.imshow(images_tmp[1].reshape((64,64)))
# #     plt.show()

    other_vars ={'range_imshow': range_imshow}
    
    return dataloaders, other_vars


def create_dataloaders_toydata(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points,args):
    
    print('Loading ',os.path.join(path_file,experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_orig'][:,:use_first_n_frames,:, None]

    if args.add_noise_to_input_data>0:
        for idx_curve in range(Data.shape[0]):
            noise_0 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            noise_1 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            Data[idx_curve,:,0,0] = Data[idx_curve,:,0,0]+noise_0
            Data[idx_curve,:,1,0] = Data[idx_curve,:,1,0]+noise_1

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])

    scaling_factor = np.quantile(np.abs(Data),0.98)
    print('scaling_factor: ',scaling_factor)

    # args.range_imshow = np.array([np.quantile(Data.flatten(), 0.4), np.quantile(Data.flatten(), 0.55)])#np.array([-0.25,0.05]) #
    # print('args.range_imshow: ',args.range_imshow)
    # args.fitted_pca = Data_dict['pca']
    # Data = to_np(Data[:,:4]) #This might be necessary in some cases. Not sure why some of these variables were saved as CUDA.

    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 #number of iterations for training. default=3k epochs
    # segment_len = args.segment_len

    Data_test = Data[train_val:,:]
    # Data = Data[:args.use_first_n_frames,:] #Data[:train_val,:]

    n_points = Data.shape[1]
    extrapolation_points = Data.shape[1]

    # t_max = 1 #frames.shape[0]
    t_max=1
    t_min=0

    index_np = np.arange(0, len(Data), 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=n_points) #Original
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    times = times.flatten()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2],1])
        tmp_times = np.zeros([1,downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:, :] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :]
        times = times[ids_downsampled[idx_for_downsample,:]]

        Data = tmp_data.copy()
        # times = tmp_times.copy()

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)
    print('times.shape: ',times.shape)
    print('Data_test.shape: ',Data_test.shape)

    # scaling_factor = to_np(Data).max()
    # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
    Data = Data/scaling_factor
    Data_test = Data_test/scaling_factor

    # Data = torch.from_numpy(Data).to(args.device)
    Data = torch.Tensor(Data)#.double()
    Data_test = torch.Tensor(Data_test)#.double()
    # times = torch.Tensor(times)#.double()

    #Original Dataset setup 
    # Data_splitting_indices = Train_val_split3(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
    # validation_split=0.3
    Data_splitting_indices = Train_val_split(np.copy(index_np),validation_split) #Just the first 100 are used for training and validation
    Train_Data_indices = Data_splitting_indices.train_IDs()
    Val_Data_indices = Data_splitting_indices.val_IDs()

    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    print('frames_to_drop.shape: ',frames_to_drop.shape)
    print('frames_to_drop: ',frames_to_drop)

    # frames_to_drop = np.random.randint(randomly_drop_n_last_frames+1, size=len(Data))
    print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
    print('Train_Data_indices: ',Train_Data_indices)
    print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
    print('Val_Data_indices: ',Val_Data_indices)
    # print('frames_to_drop [for train]: ',frames_to_drop[Train_Data_indices])
    # print('frames_to_drop [for val]: ',frames_to_drop[Val_Data_indices])
    # # #Define frames to drop
    # if args.randomly_drop_n_last_frames is not None:
    #     args.randomly_drop_n_last_frames = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    # print('args.randomly_drop_n_last_frames; ',args.randomly_drop_n_last_frames)

    # Dataset = Dynamics_Dataset2(Data,times,args.segment_len,args.segment_window_factor, frames_to_drop)
    Dataset_mine = Dynamics_Dataset3(Data,times,frames_to_drop)
    # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
    # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

    # times_np_test = np.linspace(t_min, t_max, num=Data_test.shape[0])
    # times_np_test = np.hstack([times_np_test[:, None]])
    # times_test = torch.from_numpy(times_np_test[:, :, None])#.to(args.device)
    # times_test = times_test.flatten()
    # Dataset_all = Test_Dynamics_Dataset(Data,times)

    # For the sampler
    train_sampler = SubsetRandomSampler(Train_Data_indices)
    valid_sampler = SubsetRandomSampler(Val_Data_indices)

    # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

    dataloaders = {'train': torch.utils.data.DataLoader(Dataset_mine, sampler=train_sampler, batch_size = batch_size_segments,
                                                        num_workers=6, drop_last=False),
                   'val': torch.utils.data.DataLoader(Dataset_mine, sampler=valid_sampler, batch_size = batch_size_segments, 
                                                       num_workers=6, drop_last=False),
                   # 'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(times),  num_workers=args.num_workers)
                  }
    # print('dataloaders: ',dataloaders)
    
    return dataloaders

# def create_dataloaders_toydata3(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points,args):
#     # This version returns the whole data as well. This will be used to compute the interpolation loss

    
#     print('Loading ',os.path.join(path_file,experiment_name + ".p"))
#     Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
#     print(Data_dict.keys())
#     Data = Data_dict['Data_orig'][:,:use_first_n_frames,:, None]

#     if args.add_noise_to_input_data>0:
#         for idx_curve in range(Data.shape[0]):
#             noise_0 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
#             noise_1 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
#             Data[idx_curve,:,0,0] = Data[idx_curve,:,0,0]+noise_0
#             Data[idx_curve,:,1,0] = Data[idx_curve,:,1,0]+noise_1

#     Data_orig = Data.copy()
#     # time_seq = times/t_max
#     # print('time_seq: ',time_seq)
#     print('Data.shape: ',Data.shape)

#     if downsample_points < Data.shape[1]: #check if there is a need for downsampling
#         if regularly_sampled:
#             # print('Data.shape: ',Data.shape)
#             # print('args.downsample_points: ',args.downsample_points)
#             ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
#         else: 
#             random_idxs = np.zeros([Data.shape[0],9])
#             for idx_tmp in range(Data.shape[0]):
#                 random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)
#             # print('random_idxs: ',random_idxs)

#             ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
#             # print('ids_downsampled: ',ids_downsampled)

#             ids_downsampled = np.sort(ids_downsampled)
#         print('ids_downsampled[0]: ',ids_downsampled[0])

#     scaling_factor = np.quantile(np.abs(Data),0.98)
#     print('scaling_factor: ',scaling_factor)

#     # args.range_imshow = np.array([np.quantile(Data.flatten(), 0.4), np.quantile(Data.flatten(), 0.55)])#np.array([-0.25,0.05]) #
#     # print('args.range_imshow: ',args.range_imshow)
#     # args.fitted_pca = Data_dict['pca']
#     # Data = to_np(Data[:,:4]) #This might be necessary in some cases. Not sure why some of these variables were saved as CUDA.

#     train_val = 20000 # Number of frames for train and validation. The remaining will be for test
#     n_steps = 3000 #number of iterations for training. default=3k epochs
#     # segment_len = args.segment_len

#     Data_test = Data[train_val:,:]
#     # Data = Data[:args.use_first_n_frames,:] #Data[:train_val,:]

#     n_points = Data.shape[1]
#     extrapolation_points = Data.shape[1]

#     # t_max = 1 #frames.shape[0]
#     t_max=1
#     t_min=0

#     index_np = np.arange(0, len(Data), 1, dtype=int)
#     index_np = np.hstack(index_np[:, None])
#     times_np = np.linspace(t_min, t_max, num=n_points) #Original
#     # times_np = np.linspace(t_min, t_max, num=args.segment_len)
#     times_np = np.hstack([times_np[:, None]])
#     # print('times_np: ',times_np)

#     ###########################################################
#     times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
#     times = times.flatten()
#     times_orig = times.clone()
    
#     print('times.shape: ',times.shape)
#     print('times: ',times)

#     if downsample_points < Data.shape[1]: #check if there is a need for downsampling
#         tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2],1])
#         tmp_times = np.zeros([1,downsample_points])
#         print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
#         print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


#         for idx_for_downsample in range(Data.shape[0]):
#             # print('idx_for_downsample: ',idx_for_downsample)
#             # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
#             # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
#             # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
#             # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
#             tmp_data[idx_for_downsample,:,:, :] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :]
#         times = times[ids_downsampled[idx_for_downsample,:]]

#         Data = tmp_data.copy()
#         # times = tmp_times.copy()

#     # time_seq = times/t_max
#     # print('time_seq: ',time_seq)
#     print('Data.shape: ',Data.shape)
#     print('times.shape: ',times.shape)
#     print('Data_test.shape: ',Data_test.shape)

#     # scaling_factor = to_np(Data).max()
#     # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
#     Data = Data/scaling_factor
#     Data_orig = Data_orig/scaling_factor
#     Data_test = Data_test/scaling_factor

#     # Data = torch.from_numpy(Data).to(args.device)
#     Data = torch.Tensor(Data)#.double()
#     Data_orig = torch.Tensor(Data_orig)
#     Data_test = torch.Tensor(Data_test)#.double()
#     # times = torch.Tensor(times)#.double()

#     #Original Dataset setup 
#     # Data_splitting_indices = Train_val_split3(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
#     # validation_split=0.3
#     Data_splitting_indices = Train_val_split(np.copy(index_np),validation_split) #Just the first 100 are used for training and validation
#     Train_Data_indices = Data_splitting_indices.train_IDs()
#     Val_Data_indices = Data_splitting_indices.val_IDs()

#     if args.randomly_drop_n_last_frames is not None:
#         frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
#     elif args.drop_n_last_frames is not None:
#         frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
#     elif args.num_points_for_c is not None:
#         args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
#         frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
#     print('frames_to_drop.shape: ',frames_to_drop.shape)
#     print('frames_to_drop: ',frames_to_drop)

#     # frames_to_drop = np.random.randint(randomly_drop_n_last_frames+1, size=len(Data))
#     print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
#     print('Train_Data_indices: ',Train_Data_indices)
#     print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
#     print('Val_Data_indices: ',Val_Data_indices)
#     # print('frames_to_drop [for train]: ',frames_to_drop[Train_Data_indices])
#     # print('frames_to_drop [for val]: ',frames_to_drop[Val_Data_indices])
#     # # #Define frames to drop
#     # if args.randomly_drop_n_last_frames is not None:
#     #     args.randomly_drop_n_last_frames = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
#     # print('args.randomly_drop_n_last_frames; ',args.randomly_drop_n_last_frames)

#     # Dataset = Dynamics_Dataset2(Data,times,args.segment_len,args.segment_window_factor, frames_to_drop)
#     Dataset_mine = Dynamics_Dataset4(Data,Data_orig,times,times_orig, frames_to_drop)
#     # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
#     # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

#     # times_np_test = np.linspace(t_min, t_max, num=Data_test.shape[0])
#     # times_np_test = np.hstack([times_np_test[:, None]])
#     # times_test = torch.from_numpy(times_np_test[:, :, None])#.to(args.device)
#     # times_test = times_test.flatten()
#     # Dataset_all = Test_Dynamics_Dataset(Data,times)

#     # For the sampler
#     train_sampler = SubsetRandomSampler(Train_Data_indices)
#     valid_sampler = SubsetRandomSampler(Val_Data_indices)

#     # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

#     dataloaders = {'train': torch.utils.data.DataLoader(Dataset_mine, sampler=train_sampler, batch_size = batch_size_segments,
#                                                         num_workers=6, drop_last=False),
#                    'val': torch.utils.data.DataLoader(Dataset_mine, sampler=valid_sampler, batch_size = batch_size_segments, 
#                                                        num_workers=6, drop_last=False),
#                    # 'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(times),  num_workers=args.num_workers)
#                   }
#     # print('dataloaders: ',dataloaders)
    
#     return dataloaders


def create_dataloaders_toydata3(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points,args):
    # This version returns the whole data as well. This will be used to compute the interpolation loss

    
    print('Loading ',os.path.join(path_file,experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_orig'][:,:use_first_n_frames,:, None]

    if args.add_noise_to_input_data>0:
        for idx_curve in range(Data.shape[0]):
            noise_0 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            noise_1 = np.random.normal(0,args.add_noise_to_input_data,Data.shape[1])
            Data[idx_curve,:,0,0] = Data[idx_curve,:,0,0]+noise_0
            Data[idx_curve,:,1,0] = Data[idx_curve,:,1,0]+noise_1

    Data_orig = Data.copy()
    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            if args.T_to_sample is not None:
                print('len(args.T_to_sample): ',len(args.T_to_sample))
                print('args.downsample_points: ',args.downsample_points)
                if args.downsample_points == len(args.T_to_sample): #In this case, just use the given coordinates and downsample anything
                    ids_downsampled = np.tile(args.T_to_sample,(Data.shape[0],1))
            else: 
                ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])

    scaling_factor = np.quantile(np.abs(Data),0.98)
    print('scaling_factor: ',scaling_factor)

    # args.range_imshow = np.array([np.quantile(Data.flatten(), 0.4), np.quantile(Data.flatten(), 0.55)])#np.array([-0.25,0.05]) #
    # print('args.range_imshow: ',args.range_imshow)
    # args.fitted_pca = Data_dict['pca']
    # Data = to_np(Data[:,:4]) #This might be necessary in some cases. Not sure why some of these variables were saved as CUDA.

    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 #number of iterations for training. default=3k epochs
    # segment_len = args.segment_len

    Data_test = Data[train_val:,:]
    # Data = Data[:args.use_first_n_frames,:] #Data[:train_val,:]

    n_points = Data.shape[1]
    extrapolation_points = Data.shape[1]

    # t_max = 1 #frames.shape[0]
    t_max=1
    t_min=0

    index_np = np.arange(0, len(Data), 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=n_points) #Original
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    times = times.flatten()
    times_orig = times.clone()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2],1])
        tmp_times = np.zeros([1,downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:, :] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :]
        times = times[ids_downsampled[idx_for_downsample,:]]

        Data = tmp_data.copy()
        # times = tmp_times.copy()

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)
    print('times.shape: ',times.shape)
    print('Data_test.shape: ',Data_test.shape)

    # scaling_factor = to_np(Data).max()
    # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
    Data = Data/scaling_factor
    Data_orig = Data_orig/scaling_factor
    Data_test = Data_test/scaling_factor

    # Data = torch.from_numpy(Data).to(args.device)
    Data = torch.Tensor(Data)#.double()
    Data_orig = torch.Tensor(Data_orig)
    Data_test = torch.Tensor(Data_test)#.double()
    # times = torch.Tensor(times)#.double()

    #Original Dataset setup 
    # Data_splitting_indices = Train_val_split3(np.copy(index_np),args.validation_split, args.segment_len,args.segment_window_factor) #Just the first 100 are used for training and validation
    # validation_split=0.3
    Data_splitting_indices = Train_val_split(np.copy(index_np),validation_split) #Just the first 100 are used for training and validation
    Train_Data_indices = Data_splitting_indices.train_IDs()
    Val_Data_indices = Data_splitting_indices.val_IDs()

    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    print('frames_to_drop.shape: ',frames_to_drop.shape)
    print('frames_to_drop: ',frames_to_drop)

    # frames_to_drop = np.random.randint(randomly_drop_n_last_frames+1, size=len(Data))
    print('\nlen(Train_Data_indices): ',len(Train_Data_indices))
    print('Train_Data_indices: ',Train_Data_indices)
    print('\nlen(Val_Data_indices): ',len(Val_Data_indices))
    print('Val_Data_indices: ',Val_Data_indices)
    # print('frames_to_drop [for train]: ',frames_to_drop[Train_Data_indices])
    # print('frames_to_drop [for val]: ',frames_to_drop[Val_Data_indices])
    # # #Define frames to drop
    # if args.randomly_drop_n_last_frames is not None:
    #     args.randomly_drop_n_last_frames = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    # print('args.randomly_drop_n_last_frames; ',args.randomly_drop_n_last_frames)

    # Dataset = Dynamics_Dataset2(Data,times,args.segment_len,args.segment_window_factor, frames_to_drop)
    Dataset_mine = Dynamics_Dataset4(Data,Data_orig,times,times_orig, frames_to_drop)
    # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
    # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

    # times_np_test = np.linspace(t_min, t_max, num=Data_test.shape[0])
    # times_np_test = np.hstack([times_np_test[:, None]])
    # times_test = torch.from_numpy(times_np_test[:, :, None])#.to(args.device)
    # times_test = times_test.flatten()
    # Dataset_all = Test_Dynamics_Dataset(Data,times)

    # For the sampler
    if args.mode=='train':
        train_sampler = SubsetRandomSampler(Train_Data_indices)
        valid_sampler = SubsetRandomSampler(Val_Data_indices)

        # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

        dataloaders = {'train': torch.utils.data.DataLoader(Dataset_mine, sampler=train_sampler, batch_size = batch_size_segments,
                                                            num_workers=args.num_workers, drop_last=False),
                       'val': torch.utils.data.DataLoader(Dataset_mine, sampler=valid_sampler, batch_size = batch_size_segments, 
                                                           num_workers=args.num_workers, drop_last=False),
                       # 'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(times),  num_workers=args.num_workers)
                      }
        # print('dataloaders: ',dataloaders)
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(Val_Data_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(Dataset_mine, batch_size=batch_size_segments, sampler=valid_sampler, num_workers=args.num_workers),
            }
    
    return dataloaders


def create_dataloaders_toydata2(path_file, experiment_name, use_first_n_frames, batch_size_segments, validation_split, regularly_sampled, downsample_points, args):
    
    # class Train_val_split:
    #     def __init__(self, IDs,val_size_fraction):
            
            
    #         IDs = np.random.permutation(IDs)
    #         # print('IDs: ',IDs)
    #         self.IDs = IDs
    #         self.val_size = int(val_size_fraction*len(IDs))
        
    #     def train_IDs(self):
    #         train = sorted(self.IDs[:len(self.IDs)-self.val_size])
    #         # print('len(train): ',len(train))
    #         # print('train: ',train)
    #         return train
        
    #     def val_IDs(self):
    #         val = sorted(self.IDs[len(self.IDs)-self.val_size:])
    #         # print('len(val): ',len(val))
    #         # print('val: ',val)
    #         return val
        
    # class Dynamics_Dataset3(Dataset):
    #     'Characterizes a dataset for PyTorch'
    #     def __init__(self, Data, times, frames_to_drop, segment_len):
    #         'Initialization'
    #         self.times = times.float()
    #         self.Data = Data.float()
    #         self.frames_to_drop = frames_to_drop
    #         self.segment_len=segment_len
    #         # self.batch_size = batch_size

    #     def __getitem__(self, index):
    #         # print('index: ',index)
    #         # print('self.list_IDs.shape: ',len(self.list_IDs))
    #         # print('self.Data: ',self.Data)
    #         # print('self.times: ', self.times)
    #         ID = index #self.list_IDs[index]
    #         obs = self.Data[ID,:self.segment_len]
    #         t = self.times #Because it already set the number of points in the main script
    #         frames_to_drop = self.frames_to_drop[index]

    #         return obs, t, ID, frames_to_drop 
        
    #     def __len__(self):
    #         'Denotes the total number of points'
    #         return len(self.times)
    
    print('Loading ',os.path.join(path_file,experiment_name + ".p"))
    Data_dict = pickle.load(open(os.path.join(path_file,experiment_name + ".p"), "rb" )) #This data was saved in GPU. So transform it to CPU first
    print(Data_dict.keys())
    Data = Data_dict['Data_orig'][:,:use_first_n_frames,:]

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])

    scaling_factor = np.quantile(np.abs(Data),0.98)
    print('scaling_factor: ',scaling_factor)

    # args.range_imshow = np.array([np.quantile(Data.flatten(), 0.4), np.quantile(Data.flatten(), 0.55)])#np.array([-0.25,0.05]) #
    # print('args.range_imshow: ',args.range_imshow)
    # args.fitted_pca = Data_dict['pca']
    # Data = to_np(Data[:,:4]) #This might be necessary in some cases. Not sure why some of these variables were saved as CUDA.

    train_val = 20000 # Number of frames for train and validation. The remaining will be for test
    n_steps = 3000 #number of iterations for training. default=3k epochs
    # segment_len = args.segment_len

    Data_test = Data[train_val:,:]
    # Data = Data[:args.use_first_n_frames,:] #Data[:train_val,:]

    n_points = Data.shape[1]
    extrapolation_points = Data.shape[1]

    # t_max = 1 #frames.shape[0]
    t_max=1
    t_min=0

    index_np = np.arange(0, len(Data), 1, dtype=int)
    index_np = np.hstack(index_np[:, None])
    times_np = np.linspace(t_min, t_max, num=n_points) #Original
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    times = times.flatten()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    if downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = np.zeros([Data.shape[0],downsample_points,Data.shape[2]])
        tmp_times = np.zeros([1,downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :]
        times = times[ids_downsampled[idx_for_downsample,:]]

        Data = tmp_data.copy()
        # times = tmp_times.copy()

    # time_seq = times/t_max
    # print('time_seq: ',time_seq)
    print('Data.shape: ',Data.shape)
    print('times.shape: ',times.shape)
    print('Data_test.shape: ',Data_test.shape)

    # scaling_factor = to_np(Data).max()
    # args.scaling_factor = np.quantile(Data.flatten(), 0.99) #Data.max()
    Data = Data/scaling_factor
    Data_test = Data_test/scaling_factor

    # Data = torch.from_numpy(Data).to(args.device)
    Data = torch.Tensor(Data)#.double()
    Data_test = torch.Tensor(Data_test)#.double()
    # times = torch.Tensor(times)#.double()

    Data_splitting_indices = Train_val_split(np.arange(len(Data)),validation_split)
    Train_Data_indices = Data_splitting_indices.train_IDs()
    Val_Data_indices = Data_splitting_indices.val_IDs()
    # frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames+1, size=len(Data))
    # #Define frames to drop
    if args.randomly_drop_n_last_frames is not None:
        frames_to_drop = np.random.randint(args.randomly_drop_n_last_frames, size=len(Val_Data_indices)+len(Train_Data_indices))
    elif args.drop_n_last_frames is not None:
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    elif args.num_points_for_c is not None:
        args.drop_n_last_frames = Data.shape[1]-args.num_points_for_c
        frames_to_drop = np.ones(len(Val_Data_indices)+len(Train_Data_indices),dtype=np.int8) * args.drop_n_last_frames
    print('frames_to_drop.shape: ',frames_to_drop.shape)
    print('frames_to_drop: ',frames_to_drop)

    # Dataset = Dynamics_Dataset2(Data,times,args.segment_len,args.segment_window_factor, frames_to_drop)
    # Dataset_mine = Dynamics_Dataset3(Data,times)#,frames_to_drop)
    Dataset = Dynamics_Dataset3(Data,times,frames_to_drop)
    # loader = torch.utils.data.DataLoader(Dataset, batch_size = batch_size)
    # Dataset_val = Val_Dynamics_Dataset(Data,Val_Data_indices,times)

    # times_np_test = np.linspace(t_min, t_max, num=Data_test.shape[0])
    # times_np_test = np.hstack([times_np_test[:, None]])
    # times_test = torch.from_numpy(times_np_test[:, :, None])#.to(args.device)
    # times_test = times_test.flatten()
    # Dataset_all = Test_Dynamics_Dataset(Data,times)

    # For the sampler
    train_sampler = SubsetRandomSampler(Train_Data_indices)
    valid_sampler = SubsetRandomSampler(Val_Data_indices)

    # loader_val = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size)

    dataloaders = {'train': torch.utils.data.DataLoader(Dataset, sampler=train_sampler, batch_size = batch_size_segments,
                                                        num_workers=6, drop_last=False),
                   'val': torch.utils.data.DataLoader(Dataset, sampler=valid_sampler, batch_size = batch_size_segments, 
                                                       num_workers=6, drop_last=False),
                   # 'test': torch.utils.data.DataLoader(Dataset_all, batch_size = len(times),  num_workers=args.num_workers)
                  }
    
    return dataloaders



def create_dataloaders_Navier_Stokes(Data, experiment_name, frame_size=64, margem_to_crop = [32,40,24,24], 
    fast_start=True, show_plots=False, batch_size_segments=8, segment_size = 50, behavior_variable=None, verbose=True, args=None):
    # everything from bellow in one cell
    # Defining the dataset to be used
    dataSource = 'mouse' # '2dWave' is a toy dataset
    use_SimCLR=False
    orig_video=True
    window = 50 # Number of frames in the MMD window 
    loadCNMF = False # if true, load the factorization results from the cNMF method
    add_time_to_origVideo = True # If true and orig_video==True, then add time as channels in the original video.
    label_last_frame = False # if True, the last frame of the MMD window is used as the label for the window (the idea of make future predictions based on the window information)
    use_SetTransformer = False # if True, uses the Set Transformer to encode the windows
    segment_size = segment_size
    compression_factor = 1

    data = Data
    #labels1 = path_to_preprocessed['labels']
    
    print('data.shape: ',data.shape)
    # t_max=1
    t_max=1
    t_min=0
    n_points = segment_size
    
    times = torch.linspace(t_min, t_max, n_points) #Original
    times_all = times
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if args.regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=args.downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=args.downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = torch.zeros([Data.shape[0],args.downsample_points,Data.shape[2],Data.shape[3]])
        tmp_times = torch.zeros([1,args.downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:],...]
        times = times[ids_downsampled[idx_for_downsample,:]]

        data = tmp_data.clone()#.float()
        # times = tmp_times.copy()
        
        print("Downsampled data: ",data.shape)
    
    
    class MyDataset(Dataset):
        def __init__(self, data, times, compression_factor):

            if compression_factor != 1:
                #print(data.shape)
                data = zoom(data, (1,1,compression_factor, compression_factor))
                #print(data.shape)

            self.data = data#torch.from_numpy(data).float()
            #self.target = torch.from_numpy(target).float()
            #self.segment_size = segment_size
            self.times = times.float()

        def __getitem__(self, index):
#             print('index [MyDataSet]: ', index)
            x = self.data[index,...]
            # y = self.target[index:index+self.segment_size]
            t = self.times
            return x, t

        def __len__(self):
            return len(self.data)

    
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    # times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    # times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    # times = times.flatten()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    compression_factor=1
    range_imshow = np.array([np.quantile(data.flatten(), 0.01), np.quantile(data.flatten(), 0.99)])
    print('range_imshow: ',range_imshow)
    
    dataset = MyDataset(data, times, compression_factor)
    print('created dataset')
    dataset_size  = len(dataset)
    print('dataset_size: {}'.format(dataset_size))
    validation_split=args.validation_split

    # Number of frames in the sequence (in this case, same as number of tokens). Maybe I can make this number much bigger, like 4 times bigger, and then do the batches of batches...
    # For example, when classifying, I can test if the first and the second chunk are sequence vs the first and third
    # big_batch_size=300 # Number of frames in the minibatch [initially]
    #batch_size=32 #256 #How many frames are actually going to be selected
    #REMINDER: The batch size still needs to be adjusted with the same value in train_mrpc.json and bert_base.json files

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    #print('train/val split: {}'.format(split))
    # np.random.shuffle(indices) # Randomizing the indices is not a good idea if you want to model the sequence
    train_indices, val_indices = indices[split:], indices[:split]
    
    #train_indices, val_indices = train_indices[:-segment_size], val_indices[:-segment_size] #remove the indices at the end, since we want continuous windows of length segment_size
    print('train_indices[:50]: {}'.format(train_indices[:50]))
    print('val_indices[:50]: {}'.format(val_indices[:50]))
    print('len(val_indices): ',len(val_indices))

    # -- create dataloaders
    if args.mode=='train':
        #Original
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(indices[0])

    #     # Use sequential instead of random to make sure things are still in the right order
    #     train_sampler = SequentialSampler(train_indices)
    #     valid_sampler = SequentialSampler(val_indices)

        # # To make batch of batches, in which the minibatches are sequential
        # train_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_indices),batch_size=3, drop_last=True)
        # valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_indices),batch_size=3, drop_last=True)
        
        # dataset_test = MyDataset_test(data, labels1, labels2, compression_factor, segment_size)
        
        num_workers = 6
        dataloaders   = {
            'train': DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            # 'test': DataLoader(dataset_test,  batch_size=batch_size_segments, shuffle=False, num_workers=num_workers),
            }
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(val_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            }

# #     # Check if normalization worked
#     dataiter = iter(dataloaders['test'])
#     images_tmp, labels_tmp = dataiter.next()

#     print('Dim of sample batch sample: {}\n'.format(images_tmp.shape))
#     print('Dim of labels: {}\n'.format(labels_tmp.shape))
#     #print('max {}, min {}'.format(torch.max(images_tmp),torch.min(images_tmp)))
#     #print('max_labels {}, min {}'.format(torch.max(labels_tmp),torch.min(labels_tmp)))

#     fig, ax = plt.subplots(1,5,figsize=(20,5),facecolor='w', edgecolor='k')
#     ax =ax.ravel()
#     for idx in range(5):
#         ax[idx].plot(labels_tmp[idx,:].squeeze())
#     plt.show()
# #     plt.imshow(images_tmp[1].reshape((64,64)))
# #     plt.show()

    other_vars ={'range_imshow': range_imshow}
    
    
    return dataloaders, other_vars
    

def create_dataloaders_Navier_Stokes_eval(Data, experiment_name, frame_size=64, margem_to_crop = [32,40,24,24], 
    fast_start=True, show_plots=False, batch_size_segments=8, segment_size = 50, behavior_variable=None, verbose=True, args=None):
    # everything from bellow in one cell
    # Defining the dataset to be used
    dataSource = 'mouse' # '2dWave' is a toy dataset
    use_SimCLR=False
    orig_video=True
    window = 50 # Number of frames in the MMD window 
    loadCNMF = False # if true, load the factorization results from the cNMF method
    add_time_to_origVideo = True # If true and orig_video==True, then add time as channels in the original video.
    label_last_frame = False # if True, the last frame of the MMD window is used as the label for the window (the idea of make future predictions based on the window information)
    use_SetTransformer = False # if True, uses the Set Transformer to encode the windows
    segment_size = segment_size
    compression_factor = 1
    
    if args.add_noise_to_input_data>0:
        for idx_curve in range(Data.shape[0]):
            noise = torch.normal(torch.zeros_like(Data[0,...]),std=args.add_noise_to_input_data)
            Data[idx_curve,...] = Data[idx_curve,...]+noise

    data = Data
    #labels1 = path_to_preprocessed['labels']
    
    print('data.shape: ',data.shape)
    # t_max=1
    t_max=1
    t_min=0
    n_points = segment_size
    
    times = torch.linspace(t_min, t_max, n_points) #Original
    times_all = times
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        if args.regularly_sampled:
            # print('Data.shape: ',Data.shape)
            # print('args.downsample_points: ',args.downsample_points)
            if args.T_to_sample is not None:
                print('len(args.T_to_sample): ',len(args.T_to_sample))
                print('args.downsample_points: ',args.downsample_points)
                if args.downsample_points == len(args.T_to_sample): #In this case, just use the given coordinates and downsample anything
                    ids_downsampled = np.tile(args.T_to_sample,(Data.shape[0],1))
            else: 
                ids_downsampled = np.tile(np.linspace(0,Data.shape[1]-1,num=downsample_points, dtype=np.int64),(Data.shape[0],1))
        else: 
            random_idxs = np.zeros([Data.shape[0],9])
            for idx_tmp in range(Data.shape[0]):
                random_idxs[idx_tmp,:] = np.random.choice(np.arange(1, Data.shape[1],dtype=np.int64), size=args.downsample_points-1, replace=False)
            # print('random_idxs: ',random_idxs)

            ids_downsampled = np.concatenate((np.zeros([Data.shape[0],1],dtype=np.int8),random_idxs),axis=1).astype(int)
            # print('ids_downsampled: ',ids_downsampled)

            ids_downsampled = np.sort(ids_downsampled)
        print('ids_downsampled[0]: ',ids_downsampled[0])
    
    if args.downsample_points < Data.shape[1]: #check if there is a need for downsampling
        tmp_data = torch.zeros([Data.shape[0],args.downsample_points,Data.shape[2],Data.shape[3]])
        tmp_times = torch.zeros([1,args.downsample_points])
        print('tmp_data.shape: ',tmp_data.shape) #tmp_data.shape:  (100, 10, 2, 1)
        print('tmp_times.shape: ',tmp_times.shape) #tmp_times.shape:  (100, 10)


        for idx_for_downsample in range(Data.shape[0]):
            # print('idx_for_downsample: ',idx_for_downsample)
            # print('tmp_data[idx_for_downsample,:,:, :].shape: ',tmp_data[idx_for_downsample,:,:, :].shape)
            # print('Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape: ',Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:], :, :].shape)
            # print('tmp_times[idx_for_downsample,:].shape: ',tmp_times[idx_for_downsample,:].shape)
            # print('times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape: ',times[idx_for_downsample,ids_downsampled[idx_for_downsample,:]].shape)
            tmp_data[idx_for_downsample,:,:] = Data[idx_for_downsample,ids_downsampled[idx_for_downsample,:],...]
        times = times[ids_downsampled[idx_for_downsample,:]]

        data = tmp_data.clone()#.float()
        # times = tmp_times.copy()
        
        print("Downsampled data: ",data.shape)
    
    
    class Dynamics_Dataset_NS_eval(Dataset):
        'Characterizes a dataset for PyTorch'
        def __init__(self, Data, Data_orig, times, times_orig):
            'Initialization'
            self.times = times.float()
            self.times_orig = times_orig.float()
            self.Data = Data.float()
            self.Data_orig = Data_orig.float()
            # self.segment_len=segment_len
            # self.batch_size = batch_size

        def __getitem__(self, index):
            # print('index: ',index)
            # print('self.list_IDs.shape: ',len(self.list_IDs))
            # print('self.Data: ',self.Data)
            # print('self.times: ', self.times)
            ID = index #self.list_IDs[index]
            # obs = self.Data[ID,:self.segment_len]
            obs = self.Data[ID,...]
            obs_orig = self.Data_orig[ID,...]
            t = self.times #Because it already set the number of points in the main script
            t_orig = self.times_orig #Because it already set the number of points in the main script

            return obs, obs_orig, t, t_orig 

        def __len__(self):
            'Denotes the total number of points'
            return len(self.Data)

    
    # times_np = np.linspace(t_min, t_max, num=args.segment_len)
    # times_np = np.hstack([times_np[:, None]])
    # print('times_np: ',times_np)

    ###########################################################
    # times = torch.from_numpy(times_np[:, :, None])#.to(args.device)
    # times = times.flatten()
    
    print('times.shape: ',times.shape)
    print('times: ',times)

    compression_factor=1
    range_imshow = np.array([np.quantile(data.flatten(), 0.01), np.quantile(data.flatten(), 0.99)])
    print('range_imshow: ',range_imshow)
    
    dataset = Dynamics_Dataset_NS_eval(data,Data, times, times_all)
    print('created dataset')
    dataset_size  = len(dataset)
    print('dataset_size: {}'.format(dataset_size))
    validation_split=args.validation_split

    # Number of frames in the sequence (in this case, same as number of tokens). Maybe I can make this number much bigger, like 4 times bigger, and then do the batches of batches...
    # For example, when classifying, I can test if the first and the second chunk are sequence vs the first and third
    # big_batch_size=300 # Number of frames in the minibatch [initially]
    #batch_size=32 #256 #How many frames are actually going to be selected
    #REMINDER: The batch size still needs to be adjusted with the same value in train_mrpc.json and bert_base.json files

    # -- split dataset
    indices       = list(range(dataset_size))
    split         = int(np.floor(validation_split*dataset_size))
    #print('train/val split: {}'.format(split))
    # np.random.shuffle(indices) # Randomizing the indices is not a good idea if you want to model the sequence
    train_indices, val_indices = indices[split:], indices[:split]
    
    #train_indices, val_indices = train_indices[:-segment_size], val_indices[:-segment_size] #remove the indices at the end, since we want continuous windows of length segment_size
    print('train_indices[:50]: {}'.format(train_indices[:50]))
    print('val_indices[:50]: {}'.format(val_indices[:50]))
    print('len(val_indices): ',len(val_indices))

    # -- create dataloaders
    if args.mode=='train':
        #Original
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(indices[0])

    #     # Use sequential instead of random to make sure things are still in the right order
    #     train_sampler = SequentialSampler(train_indices)
    #     valid_sampler = SequentialSampler(val_indices)

        # # To make batch of batches, in which the minibatches are sequential
        # train_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_indices),batch_size=3, drop_last=True)
        # valid_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(val_indices),batch_size=3, drop_last=True)
        
        # dataset_test = MyDataset_test(data, labels1, labels2, compression_factor, segment_size)
        
        num_workers = 6
        dataloaders   = {
            'train': DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers),
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            # 'test': DataLoader(dataset_test,  batch_size=batch_size_segments, shuffle=False, num_workers=num_workers),
            }
    elif args.mode=='inference':
        valid_sampler = SequentialSampler(val_indices)
        num_workers = 6
        dataloaders   = {
            'val': DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=num_workers),
            }

# #     # Check if normalization worked
#     dataiter = iter(dataloaders['test'])
#     images_tmp, labels_tmp = dataiter.next()

#     print('Dim of sample batch sample: {}\n'.format(images_tmp.shape))
#     print('Dim of labels: {}\n'.format(labels_tmp.shape))
#     #print('max {}, min {}'.format(torch.max(images_tmp),torch.min(images_tmp)))
#     #print('max_labels {}, min {}'.format(torch.max(labels_tmp),torch.min(labels_tmp)))

#     fig, ax = plt.subplots(1,5,figsize=(20,5),facecolor='w', edgecolor='k')
#     ax =ax.ravel()
#     for idx in range(5):
#         ax[idx].plot(labels_tmp[idx,:].squeeze())
#     plt.show()
# #     plt.imshow(images_tmp[1].reshape((64,64)))
# #     plt.show()

    other_vars ={'range_imshow': range_imshow}
    
    
    return dataloaders, other_vars
    