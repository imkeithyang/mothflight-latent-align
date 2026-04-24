# this was taken partially from the repository here: https://github.com/jpowie01/DCGAN_CelebA/blob/master/dataset.py 
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import numpy as np
import pickle
import utils
from pathlib import Path

FT_KEEP_AXES = np.array([0, 4, 5], dtype=np.int64)


def select_ft_axes(ft_data):
    return np.asarray(ft_data)[..., FT_KEEP_AXES]

def MinMaxNormalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        # Avoid division by zero if all values are the same
        if min_val == max_val:
            return tensor
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

def mean_max_normalize(data, axis=1, eps=1e-6):
    centered = data - data.mean(axis=axis, keepdims=True)
    scale = np.max(np.abs(centered), axis=axis, keepdims=True)
    scale = np.maximum(scale, eps)
    return centered / scale

def normalize_ft_arrays(ft_data, moth_id=None, mode="global", eps=1e-6):
    if mode == "none":
        return ft_data, ft_data.mean(axis=1)

    normed_ft_data = np.zeros_like(ft_data)
    normed_ft_data_mean = np.zeros_like(ft_data.mean(axis=1))

    if mode == "per_moth":
        if moth_id is None:
            raise ValueError("moth_id is required for per_moth FT normalization")
        for moth in np.unique(moth_id):
            moth_mask = moth_id == moth
            moth_ft = ft_data[moth_mask]
            mean = moth_ft.mean(axis=(0, 1), keepdims=True)
            std = np.maximum(moth_ft.std(axis=(0, 1), keepdims=True), eps)
            normed_ft_data[moth_mask] = (moth_ft - mean) / std

            mean_ft = moth_ft.mean(axis=1)
            mean_ft_mean = mean_ft.mean(axis=0, keepdims=True)
            mean_ft_std = np.maximum(mean_ft.std(axis=0, keepdims=True), eps)
            normed_ft_data_mean[moth_mask] = (mean_ft - mean_ft_mean) / mean_ft_std
        return normed_ft_data, normed_ft_data_mean

    if mode != "global":
        raise ValueError(f"Unknown FT normalization mode: {mode}")

    mean = ft_data.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(ft_data.std(axis=(0, 1), keepdims=True), eps)
    normed_ft_data = (ft_data - mean) / std

    mean_ft = ft_data.mean(axis=1)
    mean_ft_mean = mean_ft.mean(axis=0, keepdims=True)
    mean_ft_std = np.maximum(mean_ft.std(axis=0, keepdims=True), eps)
    normed_ft_data_mean = (mean_ft - mean_ft_mean) / mean_ft_std
    return normed_ft_data, normed_ft_data_mean

def kfold_indices(idx, k):
    # Shuffle
    np.random.seed(1)
    np.random.shuffle(idx)

    # Calculate number per fold
    fold_size = np.repeat(len(idx) // 5, k)
    remainder = len(idx) % k
    for i in range(remainder):
        fold_size[i] += 1

    #split data
    folds = []
    start = 0
    for i in range(k):
        end = start + fold_size[i]
        test_indices = idx[start:end]
        train_indices = np.concatenate([idx[:start], idx[end:]])
        folds.append((train_indices, test_indices))
        start = end
    return folds

def convolve_muscle(data, kernel = None):
    if kernel is None: return data
    convolved_data = np.zeros_like(data)
    for sample_idx in range(data.shape[0]):
        for muscle_idx in range(data.shape[2]):
            convolved_data[sample_idx, :, muscle_idx] = np.convolve(data[sample_idx, :, muscle_idx], kernel, mode='same')
    return convolved_data

def _resolve_position_filepath(filepath):
    filepath = Path(filepath)
    if "fatmoth_filtered" in filepath.parts:
        return Path(str(filepath).replace("fatmoth_filtered", "fatmoth_new"))
    return filepath

def _resolve_filtered_ft_filepath(filepath):
    filepath = Path(filepath)
    if "fatmoth_new" in filepath.parts:
        return Path(str(filepath).replace("fatmoth_new", "fatmoth_filtered"))
    return filepath

def _load_scaled_position(filepath):
    position_path = _resolve_position_filepath(filepath)
    with open(position_path, "rb") as file:
        position_dict = pickle.load(file)
    flower_data = np.asarray(position_dict["flower"])[..., :1] / 100.0
    return flower_data

class MuscleForcesTorque_dataset(Dataset):
    def __init__(self, muscle_data, FT_data, mean_FT_data, flower_data, moth_id, mass_id, 
                 convolution=None, device=torch.device("cpu"), normalize=False, evaluate=False,
                 perturb = False):
        if convolution=="gaussian":
            lin = np.linspace(-100, 100, 201)
            sd = 5
            kernel = (1 / (np.sqrt(2 * np.pi * sd**2))) * np.exp(-((lin)**2) / (2 * sd**2))
            kernel = (kernel -  kernel.min()) / (kernel.max() - kernel.min())
        elif convolution=="truncated_gaussian":
            lin = np.linspace(-7, 7, 15)
            sd = 5
            kernel = (1 / (np.sqrt(2 * np.pi * sd**2))) * np.exp(-((lin)**2) / (2 * sd**2))
        elif convolution=="exponential":
            lin = np.linspace(0, 100, 101)
            lambda_val = 0.05
            kernel = lambda_val * np.exp(-lambda_val * lin)
            kernel = np.concatenate((np.zeros(100), kernel))
        elif convolution=="fixed_interval":
            kernel = np.ones(20)
        elif convolution is None:
            kernel = None
        else: 
            raise NotImplementedError(f"Convolution {convolution} is not implemented!")
        
        self.evaluate = evaluate
        self.convolution = convolution
        self.perturb = perturb
        if perturb:
            muscle_data = self.perturb_spikes(muscle_data)
            
        self.muscle_data_unconvolve = muscle_data
        convolved_muscle_data = convolve_muscle(muscle_data, kernel=kernel)

        self.muscle_data = torch.from_numpy(convolved_muscle_data).float().to(device)
        self.spike_counts_data = torch.sum(torch.from_numpy(muscle_data), dim=-2).to(device)
        # print("Spike counts data shape: ", self.spike_counts_data.shape)
        self.FT_data = torch.from_numpy(FT_data).float().to(device)
        self.FT_mean_data = torch.from_numpy(mean_FT_data).float().to(device) #self.FT_data.mean(dim=1)
        #print(self.FT_mean_data.mean(), self.FT_mean_data.std())
        if normalize:
            # normalize the FT_data
            mi = self.FT_data.min(dim=1, keepdim=True).values
            ma = self.FT_data.max(dim=1, keepdim=True).values
            self.FT_data = (self.FT_data - mi) / (ma - mi)
            self.FT_mean_data = (self.FT_mean_data - mi.squeeze(1)) / (ma.squeeze(1) - mi.squeeze(1))
            # FT_mean = self.FT_data.mean(dim=(0,1), keepdim=True)
            # FT_std = self.FT_data.std(dim=(0,1), keepdim=True)
            # self.FT_data = (self.FT_data - FT_mean) / FT_std
            # self.FT_mean_data = (self.FT_mean_data - FT_mean.squeeze(1)) / FT_std.squeeze(1)
        
        self.flower_data = torch.from_numpy(flower_data).float().to(device)

        # using one_hot without specifying number of classes is dangerous
        # in this case we assume the number of moth in the moth_id is the total number of moth
        self.moth_number = torch.from_numpy(moth_id).to(device)
        self.mass_id = torch.from_numpy(mass_id).float().to(device)

        # self.moth_id = self.moth_id.repeat(1, convolved_muscle_data.shape[1], 1)
        # self.covariate = torch.cat([self.flower_data, self.moth_id], dim=-1)
        self.covariate = self.flower_data
        self.device = device
    
    def perturb_spikes(self, muscle_data):
        # perturbation on ldlm and rdlm, 
        # ldlm goes right for 10 ms
        # rdlm goes left for 10 ms
        ind, ldlm_index = np.where(muscle_data[:,:,4] == 1)
        muscle_data[ind, ldlm_index, 4] = 0
        muscle_data[ind, np.maximum(ldlm_index - 50, 0), 4] = 1
        #ind, rdlm_index = np.where(muscle_data[:,:,5] == 1)
        #muscle_data[ind, rdlm_index, 5] = 0
        #muscle_data[ind, np.maximum(rdlm_index - 50, 0), 5] = 1
        #ind, rdlm_index = np.where(muscle_data[:,:,5] == 1)
        #muscle_data[rdlm_index] = 0
        #muscle_data[ind, np.maximum(rdlm_index - 100, 0)] = 1
        return muscle_data
    
    def get_dataset(self):
        return self.covariate, self.flower_data, self.FT_data, self.FT_mean_data, self.muscle_data, self.spike_counts_data, self.moth_id, self.mass_id

    def __len__(self):
        return self.muscle_data.shape[0]

    def __getitem__(self, idx):
        # make sure convolution 
        if self.evaluate == True:
            sample = (self.covariate[idx],
                  self.FT_data[idx], 
                  self.FT_mean_data[idx],
                  self.muscle_data[idx],
                  self.spike_counts_data[idx],
                  self.muscle_data_unconvolve[idx],
                  self.moth_number[idx],
                  self.mass_id[idx],
                  )
        else:
            sample = (self.covariate[idx],
                  self.FT_data[idx], 
                  self.FT_mean_data[idx],
                  self.muscle_data[idx],
                  self.spike_counts_data[idx],
                  self.moth_number[idx],
                  self.mass_id[idx],
                  )
        
        return sample

def get_fold_MFT(filepath, num_folds, fold_idx, normalize_data=False, batch_size=16, convolution=None, device=torch.device("cpu"), use_filtered_data=True):
    assert(num_folds is not None and num_folds > 0)
    assert(fold_idx >= 0 and fold_idx < num_folds)

    with open(filepath, 'rb') as file:
        data_dict = pickle.load(file)

    muscle_Data = data_dict['spikes']
    ft_data = np.asarray(data_dict['ft_filtered']) if use_filtered_data else select_ft_axes(data_dict['ft'])
    flower_data = _load_scaled_position(filepath)
    moth_id = data_dict['moth_id']
    train_idx, test_idx = kfold_indices(np.arange(len(muscle_Data)), num_folds)[fold_idx]
    MFT_dataset = MuscleForcesTorque_dataset(muscle_data=muscle_Data, 
                                             FT_data=ft_data,
                                             flower_data=flower_data, 
                                             convolution=convolution, 
                                             device=device)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(MFT_dataset, train_idx), batch_size=batch_size, drop_last=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(MFT_dataset, test_idx), batch_size=batch_size, drop_last=False)
    return train_loader, test_loader

def get_MFT(filepath_high, filepath_low, batch_size=16, convolution=None, device=torch.device("cpu"), 
            split=0.8, normalize_ft=True, evaluate=False, perturb=False, data_seed = 0, ft_norm_mode="global",
            use_filtered_data=True):
    utils.set_seeds(data_seed)
    
    high_muscle_data, high_ft_data, high_flower_data, high_moth_id, high_mass_id = load_pickle_dataset(filepath_high, use_filtered_data=use_filtered_data)
    low_muscle_data, low_ft_data, low_flower_data, low_moth_id, low_mass_id = load_pickle_dataset(filepath_low, use_filtered_data=use_filtered_data)
    moth_id = np.concatenate([high_moth_id, low_moth_id], axis=0).reshape(-1)  # shape (N,)
    if normalize_ft:
        ft_data = np.concatenate([high_ft_data, low_ft_data], axis=0)
        normed_ft_data, normed_ft_data_mean = normalize_ft_arrays(ft_data, moth_id=moth_id, mode=ft_norm_mode)
        high_count = len(high_ft_data)
        high_ft_data = normed_ft_data[:high_count]
        low_ft_data = normed_ft_data[high_count:]
        high_ft_data_mean = normed_ft_data_mean[:high_count]
        low_ft_data_mean = normed_ft_data_mean[high_count:]
    else:
        high_ft_data_mean = high_ft_data.mean(axis=1)
        low_ft_data_mean = low_ft_data.mean(axis=1)

    if len(high_muscle_data) > len(low_muscle_data):
        extra_idx = len(high_muscle_data) - len(low_muscle_data)
        resample_idx = np.random.choice(list(range(0,len(low_muscle_data))), extra_idx, replace=True)
        res_low_muscle_data = low_muscle_data[resample_idx]
        res_low_ft_data = low_ft_data[resample_idx]
        res_low_ft_data_mean = low_ft_data_mean[resample_idx]
        res_low_flower_data = low_flower_data[resample_idx]
        res_low_moth_id = low_moth_id[resample_idx]
        res_low_mass_id = low_mass_id[resample_idx]
        
        low_muscle_data = np.vstack([low_muscle_data, res_low_muscle_data])
        low_ft_data = np.vstack([low_ft_data, res_low_ft_data])
        low_ft_data_mean = np.vstack([low_ft_data_mean, res_low_ft_data_mean])
        low_flower_data = np.vstack([low_flower_data, res_low_flower_data])
        low_moth_id = np.vstack([low_moth_id, res_low_moth_id])
        low_mass_id = np.vstack([low_mass_id, res_low_mass_id])
    
    high_MFT_dataset = MuscleForcesTorque_dataset(muscle_data=high_muscle_data, 
                                                  FT_data=high_ft_data,
                                                  mean_FT_data=high_ft_data_mean,
                                                  flower_data=high_flower_data, 
                                                  moth_id=high_moth_id,
                                                  mass_id=high_mass_id,
                                                  convolution=convolution, 
                                                  device=device,
                                                  evaluate=evaluate,
                                                  perturb=perturb)
    
    low_MFT_dataset = MuscleForcesTorque_dataset(muscle_data=low_muscle_data, 
                                                  FT_data=low_ft_data,
                                                  mean_FT_data=low_ft_data_mean,
                                                  flower_data=low_flower_data, 
                                                  moth_id=low_moth_id,
                                                  mass_id=low_mass_id,
                                                  convolution=convolution, 
                                                  device=device,
                                                  evaluate=evaluate,
                                                  perturb=perturb)
    
    # Define split lengths
    train_size = int(split * len(high_muscle_data))
    test_size = len(high_muscle_data) - train_size

    # Split the dataset
    high_train_dataset, high_test_dataset = torch.utils.data.random_split(high_MFT_dataset, [train_size, test_size])
    low_train_dataset, low_test_dataset = torch.utils.data.random_split(low_MFT_dataset, [train_size, test_size])

    # DataLoaders
    high_train_loader = torch.utils.data.DataLoader(high_train_dataset, batch_size=batch_size, shuffle=(not evaluate))
    high_test_loader = torch.utils.data.DataLoader(high_test_dataset, batch_size=batch_size, shuffle=(not evaluate))
    low_train_loader = torch.utils.data.DataLoader(low_train_dataset, batch_size=batch_size, shuffle=(not evaluate))
    low_test_loader = torch.utils.data.DataLoader(low_test_dataset, batch_size=batch_size, shuffle=(not evaluate))
    
    return high_train_loader, high_test_loader, low_train_loader, low_test_loader

def load_pickle_dataset(filepath_high, use_filtered_data=True):
    with open(filepath_high, 'rb') as file:
        data_dict_high = pickle.load(file)
    muscle_data = np.array(data_dict_high['spikes'])
    if use_filtered_data:
        ft_source_path = _resolve_filtered_ft_filepath(filepath_high)
        with open(ft_source_path, 'rb') as file:
            ft_source_dict = pickle.load(file)
        if 'ft_filtered' not in ft_source_dict:
            raise KeyError(f"use_filtered_data=True but 'ft_filtered' is missing from {ft_source_path}")
        ft_data = np.array(ft_source_dict['ft_filtered'])
    else:
        ft_data = select_ft_axes(data_dict_high['ft'])
    flower_data = _load_scaled_position(filepath_high)
    
    moth_id = np.array(data_dict_high['moth_id']).reshape(-1,1)
    mass_array = np.array(data_dict_high['mass'])
    if len(mass_array) == len(moth_id):
        mass_per_sample = mass_array.reshape(-1)
    else:
        # Older segmented pickles store one mass per moth id.
        mass_per_sample = mass_array[moth_id.flatten()]
    mass_id = mass_per_sample.reshape(-1, 1)

    return muscle_data, ft_data, flower_data, moth_id, mass_id
