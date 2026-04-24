import numpy as np
import os
from scipy.interpolate import interp1d
import pandas as pd
import pickle

# Data Parameters
min_trial_size = 200 # Minimium observations required to be a wingstroke
max_trial_size = 600 # Maximium observations required to be a wingstroke
num_points = 400     # Constant to normalize data on phase (wingstroke)
directory = '/hpc/group/tarokhlab/hy190/data/fatmoth_filtered/' # Directory where files are stored
metadata_path = '/hpc/group/tarokhlab/hy190/data/fatmoth_new/MothMetaData.csv'
save_directory = "./data/"
speciman_F = True # A flag to include speciman ID with conditional values (F/T)
wingbeat_F = False # A flag to include wingbeat length with conditional values (F/T)
stimuli_F = False # A flag to include stimuli position with conditional values (F/T)
trial_F = False # A flag to include trial number as conditional value (F/T)
meanFT_F = True # A flag to include Mean F/T instead of True Waveform (F/T)
muscles_fn = "phase_muscles_speciman_meanFT.npy" # Save file name for Muscle Spikes
ft_fn = "phase_FT_speciman_meanFT.npy" # Save file name for conditional Values
processType = "HIGH" # Process what type of data

data_dict = {'spikes':[], 'ft':[], 'ft_filtered':[], 'flower':[], 'moth_id':[], 'mass':[]}
experiment_id = []
moth_count = 0
print(sorted(os.listdir(directory)))
# Loop through each file in the directory
for filename in sorted(os.listdir(directory)):
    if processType not in filename: continue
    if not filename.endswith('.csv'): continue
    df = pd.read_csv(directory + filename)
    wingstroke_indices = df['wb'].unique()
    # find wing storkes
    for ws_Index in wingstroke_indices:
        ws = df[df['wb'] == ws_Index]
        
        # segement different columns into different types of data
        ws_spike = ws.iloc[:,0:10].to_numpy()
        ws_ft = ws.iloc[:, 12:18].to_numpy()
        ws_ft_filtered = ws.iloc[:, 19:22].to_numpy()
        ws_flower = ws.iloc[:,10:12].to_numpy()

        # filter out too long/short wing strokes
        if min_trial_size >= len(ws) or len(ws) >= max_trial_size:
            continue
        
        # Resample spikes into fixed num_points matrix
        normalized_muscle = np.zeros((num_points, 10))
        for idx in range(ws_spike.shape[1]):
            muscle_data = ws_spike[:,idx]
            muscle_timing = np.where(muscle_data==1)[0]
            muscle_timing = np.array((muscle_timing / len(muscle_data)) * num_points, dtype=int)
            normalized_muscle[muscle_timing, idx] = 1
        
        original_length = len(ws_ft)
        original_indices = np.linspace(0, 1, num=original_length)
        new_indices = np.linspace(0, 1, num=num_points)
        resampled_data = np.zeros((num_points, ws_ft.shape[1]))
        
        # Resample forces and torques into fixed num_points with linear interpolation
        for i in range(ws_ft.shape[1]):
            interpolator = interp1d(original_indices, ws_ft[:, i], kind='linear')
            resampled_data[:, i] = interpolator(new_indices)
        normalized_ws_ft = resampled_data

        original_length = len(ws_ft_filtered)
        original_indices = np.linspace(0, 1, num=original_length)
        new_indices = np.linspace(0, 1, num=num_points)
        resampled_data = np.zeros((num_points, ws_ft_filtered.shape[1]))

        # Resample filtered forces and torques into fixed num_points with linear interpolation
        for i in range(ws_ft_filtered.shape[1]):
            interpolator = interp1d(original_indices, ws_ft_filtered[:, i], kind='linear')
            resampled_data[:, i] = interpolator(new_indices)
        normalized_ws_ft_filtered = resampled_data
        
        original_length = len(ws_flower)
        original_indices = np.linspace(0, 1, num=original_length)
        new_indices = np.linspace(0, 1, num=num_points)
        resampled_data = np.zeros((num_points, ws_flower.shape[1]))
        
        # Resample flowers position and velocity into fixed num_points with linear interpolation
        for i in range(ws_flower.shape[1]):
            interpolator = interp1d(original_indices, ws_flower[:, i], kind='linear')
            resampled_data[:, i] = interpolator(new_indices)
        normalized_ws_flower = resampled_data

        data_dict['spikes'].append(normalized_muscle)
        data_dict['ft'].append(normalized_ws_ft)
        data_dict['ft_filtered'].append(normalized_ws_ft_filtered)
        data_dict['flower'].append(normalized_ws_flower)
        data_dict['moth_id'].append(moth_count)
    
    experiment_id.append(filename[0:10])
    moth_count += 1

meta = pd.read_csv(metadata_path)
# Add meta data to the data_dict
moth_to_mass = {row['moth']: (row['mass_i'], row['mass_f']) for _, row in meta.iterrows()}

# For each sample, get the corresponding mass from its moth id.
mass_by_moth_id = []
for eid in experiment_id:
    low, high = moth_to_mass[eid]
    if processType == "HIGH":
        mass_by_moth_id.append(high)
    else:
        mass_by_moth_id.append(low)
data_dict['mass'] = [mass_by_moth_id[moth_id] for moth_id in data_dict['moth_id']]

# Good check to make sure most of the Muscles are spiking
assert(len(data_dict['spikes']) == len(data_dict['ft']))
assert(len(data_dict['spikes']) == len(data_dict['ft_filtered']))
assert(len(data_dict['spikes']) == len(data_dict['mass']))
print(len(data_dict['spikes']))
with open(directory + 'fatmoth_{}.pickle'.format(processType), 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
