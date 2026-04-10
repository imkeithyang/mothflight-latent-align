
import numpy as np
# from sklearn.cluster import KMeans
from scipy.signal import find_peaks

def thinning_algorithm(probabilities, num_spikes):
    """
    Uses thinning algorithm to identify spike occurrences.
    
    param probabilities: Array of spike probabilities at each time step.
    param num_spikes: The number of spikes to generate.
    return: Array of indices where spikes occur.
    """
    if num_spikes == 0:
        return []
    
    T = len(probabilities) #Total time steps
    probabilities = (probabilities/np.sum(probabilities)) * 1000000*((T**2)/num_spikes)
    lambda_bar = np.max(probabilities)  # Maximum probability
    while True: #Repeat until poission process yeilds the number of spikes we want

        t = [] #spike times
        s = 0
        while s < T: #Repeat until the end of the time series
            u = np.random.rand() #Candidate time
            w = -np.log(u/lambda_bar) #Enforces exponential
            s = int(s + w)
            D = np.random.rand() # accept-reject value
            
            if s < T and D <= (probabilities[s] / lambda_bar):
                t.append(s)
        if len(t) == int(num_spikes):
            return t

def poisson_process_sampling(probabilities, num_spikes):
    probabilities = probabilities - np.min(probabilities) #Shift into a likelihood
    probabilities = probabilities/np.sum(probabilities)   #Make a probability
    return sorted(np.random.choice(range(len(probabilities)), num_spikes, p=probabilities))


# def kmeans_sampling(spike_likelihood, muscle_counts):
#     sampled_spikes = np.zeros(spike_likelihood.shape)

#     for idx, obs in enumerate(spike_likelihood):
#         sampled_spike = np.zeros(obs.shape)
#         for dim in range(10):
#             num_clusters = muscle_counts[idx, dim]
#             if not num_clusters == 0:
#                 p = (obs[:,dim] - obs[:,dim].min())
#                 p = p /p.sum()
                
#                 # Perform KMeans clustering on the selected dimension
#                 data_to_cluster = obs[:, dim].reshape(-1, 1)  # shape: [400, 1]
#                 data_to_cluster = (data_to_cluster - data_to_cluster.min()) / (data_to_cluster.max())
#                 samples = np.random.choice(np.linspace(0,400, 400), p=p, size = 2000, replace=True)
#                 kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.expand_dims(samples,1))
                
#                 for centroid in kmeans.cluster_centers_:
#                     for c in centroid:
#                         sampled_spike[int(c), dim] = 1
        
#         sampled_spikes[idx] = sampled_spike

#     return sampled_spikes


def find_peak(spike_likelihoods, desired_count, smoothing_param = 10, distance_param = 5):

    p_spikes = np.zeros_like(spike_likelihoods)
    for obs_idx, obs in enumerate(spike_likelihoods):
        p_spike = np.zeros_like(obs)
        for dim in range(10):
            if not desired_count[obs_idx, dim] == 0:
                smoothed_probabilities = np.convolve(obs[:,dim], np.ones(smoothing_param)/smoothing_param, mode='same')
                peaks, _ = find_peaks(smoothed_probabilities, distance=distance_param)
                peak_values = smoothed_probabilities[peaks]
                top_k_peaks = peaks[np.argsort(peak_values)[-desired_count[obs_idx, dim]:]]
                p_spike[top_k_peaks, dim] = 1
        p_spikes[obs_idx] = p_spike

    return p_spikes

def peak_sampling(spike_likelihoods, desired_count, smoothing_param = 10, distance_param = 5):
    #Alias for find_peak since peak finding is not true sampling.
    return find_peak(spike_likelihoods, desired_count, smoothing_param = smoothing_param, distance_param = distance_param)