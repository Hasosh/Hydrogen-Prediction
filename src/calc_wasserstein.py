import numpy as np
from scipy.stats import wasserstein_distance

def compute_histogram(data, bins=10, range=None):
    if range is None:
        counts, bin_edges = np.histogram(data, bins=bins)
    else:
        counts, bin_edges = np.histogram(data, bins=bins, range=range)
    return bin_edges, counts

def min_wasserstein(freq_ref, freq_obs, binning):
    dist = np.inf
    for i in range(binning.size):
        if i == 0:
            freq_ref_temp = freq_ref.copy()
            freq_obs_temp = freq_obs.copy()
        else:
            freq_ref_temp = np.concatenate((freq_ref[i:], freq_ref[:i]))
            freq_obs_temp = np.concatenate((freq_obs[i:], freq_obs[:i]))
        dist_i = wasserstein_distance(binning, binning, freq_obs_temp, freq_ref_temp)
        if dist_i < dist:
            dist = dist_i
    return dist

def calculate_wasserstein_distance(data_ref, data_obs, bins=10, range=None):
    binning_ref, counts_ref = compute_histogram(data_ref, bins=bins, range=range)
    binning_obs, counts_obs = compute_histogram(data_obs, bins=bins, range=range)
    
    if binning_ref.dtype == binning_obs.dtype:
        assert np.array_equal(binning_ref, binning_obs), 'Bins are not equal.'
    else:
        # to account for different dtypes of the numpy arrays binning_ref, binning_obs
        assert np.array_equal(binning_ref, binning_obs.astype(binning_ref.dtype)) or np.array_equal(binning_ref.astype(binning_obs.dtype), binning_obs), 'Bins are not equal.'

    if np.sum(counts_ref) == 0 or np.sum(counts_obs) == 0:
        return 'NaN'
    
    freq_ref = counts_ref / np.sum(counts_ref)
    freq_obs = counts_obs / np.sum(counts_obs)
    
    binning = binning_ref[:-1] + ((binning_ref[1:] - binning_ref[:-1]) / 2)
    distance = min_wasserstein(freq_ref, freq_obs, binning)
    
    return distance