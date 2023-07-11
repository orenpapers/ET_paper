import numpy as np

def centrelize_by_target_word ():
    #for each row, put (0,0) as the center of the time-series
    pass

def apply_mean_pooling(vec, factor = 5):
    #https://stackoverflow.com/questions/60076262/python-vector-apply-mean-across-axis-in-chunks-of-size-5

    num_elements_to_add = factor - (len(vec) % factor)
    added_val = vec[-1]
    vec += [added_val] * num_elements_to_add
    averaged = np.array(vec).reshape(-1, factor).mean(axis=1).flatten()
    return averaged


def apply_smoothing_1d(vec, sigma = 3):
    from scipy.ndimage import gaussian_filter1d
    smoothed_vec = gaussian_filter1d(vec, sigma)
    return smoothed_vec