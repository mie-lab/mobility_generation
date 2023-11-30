import numpy as np
import scipy


def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(arr, np.arange(min, max, float(max - min) / bins))
    return distribution, base[:-1]


def log_arr_to_distribution(arr, min=-30.0, bins=100):
    """
    calculate the logarithmic value of an array and convert it to a distribution
    :param arr: np.array, input array
    :param bins: int, number of bins between min and max
    :return: np.array,
    """
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = _filter_zero(arr)
    arr = np.log(arr)
    distribution, base = np.histogram(arr, np.arange(min, 0.0, 1.0 / bins))
    ret_dist, ret_base = [], []
    for i in range(bins):
        if int(distribution[i]) == 0:
            continue
        else:
            ret_dist.append(distribution[i])
            ret_base.append(base[i])
    return np.array(ret_dist), np.array(ret_base)


def _filter_zero(arr):
    """
    remove zero values from an array
    :param arr: np.array, input array
    :return: np.array, output array
    """
    arr = np.array(arr)
    filtered_arr = np.array(list(filter(lambda x: x != 0.0, arr)))
    return filtered_arr
