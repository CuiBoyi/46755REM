
from scipy.io import loadmat
import numpy as np


def load_wind_scen(path="WindScen.mat"):
    """
    Load wind scenarios from WindScen.mat.

    Returns
    -------
    wind : list of np.ndarray
        wind[z] has shape (43, 100)
        z = 0..14 corresponds to MATLAB zones 1..15
        rows = time
        columns = scenarios
    """
    mat = loadmat(path, squeeze_me=True)
    wind_raw = mat["WindScen"]

    wind = [np.array(zone) for zone in wind_raw]

    return wind


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np

def build_load_bids(total_mwh, share_array):
    """
    Build demand bid prices from load shares.

    Parameters
    ----------
    total_mwh : float
        Total system demand
    share_array : np.ndarray
        Load shares in percent

    Returns
    -------
    np.ndarray
        Bid prices for each load
    """
    pmin = 15
    pmax = 20 * 1.3

    smin = np.min(share_array)
    smax = np.max(share_array)

    price = pmax - (pmax - pmin) * (share_array - smin) / (smax - smin)

    return price


# Run example
build_load_bids(2650.5, np.array([3.8,3.4,6.3,2.6,2.5,4.8,4.4,6.0,6.1,6.8,9.3,6.8,11.1,3.5,11.7,6.4,4.5]))