import os
import numpy as np
import pandas as pd
# from astropy.io import fits
# from astropy.table import Table
import pickle as pkl
import random

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from tqdm.notebook import tqdm

%load_ext autoreload
%autoreload 2
from config import *

import subprocess as subp
import multiprocessing
from multiprocessing import Pool
n_cpus = multiprocessing.cpu_count()
print('Num CPU cores:', n_cpus)

import psutil
from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('This runtime has {:.1f} gigabytes of available RAM.'.format(ram_gb))


def load_data(feature_path, spec_path):
    """ Load input features"""
    try:
        data_features
    except:
        # data_lightcurves  = np.load(path_features, allow_pickle=True)
        # data_features = pd.DataFrame(data_lightcurves['dataLc'], columns=data_lightcurves['cols'])
        # data_features.columns = data_features.columns.astype(str)
        data_features = pd.read_csv(feature_path)

    """ Load spectral labels"""
    try:
        data_specs
    except:
        # data_specs = np.load(path_spectra_labels, allow_pickle=True)
        # data_specs.columns = data_specs.columns.astype(str)
        data_specs = pd.read_csv(spec_path)
        
    return data_features, data_specs

