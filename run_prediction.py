# Import basic packages
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.table import Table
import pickle as pkl
import random
import torch
import torch.nn as nn
import argparse

# environment & data variables
import config

# Model & Prediction function
from models import *
from helpers_prediction import bin_photon_counts, predict_lightcurve, predict_normalized_spectra, smooth
from helpers_prediction import plotSpecs_freq_domain, plotSpecs_time_domain
from eval_metrics import spectra_rmse, spectra_harnessRatio_rmse, ns3_score

# Path to models
from config_prediction import dnn_weight_path, standardize_path
from config_prediction import ebins_lvl1_path, ebins_lvl2_path, model_cluster_lvl1_path, model_cluster_lvl2_path, spec_library_path

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""#######################################################################################
       Standardization process
        
       :param X - input .csv file of events in an ObsID
                   Expected format: <#events, #MKF parameters>
                   
       :return - standardized X
                   X[feature] = (X[feature] - mean[feature]) / (std[feature] + 1e3)
                   
       * NOTE: use (mean, std) defined in `config_prediction.standardize_path`
               -> a `.pkl` file in dictionary format: { <feature>: <mean, std> }
########################################################################################"""
def standardize(X):
    """ 1. Load features' mean and std (computed using X_train)"""
    with open(standardize_path, 'rb') as f:
        feature_mu_sigma = pkl.load(f)
        
    """ 2. Standardization"""
    for fname in config.input_features:
        mu, sigma = feature_mu_sigma[fname] # mean & std.dev of each feature
        X[fname] = (X[fname] - mu) / (sigma + 1e-3)
        
    return X

"""#######################################################################################
       Inference process: save normalized background spectra of ObsID in a `.csv` file
        
       :param X - input .csv file of an ObsID's events 
                  in original values (un-standardized)
                  * Expected format: <#events, #MKF parameters>
                   
       :return - normalized background spectra of ObsID in a Pandas Dataframe
                   
       * NOTE: use DNN model (Stage 1) & 2-level K-means Clustering (Stage 2)
               defined in `config_prediction` to run prediction
########################################################################################"""
def get_bkgd_normalized_spectra(X):
    """ 1. Standardization process"""
    X = standardize(X)
    
    """ 2. Load trained DNN model (Stage 1)"""
    dnn = torch.load(dnn_weight_path, map_location=torch.device(device))
    
    """ 3. Load trained 2-level Clustering model (Stage 2)"""
    # load energy bin group used to train the cluster model
    with open(ebins_lvl1_path, "rb") as f:
        kmeans_ebins = pkl.load(f)
    with open(ebins_lvl2_path, "rb") as f:
        kmeans_ebins_2 = pkl.load(f)
        
    # load trained cluster model
    with open(model_cluster_lvl1_path, "rb") as f:
        model_cluster = pkl.load(f)
    with open(model_cluster_lvl2_path, "rb") as f:
        models_cluster_2 = pkl.load(f)

    # load spectra library for each cluster
    with open(spec_library_path, "rb") as f:
        spec_library = pkl.load(f)
        
    """ 4. Predict normalized background spectra"""
    obsid_specs_preds = predict_normalized_spectra(dnn, X=torch.Tensor(np.array(X)).to(device), 
                                                   model_cluster=model_cluster, 
                                                   models_cluster_2=models_cluster_2,
                                                   cluster_ebins=kmeans_ebins, 
                                                   cluster_ebins_2=kmeans_ebins_2,
                                                   spec_library=spec_library)
    
    """ 5. ADD CODE HERE TO SAVE PREDICTED Background Spectra """
    
    return obsid_specs_preds

"""#######################################################################################
       Inference process: save background light curve of ObsID in a `.csv` file
        
       :param X - input .csv file of an ObsID's events 
                  in original values (un-standardized)
                  * Expected format: <#events, #MKF parameters>
                   
       :return - background light curve of ObsID in a Pandas Dataframe
                   
       * NOTE: use DNN model (Stage 1) defined in `config_prediction` to run prediction
       * NOTE: light curve prediction does not use clustering models (Stage 2)
########################################################################################"""
def get_bkgd_lightcurve(X):
    """ 1. Standardization process"""
    X = standardize(X)
    
    """ 2. Load trained DNN model (Stage 1)"""
    dnn = torch.load(dnn_weight_path, map_location=torch.device(device))
        
    """ 3. Predict lightcurve background spectra (NO NEED FOR CLUSTERING MODEL)"""
    obsid_lc_preds = predict_lightcurve(dnn, X=torch.Tensor(np.array(X)).to(device))
    
    """ 4. ADD CODE HERE TO SAVE PREDICTED Background Light curve """
    
    return obsid_lc_preds


"""######################################################################################
	command line arguments to run predictions:
        python3 run_prediction.py --obsid <ObsID> --output_type <"spec" OR "lc">)
        
    description:
    	--obsid: input observation ID
        	-> Assume the file is `.csv` file: 
            	=> MAY NEED TO CHANGE THIS IMPLEMENTATION FOR RAW .gz files
            -> format: mkf_path = f"./test_obsIDs/MKF_params_{args.obsid}.csv" 
            	=> CHANGE THIS IN PACKAGE IMPLEMENTATION
        
        --output_type: 	'spec' for normalized bkgd spectra prediction
        				'lc' for bkgd light curve prediction
        
"######################################################################################"""
if __name__ == '__main__':
    """ IMPLEMENT CODE HERE TO RUN PREDICTION BASED ON CMD ARGUMENTS"""
    
    """*******************************************************************************
        SYNTAX: 
            python3 run_prediction.py --obsid <ObsID> --output_type <"spec" OR "lc">
    *******************************************************************************"""
    parser = argparse.ArgumentParser(description="cmd format: python3 run_prediction.py --obsid <ObsID> --output_type <\"spec\" OR \"lc\">")
    parser.add_argument("--obsid", type=str, 
                        help="Specify ObsID")
    parser.add_argument("--output_type", type=str, 
                        help="Specify prediction type: \"spec\" for bkgd spectra or \"lc\" for bkgd light curve.")
    args = parser.parse_args()
    
    """ 1. Load input MKF parameters"""
    mkf_path = f"./test_obsIDs/MKF_params_{args.obsid}.csv" # CHANGE THIS IN PACKAGE IMPLEMENTATION
    X = pd.read_csv(mkf_path)
    ebins = [int(e) for e in config.energy_bins]
    
    """ 2. Run predictions"""
    if args.output_type == 'spec' or args.output_type == 'spectra' or args.output_type == 'spectrum':
        obsid_specs_preds = get_bkgd_normalized_spectra(X)
        title = f'ObsID {args.obsid}: background spectra'
        
        # plots the results
        plt.figure(figsize=(9,5))
        plt.title(title, fontsize=20)
        plt.plot(ebins, obsid_specs_preds, markersize=16, color='red', linewidth=3, alpha=0.9)
        plt.grid('--')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim([1e-5, max(1e1, 1+max(obsid_specs_preds))])
        plt.xlabel('Energy (10eV)', fontsize=16)
        plt.ylabel('Photon Counts\n(counts/s/10eV)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('0.TEST_spectra.png', transparent=False)
        
    elif args.output_type == 'lc' or args.output_type == 'lightcurve':
        obsid_lc_preds = get_bkgd_lightcurve(X)
        title = f'ObsID {args.obsid}: background spectra'
        
        # plots the results
        plt.figure(figsize=(9,5))
        plt.title(title, fontsize=20)
        plt.plot(obsid_lc_preds, color='red', linewidth=2)
        plt.grid('--')
        plt.ylim([0, max(30, max(obsid_lc_preds))])
        plt.xlabel('Time (s)', fontsize=16)
        plt.ylabel('Total photon counts\n(cts/s)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('0.TEST_lightcurve.png', transparent=False)
    else:
        raise Error("--output_type must be either \"spec\" (bkgd spectra) or \"lc\" for (bkgd light curve)!!!")