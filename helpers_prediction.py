import torch
import numpy as np

import pickle
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


"""################################################################################
    Aggregate raw photon counts in original channels (R^1180) 
    by different energy bins (R^8)
    -> to assign basis spectra for each raw prediction
    
    :param orig_counts: original photon counts for 1180 channels (0.2-12keV)
    :param ebins: a dictionary containing list of energy channels for 
                  different energy bins
###############################################################################"""
def bin_photon_counts(orig_counts, ebins):
    nbins = len(ebins)
    agg_counts = np.zeros((orig_counts.shape[0], nbins))
    
    for i, ebin in enumerate(ebins):
        c_min = int(ebin[0]) - 20  # min channel
        c_max = int(ebin[-1]) - 20 # max channel
        agg_counts[:, i] = orig_counts[:, c_min:c_max+1].sum(axis=1)
        
    return agg_counts

"""====================================================================
    Output the predicted light curve for an ObsID
    
    :param model - the trained model
    :param X - input to the model (sequence of events in the ObsID)
===================================================================="""
def predict_lightcurve(model, X):
    # execute spectra prediction & aggregate along channel dimension
    preds = model(X).detach().cpu().numpy()
    return preds.sum(axis=1)

"""====================================================================
    Ouput the predicted normalized spectra for an ObsID
    
    :param model - the trained model
    :param X - input to the model (sequence of events in the ObsID)
    :param model_cluster - trained unsupervised clustering model
    :param cluster_ebins - energy bin groups used to train the 
                           clustering model
    :param spec_library - library of basis spectra, which is the
                          normalized spectra to each cluster
===================================================================="""
def predict_normalized_spectra(model, X, 
                               model_cluster, models_cluster_2,
                               cluster_ebins, cluster_ebins_2,
                               spec_library):
    # Stage 1: execute spectra prediction
    preds = model(X).detach().cpu().numpy()
    
    # Stage 2: assign basis spectra for each event
    # a. group predicted photon counts into predefined bins (kmeans_ebins)
    preds_binned_lvl1 = bin_photon_counts(preds, cluster_ebins)
    preds_binned_lvl2 = bin_photon_counts(preds, cluster_ebins_2)
    # b. take sqrt of grouped photon counts for spectra clustering
    preds_binned_lvl1 = np.sqrt(preds_binned_lvl1)
    preds_binned_lvl2 = np.sqrt(preds_binned_lvl2)
    # c. assigned each event to a basis spectrum (defined in spec_library)
    for i in range(len(preds)):
        # Get LEVEL 1 cluster_id of the closest kmeans cluster centroid 
        cluster_id = model_cluster.predict(preds_binned_lvl1[i].reshape(1,-1))
        cluster_id = int(cluster_id)
        
        # Get LEVEL 2 cluster_id of the closest kmeans cluster centroid 
        # CASE 1: NO LEVEL 2 CLUSTERING MODEL
        if models_cluster_2[cluster_id] is None:
            # Assign spectra basis based on the LEVEL 1 spectra id
            preds[i] = spec_library[cluster_id] # LEVEL 1 normalized spectra
        # CASE 2: USE LEVEL 2 CLUSTERING MODEL
        else:
            cluster2_id = models_cluster_2[cluster_id].predict(preds_binned_lvl2[i].reshape(1,-1))
            cluster2_id = int(cluster2_id)
            preds[i] = spec_library[cluster_id][cluster2_id] # LEVEL 2 normalized spectra
        
        
    # d. normalized spectra = mean of all predictions
    norm_spec = preds.mean(axis=0)
    
    return norm_spec

# """
#     DEPRECATED: THIS IS FOR SINGLE-LEVEL CLUSTERING PREDICTION
# """
# """====================================================================
#     Ouput the predicted normalized spectra for an ObsID
    
#     :param model - the trained model
#     :param X - input to the model (sequence of events in the ObsID)
#     :param model_cluster - trained unsupervised clustering model
#     :param cluster_ebins - energy bin groups used to train the 
#                            clustering model
#     :param spec_library - library of basis spectra, which is the
#                           normalized spectra to each cluster
# ===================================================================="""
# def predict_normalized_spectra(model, X, 
#                                model_cluster, 
#                                cluster_ebins, 
#                                spec_library):
#     # Stage 1: execute spectra prediction
#     preds = model(X).detach().cpu().numpy()
    
#     # Stage 2: assign basis spectra for each event
#     # a. group predicted photon counts into predefined bins (kmeans_ebins)
#     preds_binned = bin_photon_counts(preds, kmeans_ebins)
#     # b. take sqrt of grouped photon counts for spectra clustering
#     preds_binned = np.sqrt(preds_binned)
#     # c. assigned each event to a basis spectrum (defined in spec_library)
#     for i in range(len(preds)):
#         # Get cluster_id of the closest kmeans cluster centroid 
#         cluster_id = model_cluster.predict(preds_binned[i].reshape(1,-1))
#         cluster_id = int(cluster_id)
#         # Assign spectra basis based on the spectra id
#         preds[i] = spec_library[cluster_id]
#     # d. normalized spectra = mean of all predictions
#     norm_spec = preds.mean(axis=0)
    
#     return norm_spec

""" Smooth Prediction Spectrum to for easier visualization"""
def smooth(y, window):
    kernel = np.ones(window)/window
    y_smooth = np.convolve(y, kernel, mode='same')
    # Concatenate to reshape to original y's shape
    pad_size = window // 2
    # print(concat_shape)
    y_smooth[:pad_size] = y[:pad_size]
    return y_smooth

from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score



"""============================================================
    Plotting function:
        Plot Estimated vs Ground-Truth Normalized Spectra 
        (in frequency domain / Energy domain)
============================================================"""
def plotSpecs_freq_domain(ebins, specs_pred, 
                          specs_gt, specs_gt_std, 
                          specs_scorpeon, 
                          title='X-Ray Spectrum'):
    plt.figure(figsize=(9,5))
    plt.title(title, fontsize=20)
    plt.errorbar(ebins, specs_gt, specs_gt_std, marker='+', linestyle='None', 
                 markersize=12, color='darkblue', 
                 linewidth=2, label=f'Ground Truth', alpha=0.5)
    # plt.errorbar(ebins, specs_gt+1e-3, list(specs_gt_std), marker='+', linestyle=None)
    
    """ 3.1. SCORPEON Spectrum Visualization"""
    plt.plot(ebins, specs_scorpeon, '-', markersize=20, color='darkorange', linewidth=3,
             alpha=0.9, label=f'SCORPEON Estimation')

    """ 3.2. Predicted Spectrum Visualization"""
    # specs_pred = smooth(specs_pred, 1)
    plt.plot(ebins, specs_pred, '-', markersize=16, color='red', linewidth=3,
             alpha=0.9, label=f'ML Estimation')
    plt.xlabel('Energy (10eV)', fontsize=16)
    plt.ylabel('Photon Counts\n(counts/s/10eV)', fontsize=16)
    plt.grid('--')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0.1*min(specs_pred.min(), specs_gt.min()), 
              1+max(10, max(max(specs_pred.max(), specs_scorpeon.max()), specs_gt.max()))])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'./fig/{title}.png', transparent=False)
    plt.show()
    
"""============================================================
    Plotting function:
        Plot Estimated vs GT Light curve (Time domain)
============================================================"""
def plotSpecs_time_domain(ebins, total_cts_pred, total_cts_gt, total_cts_sc, 
                          title='Photon Counts in Time Domain'):
    plt.figure(figsize=(9,5))
    plt.title(title, fontsize=20)
    plt.plot(total_cts_gt, '-', markersize=12, color='darkblue', linewidth=2, label='Ground Truth')
    plt.plot(total_cts_sc, '-', color='darkorange', linewidth=2.5, 
             alpha=0.7, label=f'SCORPEON Estimation')
    plt.plot(total_cts_pred, color='red', linewidth=2, 
             alpha=0.7, label=f'ML Estimation')
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Total photon counts\n(cts/s)', fontsize=16)
    plt.grid('--')
    max_range = max(30, total_cts_gt.max())
    max_range = max(max_range, total_cts_pred.max())
    max_range = max(max_range, total_cts_sc.max())
    plt.ylim([0, max_range+10])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.savefig(f'./fig/{title}.png', transparent=False)
    plt.show()