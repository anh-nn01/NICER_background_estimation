import numpy as np

"""======================================================
		Path to Stage 1 model: DNN 
======================================================"""
dnn_savedir = './DNN_weights'
# dnn_weight_path = f'{dnn_savedir}/resnet_freq_reduced_v4_t5_46features_v1.pt'
# dnn_weight_path = '{dnn_savedir}/resnet_freq_reduced_v4_t5_46features_v2.pt'
dnn_weight_path = f'{dnn_savedir}/resnet_freq_reduced_v4_t5_46features_v3.pt'

# dnn_weight_path = f'{dnn_savedir}/bertx4_freq_reduced_v4_t5_46features_v1.pt'
# dnn_weight_path = f'{dnn_savedir}/bertx4_freq_reduced_v4_t5_46features_v2.pt'
# dnn_weight_path = f'{dnn_savedir}/bertx4_freq_reduced_v4_t5_46features_v3.pt'

""" Path to standardization (mean, mu) for each feature"""
standardize_path = f'{dnn_savedir}/feature_mu_sigma.pkl'

"""======================================================
		Path to Stage 2 model: spectra clustering
======================================================"""
cluster_saveDir = 'cluster_model/2levels'
n_clusters = 10 # 15
ver = 'v2' # 'v3'

# load energy bin group used to train the cluster model
ebins_lvl1_path = f"{cluster_saveDir}/kmeans_ebins_lvl1.pkl"
ebins_lvl2_path = f"{cluster_saveDir}/kmeans_ebins_lvl2.pkl"

# load trained cluster model
model_cluster_lvl1_path = f"{cluster_saveDir}/kmeans_k1={n_clusters}_k2=dynamic_lvl1_{ver}.pkl"
model_cluster_lvl2_path = f"{cluster_saveDir}/kmeans_k1={n_clusters}_k2=dynamic_lvl2_{ver}.pkl"

# load spectra library for each cluster
spec_library_path = f"{cluster_saveDir}/spec_library_k={n_clusters}_2lvls_{ver}.pkl"