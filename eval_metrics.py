import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score


"""
    Compute RMSE between lists of spectra x1 and x2
        formula: err = 1/N * (rmse(x1[0],x2[0]) + ... rmse(x1[n],x2[n])))
            * where N = # normalized spectra
            -> mean of each spectra pair's rmse
"""
def spectra_rmse(x1, x2):
    n = len(x1)
    err_list = [mean_squared_error(x1[i], x2[i])**0.5 for i in range(n)]
    return np.array(err_list).mean()

"""
    Compute RMSE between hardness ratio of x1 and x2
        formula: err = 1/N * (rmse(Hardness_Ratio(x1[0]), Hardness_Ratio(x2[0]) 
                                + ... rmse(Hardness_Ratio(x1[n]), Hardness_Ratio(x2[n]))))
            * where N = # normalized spectra
            -> mean of each spectra pair's rmse
"""
def hardness_ratio(specs):
    # Soft range: 0.2 - 2.5 keV
    soft_counts = specs[:, :231].sum(axis=1)
    # Hard range: 5 - 12 keV
    hard_counts = specs[:, 480:].sum(axis=1)
    # Hardness ratio = (5-12keV) / (0.2-2.5keV)
    hratio = np.abs(hard_counts - soft_counts) / (soft_counts+hard_counts+1e-9)
    
    return hratio

def spectra_harnessRatio_rmse(x1, x2):
    n = len(x1)
    hratio_1 = hardness_ratio(x1)
    hratio_2 = hardness_ratio(x2)
    
    return mean_squared_error(hratio_1, hratio_2)**0.5

"""
    Compute RMSLE between lists of spectra x1 and x2
        formula: err = 1/N * (rmsle(x1[0],x2[0]) + ... rmsle(x1[n],x2[n])))
            * where N = # normalized spectra
            -> mean of each spectra pair's rmse
"""
def spectra_rmsle(x1, x2):
    n = len(x1)
    err_list = [mean_squared_error(np.log(x1[i]+1), np.log(x2[i]+1))**0.5 for i in range(n)]
    return np.array(err_list).mean()

"""
    Compute RMSLE between lists of spectra x1 and x2
        formula: err = 1/N * (corr(x1[0],x2[0]) + ... corr(x1[n],x2[n])))
            * where N = # normalized spectra
            -> mean of each spectra pair's rmse
"""
def spectra_corr(x1, x2):
    n = len(x1)
    corr_list = [pearsonr(x1[i], x2[i])[0] for i in range(n)]
    return np.array(corr_list).mean()


"""
    Compute Normalized Spectra Similarity Score (NS3) between lists of spectra x1 and x2
    citation: at the end of this page: 
        https://www.mathworks.com/help/images/ref/ns3.html#mw_bf2b1a70-7c4e-486f-b64a-94232f05c7ce_seealso
"""
def ns3_score(x1, x2):
    total_score = 0
    n = len(x1)
    
    for i in range(n):
        """ 1. compute Euclidean distances: rmse """
        euclidean_dist = mean_squared_error(x1[i], x2[i])**0.5
        """ 2. compute SAM distances"""
        sam_dist = np.dot(x1[i], x2[i]) / (np.linalg.norm(x1[i]) * np.linalg.norm(x2[i]))
        sam_dist = np.arccos(sam_dist)
        """ 3. compute NS3 score"""
        score = np.sqrt(euclidean_dist**2 + (1-np.cos(sam_dist))**2)
        """ 4. add to total score"""
        total_score += score
        
    return total_score / n