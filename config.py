"""#############################
    STEP 0: DATA PREPARATION
#############################"""
""" 1. Environment's directories & file names """
wdir = '/home/idies/workspace/Temporary/anh_globus/scratch/nicer-background-2'
nicerObs_dir = '/FTP/nicer/data/obs'
obs_dir = '/FTP/nicer/data/catalog/nicermastr.tdat'
obsDataFile = 'obsData.csv'

""" 2. Information for nicerl2 filtering & screening process """
GEOMAG_PATH = '/FTP/caldb/data/gen/pcf/geomag/' # Path to Geomagnetic data
FILTCOLUMNS = 'NICERV4,3C50'
DETLIST = 'launch,-14,-34'  # List of Selected FPU detectors

# missing files: OBS_ID=1012020134, 

"""#############################
    STEP 1: DATA PREPROCESSING
#############################"""
input_features = ['ROLL', 'SAT_LAT', 'SAT_LON', 'SAT_ALT', 'ELV', 'BR_EARTH', 'SUNSHINE', 'TIME_SINCE_SUNSET', 
                  'SUN_ANGLE', 'BETA_ANGLE', 'LOCAL_TIME', 'MOON_ANGLE', 'RAM_ANGLE', 'EAST_ANGLE', 'ANG_DIST', 
                  'SAA', 'SAA_TIME', 'COR_ASCA', 'COR_SAX', 'MCILWAIN_L', 'MAGFIELD', 'MAGFIELD_MIN', 'MAG_ANGLE', 
                  'AP8MIN', 'AE8MIN', 'ATT_ANG_AZ', 'ATT_ANG_EL', 'FPM_RATIO_REJ_COUNT', 'FPM_OVERONLY_COUNT', 'FPM_UNDERONLY_COUNT', 
                  'FPM_DOUBLE_COUNT', 'FPM_FT_COUNT', 'FPM_NOISE25_COUNT', 'XTI_PNT_JITTER', 'KP', 'SOLAR_PHI', 'COR_NYM', 
                  'ANG_DIST_X', 'ANG_DIST_Y', 'FPM_TRUMP_SEL_1500_1800', 'FPM_RATIO_REJ_300_1800', 'FPM_SLOW_LLD', 
                  'MPU_NOISE20_COUNT', 'MPU_NOISE25_COUNT', 'MPU_OVERONLY_COUNT', 'MPU_UNDERONLY_COUNT']

energy_bins = [str(e) for e in range(20, 1200)]

""" Path to data products"""
# Input features
path_features = '/home/idies/workspace/Storage/anh_globus/persistent/nicerbackground-products/dataLc.npz'
# Targets (22 bins)
path_spec_22bin = '/home/idies/workspace/Storage/anh_globus/persistent/nicerbackground-products/spec_all.pkl'
# Targets (full spectra from 20-1200 keV)
path_spec_full = '/group/wanglei/data/QG_outputs/spec_all.20.1200.pkl'


""" Original Data Path"""
path_features = './dataLc.npz'
path_spectra_labels = './spec_all.pkl'

""" CSV-converted Data Path"""
path_features_csv = '/group/wanglei/data/QG_outputs/input_features.csv'
path_new_features_csv = '/group/wanglei/data/QG_outputs/new_features.csv'
path_spec_22bin_csv = './spectras.csv'
path_spec_full_csv = '/group/wanglei/data/QG_outputs/full_spectras.csv'

""" NPY-converted Data Path"""
path_all_features_npy = '/group/wanglei/data/QG_outputs/all_features.npy'
path_spec_full_npy = '/group/wanglei/data/QG_outputs/full_spectras.npy'
path_full_XY_npy = '/group/wanglei/data/QG_outputs/full_dataset.npy'


"""#############################
    STEP 2: MACHINE LEARNING
#############################"""
# import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
""" Debugging mode: use 10% of Training set for Experiments"""
debug = True
""" Train-Val-Test Ratio"""
r_val = 0.1
r_test = 0.1
""" Hyperparameters for Deep Networks """
bs = 512 # batch size
lr = 0.001