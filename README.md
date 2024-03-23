# NICER_background_estimation
Two-stage ML framework for NICER background estimation (spectra &amp; light curve)

Source code for running prediction of NICER background spectra / light curve given MKF parameters of an ObsID in CSV format stored in `./test_obsIDs/MKF_params_{obsID}.csv`

---
Script for prediction: <br>
  ``` python3 run_prediction.py --obsid <ObsID> --output_type <"spec" OR "lc">) ```<br>
* `--obsid`: input observation ID <br>
-> Assume the file is CSV file => WE MAY NEED TO CHANGE THIS IMPLEMENTATION FOR RAW .gz files <br>
-> format: `mkf_path = f"./test_obsIDs/MKF_params_{args.obsid}.csv"` 
=> WE CAN CHANGE THIS IN PACKAGE IMPLEMENTATION

* `--output_type`: 'spec' for normalized bkgd spectra prediction
        				 'lc' for bkgd light curve prediction

---
Notebooks to train & evaluate the model (both quantitatively & qualitatitvely):
1. `notebooks/train.ipynb`
2. `notebook/val.ipynb`

We will publish the dataset soon. Please update the path to the dataset in `config.path_new_features_csv` and `config.path_spec_full_csv`
* `config.path_new_features_csv`: input MKF parameters (46 parameters)
* `config.path_spec_full_csv`: target bkgd photon counts (0.2 - 12 keV)
