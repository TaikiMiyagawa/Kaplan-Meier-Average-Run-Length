# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" Ignore IntegrationWarning. """
import glob
import os
import pickle
from datetime import datetime

import numpy as np
import ruptures as rpt
from changepoint_online import NPFocus
from tqdm import tqdm

from metrics import calc_naive_ARL
from ocpdet import CUSUM, EWMA
from statistic_tools import calc_KME

name_process = "WISDM"

np.random.seed(0)

# =========================================================
# Estimate ARL
# =========================================================
# *************** User-defined parameters ********************** #
data_label: str = ["labeled", "unlabeled"][1]  # "labeled" or "unlabeled"
# mode = common is the one used in the paper
mode: str = ["rare", "normal", "common"][2]  # "rare", "normal", or "common"

# Directory where the dataset is saved
dir_data = "./dataset"

# Directory where the results will be saved
dir_results = f"./results/{name_process}_{data_label}_{mode}"
# ************************************************************** #

ls_statistics_type = [
    "WinL1", "WinNorm", "WinAR", "NPFocus", "CUSUMraw", "EWMA"]
window_size = 30
dc_rpt_models = {
    "WinL1": "l1",
    "WinNorm": "normal",
    "WinAR": "ar",
}

for statistics_type in ls_statistics_type:
    # Set the model and parameters
    if statistics_type == "WinL1":
        if data_label == "labeled":
            ls_thresh = np.linspace(1, 2, 10)
        else:
            ls_thresh = np.linspace(0.5, 50, 10)
    elif statistics_type == "WinNorm":
        if data_label == "labeled":
            ls_thresh = np.logspace(2.5, 4, 10)
        else:
            ls_thresh = np.logspace(3,  5, 10)
    elif statistics_type == "WinAR":
        if data_label == "labeled":
            ls_thresh = np.logspace(-2, 1, 10)
        else:
            ls_thresh = np.logspace(-2, 2, 10)
    elif statistics_type == "NPFocus":
        if data_label == "labeled":
            ls_thresh = np.logspace(2, np.log10(2846), 12)  # 3.45
        else:
            ls_thresh = np.logspace(2.5, 4.2, 12)
    elif statistics_type == "CUSUMraw":
        if data_label == "labeled":
            ls_thresh = np.linspace(1.5, 4.53, 10)
        else:
            ls_thresh = np.linspace(1.5, 4.53, 10)
    elif statistics_type == "EWMA":
        if data_label == "labeled":
            ls_thresh = np.linspace(0.5, 1.7, 10)
        else:
            ls_thresh = np.linspace(0.1, 1.7, 10)
    else:
        raise ValueError(
            f"Unknown statistic_type={statistics_type}.")
    if statistics_type in ["WinL1", "WinNorm", "WinAR"]:
        algo = rpt.Window(
            width=window_size,
            model=dc_rpt_models[statistics_type])
    else:
        algo = None

    # Prepare the output directory
    dir_name = f"{dir_results}/{statistics_type}estARL"
    os.makedirs(dir_name, exist_ok=True)

    # Load the pre-saved datasets
    dataset_files = sorted(glob.glob(
        f"{dir_data}/{name_process}_{data_label}_{mode}.pkl"))
    if not dataset_files:
        raise FileNotFoundError(
            f"No dataset file found in {dir_data}/{name_process}_{data_label}_{mode}.pkl. Please run WISDMactitracker.ipynb first or check the dataset paths.")
    assert len(dataset_files) == 1
    print(
        f"Found {len(dataset_files)} dataset files in {dir_data}")

    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* #
    # Start processing
    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* #
    it_dataset_path = dataset_files[0]
    with open(it_dataset_path, 'rb') as f:
        data = pickle.load(f)
        raw_data = data['feature_sequences']
        changepoints = data['changepoints']
        durations = data['durations']

    for thresh in tqdm(ls_thresh):
        print(f"Thresh:   {thresh}")
        ls_preds = []
        ls_cps = []
        ls_durations = []

        for it_idx in range(len(changepoints)):
            # Get the raw data and changepoint for this index
            # np.ndarray with shape [duration, num_features]
            it_raw_data = raw_data[it_idx]
            it_cp = changepoints[it_idx]
            if len(it_raw_data) < 2:
                print(
                    f"Skipping sample {it_idx} with length {len(it_raw_data)} < 2.")
                continue

            # Generate predictions
            if statistics_type in ["WinL1", "WinNorm", "WinAR"]:
                # Minus 1 because `predict` is 1-based
                if len(it_raw_data) < window_size:
                    it_pred = - 1
                else:
                    it_pred = algo.fit(it_raw_data).predict(
                        pen=thresh)[0] - 1  # scalar
            elif statistics_type in ["NPFocus"]:

                if len(it_raw_data) < window_size:
                    it_pred = - 1
                else:
                    # V2
                    qs = [None] * it_raw_data.shape[1]
                    for d in range(it_raw_data.shape[1]):
                        qs[d] = [np.quantile(it_raw_data[:window_size, d], q) for q in [
                            0.25, 0.5, 0.75]]
                    detectors = [NPFocus(qs_d) for qs_d in qs]
                    for t, y in enumerate(it_raw_data):
                        # update each 1-D NPFocus
                        s = 0.0
                        for d, det in enumerate(detectors):
                            det.update(y[d])
                            # det.statistic() is a vector of length len(qs_d); sum to get one score per dim
                            s += np.sum(det.statistic())

                        if s > thresh:
                            break
                    it_pred = t

            elif statistics_type == "CUSUMraw":
                model = CUSUM(
                    k=0.25, h=thresh, burnin=window_size,
                    mu=0., sigma=1.)
                normalized_it_raw_data = np.linalg.norm(it_raw_data, axis=1)
                model.process(normalized_it_raw_data)
                preds = model.changepoints
                if len(preds) > 0:
                    it_pred = preds[0]
                else:
                    it_pred = - 1
            elif statistics_type == "EWMA":
                normalized_it_raw_data = np.linalg.norm(it_raw_data, axis=1)
                model = EWMA(
                    r=0.1, L=thresh, burnin=window_size,
                    mu=0., sigma=1.)
                model.process(normalized_it_raw_data)
                preds = model.changepoints
                if len(preds) > 0:
                    it_pred = preds[0]
                else:
                    it_pred = - 1
            else:
                raise ValueError(
                    f"Unknown statistics_type={statistics_type}.")

            # Store the results
            ls_preds.append(it_pred)
            ls_cps.append(it_cp)
            ls_durations.append(len(it_raw_data))

        # All done
        preds_all = np.array(ls_preds)  # [num_samples, ]
        cps_all = np.array(ls_cps)  # [num_samples,]
        durations_all = np.array(
            ls_durations)  # [num_samples,]
        max_duration = np.max(durations_all)
        num_samples = len(preds_all)

        # Naive ARL
        naiARL, naisterr, naieffective_num_samples = calc_naive_ARL(
            preds_all, cps_all, flag_less_biased=False, flag_verbose=False)
        it_filename = f"{dir_name}/ARL_nai_thr{thresh:.7f}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(
            it_filename,
            ARL=naiARL, sterr=naisterr, effective_num_samples=naieffective_num_samples, num_all_samples=num_samples)
        print(f"File saved: {it_filename}")
        print(f"naiARL: {naiARL} ± {naisterr}")

        # Less-biased ARL
        lbARL, lbsterr, lbeffective_num_samples = calc_naive_ARL(  # lbARL can be nan if no false alarms or no cp=inf samples
            preds_all, cps_all, flag_less_biased=True, flag_verbose=False)
        it_filename = f"{dir_name}/ARL_lb__thr{thresh:.7f}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(
            it_filename,
            ARL=lbARL, sterr=lbsterr, effective_num_samples=lbeffective_num_samples, num_all_samples=num_samples)
        print(f"File saved: {it_filename}")
        print(f"lbARL : {lbARL} ± {lbsterr}")

        # KME-ARL
        kmeARL, kmesterr, kmeeffective_num_samples = calc_KME(
            preds_all, cps_all, duration=None, duration_array=durations_all, flag_verbose=False)
        it_filename = f"{dir_name}/ARL_kme_thr{thresh:.7f}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(
            it_filename,
            ARL=kmeARL, sterr=kmesterr, effective_num_samples=kmeeffective_num_samples, num_all_samples=num_samples)
        print(f"File saved: {it_filename}")
        print(f"kmeARL: {kmeARL} ± {kmesterr}")
