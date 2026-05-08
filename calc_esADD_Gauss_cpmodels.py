# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" Ignore IntegrationWarning. """
import glob
import os
from datetime import datetime

import numpy as np
import ruptures as rpt
import torch
from changepoint_online import NPFocus
from tqdm import tqdm

from metrics import calc_naive_ADD
from ocpdet import CUSUM, EWMA
from statistic_tools import calc_KME_ADD

name_process = "Gauss"

torch.manual_seed(0)
np.random.seed(0)

# =========================================================
# Calc estimated ADD
# =========================================================
# *************** User-defined parameters ********************** #
# Directory where the dataset is saved
dir_data = "./dataset"

# Directory where the results will be saved
dir_results = f"./results/{name_process}"
# *********************************"**************************** #

ls_org_duration = [300, 1000]  # max duration to consider
ls_statistics_type = [
    "WinL1", "WinNorm", "WinAR", "NPFocus", "CUSUMraw", "EWMA"]
ls_cp_method = ["uniform"]  # "geometric" or "uniform"
ls_irregular_length_level = [0, 2]  # 0: no, 1: moderate, 2: intense
ls_num_samples = [10000]
window_size = 30
dc_rpt_models = {
    "WinL1": "l1",
    "WinNorm": "normal",
    "WinAR": "ar",
}
for org_duration in ls_org_duration:
    for statistics_type in ls_statistics_type:
        for cp_method in ls_cp_method:
            for irregular_length_level in ls_irregular_length_level:
                for num_samples in ls_num_samples:
                    print(f"Processing: org_duration={org_duration}, "
                          f"statistics_type={statistics_type}, "
                          f"cp_method={cp_method}, "
                          f"irregular_length_level={irregular_length_level}, "
                          f"num_samples={num_samples}")

                    if cp_method == "geometric":
                        # geometric distribution parameter p
                        ls_p = [0.001, 0.1, 0.25]
                    elif cp_method == "uniform":
                        # how many with-cp samples
                        ls_p = [0.5,]
                    else:
                        raise ValueError(f"Unknown cp_method={cp_method}.")

                    for p in ls_p:
                        print(f"Using p={p}.")

                        # Set the model and parameters
                        if statistics_type == "WinL1":
                            # WinL1: ARL 100-999
                            ls_thresh = np.logspace(-4, 0, 10)
                        elif statistics_type == "WinNorm":
                            # WinNorm: ARL 57-900
                            ls_thresh = np.linspace(0.1, 10**1, 10)
                        elif statistics_type == "WinAR":
                            # WinAR: ARL 30-900
                            ls_thresh = np.linspace(10**(-2.5), 10**(-0.5), 10)
                        elif statistics_type == "NPFocus":
                            # NPFocus: ARL 10-800
                            ls_thresh = np.linspace(7, 17, 12)
                        elif statistics_type == "CUSUMraw":
                            ls_thresh = np.linspace(1, 9, 10)
                        elif statistics_type == "EWMA":
                            ls_thresh = np.linspace(0.5, 3, 10)
                        else:
                            raise ValueError(
                                f"Unknown statistic_type={statistics_type}.")
                        if statistics_type in ["WinL1", "WinNorm", "WinAR"]:
                            algo = rpt.Window(
                                width=window_size,
                                model=dc_rpt_models[statistics_type])
                        else:
                            algo = None

                        # Prepare output directory
                        dir_name = f"{dir_results}/{statistics_type}estADD_p{p}{cp_method}_dur{org_duration}_{num_samples}samples_irreg{irregular_length_level}"
                        os.makedirs(dir_name, exist_ok=True)

                        # Load the pre-saved datasets
                        dataset_dir = f"{dir_data}/{name_process}_p{p}{cp_method}-dur1000"
                        dataset_files = sorted(
                            glob.glob(f"{dataset_dir}/{name_process}_p{p}{cp_method}-dur1000_*.pth"))
                        if not dataset_files:
                            raise FileNotFoundError(
                                f"No dataset files found in {dataset_dir}. Please run save_dataset_{name_process}.py first or check the dataset paths.")
                        print(
                            f"Found {len(dataset_files)} dataset files in {dataset_dir}")

                        # Define max_duration_subtraction based on irregular_length_level
                        if irregular_length_level == 0:
                            max_duration_subtraction = 0.0
                        elif irregular_length_level == 1:
                            max_duration_subtraction = 0.5 * org_duration
                        elif irregular_length_level == 2:
                            max_duration_subtraction = 0.9 * org_duration
                        else:
                            raise ValueError(
                                f"Unknown irregular_length_level={irregular_length_level}.")

                        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* #
                        # Start processing
                        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* #
                        for thresh in tqdm(ls_thresh):
                            print(f"Thresh:   {thresh}")
                            print(f"Duration: {org_duration}")
                            ls_preds = []
                            ls_cps = []
                            ls_durations = []
                            it_cnt = 0
                            for it_dataset_file in dataset_files:
                                # Process a subset of dataset files
                                data = torch.load(it_dataset_file)
                                # [*, T]
                                raw_data: torch.Tensor = data["raw_sequences"]
                                # [*]
                                changepoints: torch.Tensor = data["changepoints"]

                                # Process each sample in the dataset
                                for it_idx in range(changepoints.shape[0]):
                                    # Generate a random length for irregular lengths
                                    it_duration_subtraction = int(
                                        max_duration_subtraction * np.random.rand())
                                    duration: int = org_duration - it_duration_subtraction
                                    ls_durations.append(duration)

                                    # Modify statistics and changepoints for irregular lengths
                                    # [duration]
                                    it_raw_data = raw_data[
                                        it_idx, :duration].numpy()
                                    it_cp = changepoints[it_idx]
                                    it_cp = it_cp.item()  # Convert to scalar
                                    # Ensure cp is within the duration
                                    it_cp = np.inf if it_cp >= duration else it_cp

                                    # Generate predictions
                                    if statistics_type in ["WinL1", "WinNorm", "WinAR"]:
                                        # Minus 1 because `predict` is 1-based
                                        it_pred = algo.fit(it_raw_data).predict(
                                            pen=thresh)[0] - 1  # scalar
                                        if it_pred == duration - 1:
                                            # If the prediction is the last index, set it to -1
                                            it_pred = -1
                                    elif statistics_type in ["NPFocus"]:
                                        # Create and use NPFocus detector
                                        quantiles = [np.quantile(it_raw_data[:window_size], q) for q in [
                                            0.25, 0.5, 0.75]]
                                        detector = NPFocus(quantiles)
                                        for y in it_raw_data:
                                            detector.update(y)
                                            # we can sum the statistics over to get a detection
                                            # see  (Romano, Eckley, and Fearnhead 2024) for more details
                                            if np.sum(detector.statistic()) > thresh:
                                                break
                                        changepoint_info = detector.changepoint()
                                        # Minus one because `detector` is 1-based
                                        it_pred = changepoint_info["stopping_time"] - 1
                                        if it_pred == duration - 1:
                                            # If the prediction is the last index, set it to -1
                                            it_pred = -1
                                    elif statistics_type == "CUSUMraw":
                                        model = CUSUM(
                                            k=0.25, h=thresh, burnin=window_size,
                                            mu=0., sigma=1.)
                                        model.process(it_raw_data)
                                        preds = model.changepoints
                                        if len(preds) > 0:
                                            it_pred = preds[0]
                                        else:
                                            it_pred = - 1
                                    elif statistics_type == "EWMA":
                                        model = EWMA(
                                            r=0.1, L=thresh, burnin=window_size,
                                            mu=0., sigma=1.)
                                        model.process(it_raw_data)
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

                                    # Break if we have enough samples
                                    it_cnt += 1
                                    if it_cnt >= num_samples:
                                        break
                                if it_cnt >= num_samples:
                                    print(
                                        f"Processed {it_cnt} samples for thresh {thresh}.")
                                    break

                            # All done
                            preds_all = np.array(ls_preds)  # [num_samples,]
                            cps_all = np.array(ls_cps)  # [num_samples,]
                            durations_all = np.array(
                                ls_durations)  # [num_samples,]
                            max_duration = np.max(durations_all)

                            # Naive ADD
                            naiADD, sterr, effective_num_samples = calc_naive_ADD(
                                preds_all, cps_all, flag_verbose=False)
                            it_filename = f"{dir_name}/ADD_nai_thr{thresh:.7f}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                            np.savez(
                                it_filename,
                                ADD=naiADD, sterr=sterr, effective_num_samples=effective_num_samples,
                                num_all_samples=num_samples)
                            print(f"File saved: {it_filename}")
                            print(f"naiADD: {naiADD} ± {sterr}")

                            # Calculate KME ADD
                            kmeADD, kmesterr, kmeeffective_num_samples = calc_KME_ADD(
                                preds_all, cps_all, duration=None, duration_array=durations_all, flag_verbose=False)
                            it_filename = f"{dir_name}/ADD_kme_thr{thresh:.7f}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                            np.savez(
                                it_filename,
                                ADD=kmeADD, sterr=kmesterr, effective_num_samples=kmeeffective_num_samples,
                                num_all_samples=num_samples)
                            print(f"File saved: {it_filename}")
                            print(f"kmeADD: {kmeADD} ± {kmesterr}")
