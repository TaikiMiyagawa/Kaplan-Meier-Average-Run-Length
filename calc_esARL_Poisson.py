# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

""" Ignore IntegrationWarning. """
import glob
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from metrics import calc_naive_ARL
from statistic_tools import calc_KME, first_flip

name_process = "Poisson"

torch.manual_seed(0)
np.random.seed(0)

# =========================================================
# Estimate ARL
# =========================================================
# *************** User-defined parameters ********************** #
# Directory where the dataset is saved
dir_data = "./dataset"

# Directory where the results will be saved
dir_results = f"./results/{name_process}"
# ************************************************************** #

ls_org_duration = [100, 300, 500, 1000]  # max duration to consider
ls_statistics_type = ["GSR", "CUSUM"]
ls_cp_method = ["geometric", "uniform"]  # "geometric" or "uniform"
ls_irregular_length_level = [0, 1, 2]  # 0: no, 1: moderate, 2: intense
ls_num_samples: int = [100, 1000, 10000]  # number of iterations to run

for org_duration in ls_org_duration:
    for statistics_type in ls_statistics_type:
        for cp_method in ls_cp_method:
            for irregular_length_level in ls_irregular_length_level:
                for num_samples in ls_num_samples:
                    print(f"Processing: duration={org_duration}, "
                          f"statistics_type={statistics_type}, "
                          f"cp_method={cp_method}, "
                          f"irregular_length_level={irregular_length_level}, "
                          f"num_samples={num_samples}")

                    if cp_method == "geometric":
                        # geometric distribution parameter p
                        ls_p = [0.001, 0.1, 0.25]
                    elif cp_method == "uniform":
                        ls_p = [0.1, 0.5, 0.9]  # how many with-cp samples
                    else:
                        raise ValueError(f"Unknown cp_method={cp_method}.")

                    for p in ls_p:
                        print(f"Processing p={p}...")

                        if statistics_type == "GSR":
                            ls_thresh = [
                                0.5,  # 0
                                1.,  # 1
                                1.5,  # 2
                                2.,  # 3
                                2.5,  # 4
                                3.,  # 5
                                3.5,  # 6
                                4.,  # 7
                                4.2,  # 8
                                4.5,  # 9
                                4.6,  # 10
                                4.8,  # 11
                                5.,  # 12
                                5.5,  # 13
                            ]
                        elif statistics_type == "CUSUM":
                            ls_thresh = [
                                0.3,  # 0
                                0.5,  # 1
                                1.,  # 2
                                1.5,  # 3
                                1.8,  # 4
                                2.,  # 5
                                2.2,  # 6
                                2.5,  # 7
                                2.8,  # 8
                                3.,  # 9
                                3.5,  # 10
                            ]
                        else:
                            raise ValueError(
                                f"Unknown statistic_type={statistics_type}.")

                        # Prepare the output directory
                        dir_name = f"{dir_results}/{statistics_type}estARL_p{p}{cp_method}_dur{org_duration}_{num_samples}samples_irreg{irregular_length_level}"
                        os.makedirs(dir_name, exist_ok=True)

                        # Load the pre-saved datasets
                        dataset_dir = f"{dir_data}/{name_process}_p{p}{cp_method}-dur1000"
                        dataset_files = sorted(glob.glob(
                            f"{dataset_dir}/{name_process}_p{p}{cp_method}-dur1000_*.pth"))
                        if not dataset_files:
                            raise FileNotFoundError(
                                f"No dataset files found in {dataset_dir}. Please run save_dataset_{name_process}.py first or check the dataset paths.")
                        print(
                            f"Found {len(dataset_files)} dataset files in {dataset_dir}")

                        # Define max_duration_subtraction
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
                            for it_dataset_path in dataset_files:  # more samples loop
                                # Process a subset of dataset files
                                data = torch.load(it_dataset_path)
                                statistics: torch.Tensor
                                if statistics_type == "GSR":
                                    # [*, T]
                                    statistics = data["gsr_all"]
                                elif statistics_type == "CUSUM":
                                    # [*, T]
                                    statistics = data["cusum_all"]
                                else:
                                    raise ValueError(
                                        f"Unknown statistics_type={statistics_type}.")
                                # [*]
                                changepoints: torch.Tensor = data["changepoints"]
                                # [*, org_duration]
                                statistics = statistics[:, :org_duration]

                                for it_idx in range(statistics.shape[0]):
                                    # Generate a random length
                                    it_duration_subtraction = int(
                                        max_duration_subtraction * np.random.rand())
                                    duration: int = org_duration - it_duration_subtraction
                                    ls_durations.append(duration)

                                    # Modify statistics and changepoints for irregular lengths
                                    # [1, duration]
                                    it_stat = statistics[it_idx:it_idx +
                                                         1, :duration]
                                    it_cp = changepoints[it_idx]
                                    it_cp = it_cp.item()  # Convert to scalar
                                    # Ensure cp is within the duration
                                    it_cp = np.inf if it_cp >= duration else it_cp

                                    # Generate predictions
                                    thresh_array = torch.full(
                                        (1, duration), thresh)  # [1, T]
                                    it_pred: torch.Tensor = first_flip(
                                        it_stat, thresh_array, eps=0.0)  # [1,]
                                    it_pred = it_pred.item()

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
                            preds_all = np.array(ls_preds)  # [num_samples, ]
                            cps_all = np.array(ls_cps)  # [num_samples,]
                            durations_all = np.array(
                                ls_durations)  # [num_samples,]
                            max_duration = np.max(durations_all)

                            # Naive ARL
                            naiARL, naisterr, naieffective_num_samples = calc_naive_ARL(
                                preds_all, cps_all, flag_less_biased=False, flag_verbose=False)
                            it_filename = f"{dir_name}/ARL_nai_thr{thresh}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                            np.savez(
                                it_filename,
                                ARL=naiARL, sterr=naisterr, effective_num_samples=naieffective_num_samples, num_all_samples=num_samples)
                            print(f"File saved: {it_filename}")
                            print(f"naiARL: {naiARL} ± {naisterr}")

                            # Less-biased ARL
                            lbARL, lbsterr, lbeffective_num_samples = calc_naive_ARL(  # lbARL can be nan if no false alarms or no cp=inf samples
                                preds_all, cps_all, flag_less_biased=True, flag_verbose=False)
                            it_filename = f"{dir_name}/ARL_lb__thr{thresh}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                            np.savez(
                                it_filename,
                                ARL=lbARL, sterr=lbsterr, effective_num_samples=lbeffective_num_samples, num_all_samples=num_samples)
                            print(f"File saved: {it_filename}")
                            print(f"lbARL : {lbARL} ± {lbsterr}")

                            # KME-ARL
                            kmeARL, kmesterr, kmeeffective_num_samples = calc_KME(
                                preds_all, cps_all, duration=None, duration_array=durations_all, flag_verbose=False)
                            it_filename = f"{dir_name}/ARL_kme_thr{thresh}_dur{max_duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
                            np.savez(
                                it_filename,
                                ARL=kmeARL, sterr=kmesterr, effective_num_samples=kmeeffective_num_samples, num_all_samples=num_samples)
                            print(f"File saved: {it_filename}")
                            print(f"kmeARL: {kmeARL} ± {kmesterr}")
