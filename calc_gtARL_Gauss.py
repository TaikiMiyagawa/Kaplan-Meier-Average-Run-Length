# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from metrics import calc_combined_mean_and_sem, calc_naive_ARL
from save_dataset_Gauss import generate_gaussian_LLRs
from statistic_tools import (LLRs_to_CLLRs, calc_all_CUSUMstatistics,
                             calc_all_logGSRstatistics_v2, first_flip)

name_process = "Gauss"

# =========================================================
# Calc true ARL under approximately infinite duration (>> hitting times)
# =========================================================
# *************** User-defined parameters ********************** #
statistic_type: str = ["GSR", "CUSUM"][0]
dir_result = f"./results/{name_process}"

batch_size = 100  # too large -> OOM
num_iterations_per_loop = 100  # too large -> OOM
num_iterations = 100  # no worry about OOM
num_threads = 2  # number of threads for torch

if statistic_type == "GSR":
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
        # > 6.5 requres length>8000 (and gives ARL > 1200), leading to long time and poitential OOM
    ]
    ls_duration = [
        100,  # 0
        100,  # 1
        300,  # 2
        500,  # 3
        500,  # 4
        1000,  # 5
        2000,  # 6
        2000,  # 7
        2000,  # 8
        4000,  # 9
        4000,  # 10
        4000,  # 11
        5000,  # 12
        6000,  # 13
    ]
elif statistic_type == "CUSUM":
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
    ls_duration = [
        100,  # 0
        200,  # 1
        320,  # 2
        700,  # 3
        900,  # 4
        1200,  # 5
        1200,  # 6
        3000,  # 7
        3000,  # 8
        3000,  # 9
        5000,  # 10
    ]
else:
    raise ValueError(
        f"Unknown statistic_type={statistic_type}.")
# *************** User-defined parameters ********************** #

omega = 0.  # non-negative const for GSR statistics
dir_name = f"{dir_result}/{statistic_type}trueARL"
os.makedirs(dir_name, exist_ok=True)
torch.set_num_threads(num_threads)

assert len(ls_thresh) == len(ls_duration)
for it_cnt, (thresh, duration) in enumerate(zip(ls_thresh, ls_duration)):  # thresh loop
    print(f"Thresh:   {thresh}")
    print(f"Duration: {duration}")
    ls_means = []
    ls_sems = []
    ls_ns = []
    ls_num_all_samples = []

    for it_i in tqdm(range(num_iterations)):  # more samples loop
        # Stat loop
        ls_preds = []
        thresh_array = torch.full((batch_size, duration), thresh)  # [B, T]
        for _ in range(num_iterations_per_loop):  # mini-batch loop
            it_LLRs = generate_gaussian_LLRs(
                batch_size=batch_size, duration=duration, changepoint=duration,  # no changepoint
                mu1=0., mu2=0.1, sigma1=0.1, sigma2=0.1,)  # [B, T, 2, 2]
            it_CLLRs = LLRs_to_CLLRs(it_LLRs)  # [B, T, 2, 2]
            if statistic_type == "GSR":
                it_stat = calc_all_logGSRstatistics_v2(
                    it_CLLRs, omega=omega)  # [B, T, 2, 2] -> [B, T]
            elif statistic_type == "CUSUM":
                it_stat = calc_all_CUSUMstatistics(
                    it_CLLRs)  # [B, T, 2, 2] -> [B, T]
            else:
                raise ValueError(f"Unknown statistic_type={statistic_type}.")

            predictions = first_flip(
                it_stat,
                thresh_array,
                eps=0.0)  # [B]

            ls_preds.append(predictions)

            # Error check for overrun
            overrun = np.sum(predictions.numpy() == -1)
            if overrun > 0:
                raise ValueError(
                    f"Overrun detected: {overrun} samples. Consider increasing duration (thresh: {thresh}, duration: {duration}).")

        preds_all = np.concatenate(ls_preds)  # [num_samples]
        num_samples = preds_all.shape[0]

        if np.min(preds_all) < 0:
            raise ValueError(
                f"pred=-1 detected, meaning an overrun. Consider increasing duration (thresh: {thresh}, duration: {duration}).")

        ARL, sterr, effective_num_samples = calc_naive_ARL(
            preds_all, np.full_like(preds_all.astype(float), np.inf), flag_less_biased=True, flag_verbose=False)
        ls_means.append(ARL)
        ls_sems.append(sterr)
        ls_ns.append(effective_num_samples)
        ls_num_all_samples.append(num_samples)

    means = np.array(ls_means)
    sems = np.array(ls_sems)
    ns = np.array(ls_ns)
    num_all_samples = np.array(ls_num_all_samples)
    combined_mean, combined_sem = calc_combined_mean_and_sem(
        means, sems, ns)
    it_filename = f"{dir_name}/ARL_thr{thresh}_dur{duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(
        it_filename,
        means=means, sems=sems, ns=ns, num_all_samples=num_all_samples,
        combined_mean=combined_mean, combined_sem=combined_sem)
    print(f"File saved: {it_filename}")
    print(f"ARL: {combined_mean} ± {combined_sem}\n")
