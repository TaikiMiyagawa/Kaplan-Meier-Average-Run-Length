# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
We assume that the detection delay is independent of the changepoint time to save computation time; 
otherwise we need to use extremely large durations for p=0.001.
"""

import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from metrics import calc_combined_mean_and_sem, calc_naive_ADD
from save_dataset_Gauss import generate_gaussian_LLRs
from statistic_tools import (LLRs_to_CLLRs, calc_all_CUSUMstatistics,
                             calc_all_logGSRstatistics_v2, first_flip)

name_process = "Gauss"

# =========================================================
# Calc true ADD
# =========================================================
# *************** User-defined parameters ********************** #
dir_result = f"./results/{name_process}"
statistic_type = ["GSR", "CUSUM"][1]
cp_type = ["geometric", "uniform"][1]  # "geometric" or "uniform"

if cp_type == "geometric":
    p: float = [0.001, 0.1, 0.25][0]  # geometric distribution parameter
elif cp_type == "uniform":
    p = 1.  # Assumes detection delay is independent of the changepoint time

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
        # Not used for ARL-ADD curve
        6.,  # 14
        7.,  # 15
        8.,  # 16
        9.,  # 17
        10.,  # 18
        12.,  # 19
        15.,  # 20
        20.,  # 21
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
        # Not used for ARL-ADD curve
        4.,  # 11
        5.,  # 12
        6.,  # 13
        7.,  # 14
        8.,  # 15
        12.,  # 16
        15.,  # 17
        20.,  # 18
    ]

if cp_type == "geometric":
    if p == 0.001:
        ls_duration = [
            4000,
        ] * len(ls_thresh)

    elif p == 0.1:
        ls_duration = [200,] * len(ls_thresh)

    elif p == 0.25:
        ls_duration = [150,] * len(ls_thresh)

    else:
        raise ValueError(f"Unknown p={p}.")
elif cp_type == "uniform":
    # Assumes detection delay is independent of the changepoint time
    ls_duration = [1000,] * len(ls_thresh)
else:
    raise ValueError(f"Unknown cp_type={cp_type}.")

# *************** User-defined parameters ********************** #


torch.set_num_threads(num_threads)
dir_name = f"{dir_result}/{statistic_type}trueADD_p{p}{cp_type}"
os.makedirs(dir_name, exist_ok=True)
omega = 0.

assert len(ls_thresh) == len(ls_duration)
for it_cnt, (thresh, duration) in enumerate(zip(ls_thresh, ls_duration)):  # thresh loop
    print(f"Thresh:   {thresh}")
    print(f"Duration: {duration}")
    ls_means = []
    ls_sems = []
    ls_ns = []
    ls_num_all_samples = []

    for it_i in tqdm(range(num_iterations)):  # more samples loop
        if it_i % 10 == 0:
            print(f"Iteration: {it_i}/{num_iterations}")

        # Generate changepoints
        if cp_type == "uniform":
            dummy_duration = int(duration / 2)  # to avoid overruns
            cps_all = torch.randint(
                0, dummy_duration,
                size=[num_iterations_per_loop * batch_size])  # [size]
        elif cp_type == "geometric":
            m = torch.distributions.geometric.Geometric(
                torch.tensor([p] * (num_iterations_per_loop * batch_size)))
            cps_all = m.sample().int()  # [size]
        cps_all = torch.where(
            cps_all >= duration, torch.inf, cps_all)  # [size]

        if sum(cps_all == torch.inf) > 0:
            print(
                f"WARNING: Num cps=inf = {sum(cps_all == torch.inf)} > 0. Consider increadsing duration (thresh={thresh}, duration={duration}). This warning can be ignored if you assume that detection delay is independent of the changepoint time; however, note that the probability that `overruns` > 1 occurs is now non-negligible.")

            """ 
            Note:
            The `overruns` means the number of sequences with changepoint < inf and pred = -1 (there is a changepoint but no alarm was raised), which cause a bias to ADD. This must be avoided when computing (simulating) true ADD.
            """

        # Stat loop
        ls_preds = []
        thresh_array = torch.full((batch_size, duration), thresh)  # [B, T]
        for it_per_loop in range(num_iterations_per_loop):  # mini-batch loop
            it_cps = cps_all[
                batch_size * it_per_loop: batch_size * (it_per_loop + 1)]
            it_LLRs = generate_gaussian_LLRs(
                batch_size=batch_size,
                duration=duration,
                changepoint=it_cps,
                mu1=0., mu2=0.1, sigma1=0.1, sigma2=0.1)  # [B, T, 2, 2]
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

            if sum(predictions == -1) > 0:
                print(
                    f"WARNING: Num pred=-1 = {sum(predictions==-1)} > 0. Consider increasing duration (thresh={thresh}, duration={duration}). This warning is not critical; however, note that the probability that `overruns` > 1 occurs is now non-negligible.")

            # Error check
            overruns = sum(torch.logical_and(
                predictions > it_cps, predictions == -1))
            if overruns > 0:
                raise ValueError(
                    f"Num overruns = {overruns} > 0. This is an critical error, as we are computing unbiased true ADD. Consider increasing duration (thresh={thresh}, duration={duration}).")
            if sum(torch.logical_and(predictions == -1, it_cps == -1)) > 0:
                raise ValueError(
                    f"Num `pred = -1 (no alarm) and cps=inf = {sum(predictions==-1 and it_cps==np.inf)}` > 0. This is an critical error, as we are computing unbiased true ADD. Consider increasing duration (thresh={thresh}, duration={duration}).")

            ls_preds.append(predictions)

        preds_all = np.concatenate(ls_preds)  # [num_samples]
        num_samples = preds_all.shape[0]

        ADD, sterr, effective_num_samples = calc_naive_ADD(
            preds_all, cps_all.numpy())

        if np.isnan(ADD) or np.isnan(sterr):
            # All alarms are false alarms.
            print(
                f"Got ADD=np.nan. All alarms were false alarms. Consider increasing threshold (thresh={thresh}, duration={duration}, p={p}).")
            continue
        else:
            ls_means.append(ADD)
            ls_sems.append(sterr)
            ls_ns.append(effective_num_samples)
            ls_num_all_samples.append(num_samples)
        print()

    if len(ls_means) == 0:
        raise ValueError(
            f"All ADDs are np.nan for the loop with thresh={thresh}, duration={duration}, p={p}. Consider increasing threshold (thresh={thresh}, duration={duration}, p={p}).")

    means = np.array(ls_means)
    sems = np.array(ls_sems)
    ns = np.array(ls_ns)
    num_all_samples = np.array(ls_num_all_samples)
    combined_mean, combined_sem = calc_combined_mean_and_sem(
        means, sems, ns)
    it_filename = f"{dir_name}/ADD_thr{thresh}_dur{duration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(
        it_filename,
        means=means, sems=sems, ns=ns, num_all_samples=num_all_samples,
        combined_mean=combined_mean, combined_sem=combined_sem)
    print(f"File saved: {it_filename}")
    print(f"ADD: {combined_mean} ± {combined_sem}")
    # num overruns = 0 is must; otherwise some samples are like "post-change overruns".
    # num samples for computing ADD < num samples is not a problem (lots of samples are wasted and sterr becomes large, tho).
    # Max pred < Max cp is not a problem but the model is likely to be trigger-happy.
