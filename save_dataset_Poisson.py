# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from statistic_tools import (LLRs_to_CLLRs, calc_all_CUSUMstatistics,
                             calc_all_logGSRstatistics_v2)


def generate_poisson_LLRs(batch_size, duration, changepoint: Union[int, float, torch.Tensor],
                          lambda1, lambda2, flag_output_raw_data: bool = False):
    """ Generate LLR sequence of i.i.d. Poisson distributions.

    Args:
        batch_size (int): Batch size.
        duration (int): Duration of LLR sequence.
        changepoint (int, float, or Tensor): Changepoint index or tensor of indices [batch_size].
        lambda1: Rate parameter before change (scalar or 1D tensor of length D).
        lambda2: Rate parameter after change (scalar or 1D tensor of length D).
        flag_output_raw_data (bool): Whether to output raw data samples.
    Returns:
        If flag_output_raw_data:
            x: raw samples (shape [B, T] for univariate or [B, T, D] for multivariate)
            result: LLR matrices (shape [B, T, 2, 2])
        Else:
            result: LLR matrices (shape [B, T, 2, 2])
    """
    # Convert to tensors
    lambda1 = torch.as_tensor(lambda1, dtype=torch.float32)
    lambda2 = torch.as_tensor(lambda2, dtype=torch.float32)

    # Ensure parameters are valid (Poisson requires positive rates)
    if torch.any(lambda1 <= 0) or torch.any(lambda2 <= 0):
        raise ValueError("Poisson rate parameters must be positive")

    # Determine dimensionality
    if lambda1.ndim == 0:
        D = 1
        lambda1_vec = lambda1.view(1)
        lambda2_vec = lambda2.view(1)
    elif lambda1.ndim == 1:
        D = lambda1.shape[0]
        lambda1_vec = lambda1
        lambda2_vec = lambda2
        # Check that lambda2 has the same shape
        if lambda2.shape[0] != D:
            raise ValueError("lambda1 and lambda2 must have the same shape")
    else:
        raise ValueError("lambda1 must be a scalar or 1D tensor")

    # Create Poisson distributions
    dist1 = torch.distributions.Poisson(lambda1_vec)
    dist2 = torch.distributions.Poisson(lambda2_vec)

    # Sample raw data
    x1 = dist1.sample((batch_size, duration))  # [B, T, D]
    x2 = dist2.sample((batch_size, duration))  # [B, T, D]

    # Handle changepoint mask
    if isinstance(changepoint, (int, float)):
        if changepoint < 0:
            x = x2
        elif changepoint >= duration:
            x = x1
        else:
            x = x1.clone()
            mask = torch.arange(duration).unsqueeze(
                0).expand(batch_size, duration) >= changepoint
            x[mask] = x2[mask]
    elif isinstance(changepoint, torch.Tensor):
        assert changepoint.shape[0] == batch_size, "changepoint tensor must have shape [batch_size]"
        time_idx = torch.arange(duration).unsqueeze(
            0).expand(batch_size, duration)
        cp_expanded = changepoint.unsqueeze(1).expand(-1, duration)
        mask = time_idx >= cp_expanded
        x = torch.where(mask.unsqueeze(-1), x2, x1)
    else:
        raise ValueError("changepoint must be an int, float, or tensor.")

    # Compute log-likelihoods and LLRs
    # For Poisson: log_p(x|lambda) = x*log(lambda) - lambda - log(x!)
    # We can ignore log(x!) since it cancels out in the LLR
    log_p1 = x * torch.log(lambda1_vec) - lambda1_vec
    log_p2 = x * torch.log(lambda2_vec) - lambda2_vec

    # Sum over the feature dimension
    log_p1 = log_p1.sum(dim=-1)
    log_p2 = log_p2.sum(dim=-1)

    llr = log_p1 - log_p2  # [B, T]
    llr = torch.cumsum(llr, dim=1)  # i.i.d. sequence

    # Matrixize: [[0, llr], [-llr, 0]]
    result = torch.zeros(batch_size, duration, 2, 2, dtype=llr.dtype)
    result[:, :, 0, 1] = llr  # = log(pre-change) - log(post-change)
    result[:, :, 1, 0] = -llr  # = log(post-change) - log(pre-change)

    if flag_output_raw_data:
        # For univariate, squeeze last dim
        raw = x.squeeze(-1) if D == 1 else x
        return raw, result
    else:
        return result


def save_pth_dataset_Poisson(size, duration, cp_method, data_dir, p, lambda1=1, lambda2=5, omega=0.):
    """
    Save a dataset of raw sequence, LLRs, CLLRs, GSR statistics, and CUSUM statistics to a pth file.
    The dataset consists of LLRs generated from a Poisson distribution with specified parameters.
    The changepoint is determined by the cp_method parameter.
    cp = torch.inf means no changepoint.
    The dataset is saved in a .pth file with the specified filename.
    Args:
        size (int): Number of samples in the dataset.
        duration (int): Duration of each sample.
        data_dir (str): Directory to save the pth datasets.
        p (float): Probability parameter for geometric changepoint generation.
        cp_method (str): Method to generate changepoints, either "uniform" or "geometric".
        lambda1 (float): Rate parameter for the Poisson distribution before the changepoint.
        lambda2 (float): Rate parameter for the Poisson distribution after the changepoint.
        omega (float): Non-negative parameter for GSR statistic calculation.
    Returns:
        None
    """
    assert cp_method in ["uniform", "geometric"]
    per_dataset = 100  # Number of samples per dataset
    assert size % per_dataset == 0, "Size must be divisible by per_dataset."
    num_iterations = size // per_dataset

    # Define directory and filename
    dir_name = f"{data_dir}/Poisson_p{p}{cp_method}-dur{duration}"
    filename = f"{dir_name}/Poisson_p{p}{cp_method}-dur{duration}.pth"
    os.makedirs(dir_name, exist_ok=True)

    # Generate changepoints
    if cp_method == "uniform":
        cps = torch.randint(0, duration+1, size=[size])  # [size]
    elif cp_method == "geometric":
        m = torch.distributions.geometric.Geometric(torch.tensor([p] * size))
        cps = m.sample().int()  # [size]
    cps = torch.where(cps >= duration, torch.inf, cps)  # [size]

    # Generate LLRs, CLLRs, and GSR statistics
    print(f"Generating pth files...")
    for it_i in tqdm(range(num_iterations)):
        # Placeholder
        raw_sequences = torch.empty(
            (per_dataset, duration), dtype=torch.float32)
        LLRs = torch.empty((per_dataset, duration, 2, 2), dtype=torch.float32)
        CLLRs = torch.empty((per_dataset, duration, 2, 2), dtype=torch.float32)
        gsr_all = torch.empty((per_dataset, duration), dtype=torch.float32)
        cusum_all = torch.empty((per_dataset, duration), dtype=torch.float32)

        for i in range(per_dataset):
            serial_num = it_i * per_dataset + i
            if cps[serial_num] == torch.inf:
                changepoint = duration
            else:
                changepoint = int(cps[serial_num].numpy())

            raw_sequences[i], LLRs[i] = generate_poisson_LLRs(
                batch_size=1,
                duration=duration,
                changepoint=changepoint,
                lambda1=lambda1,
                lambda2=lambda2,
                flag_output_raw_data=True)  # [1, T], [1, T, 2, 2]
            CLLRs[i] = LLRs_to_CLLRs(LLRs[i:i+1])  # [1, T, 2, 2]
            gsr_all[i] = calc_all_logGSRstatistics_v2(
                CLLRs[i:i+1], omega=omega)  # [1, T, 2, 2] -> [T]
            cusum_all[i] = calc_all_CUSUMstatistics(
                CLLRs[i:i+1])  # [1, T, 2, 2] -> [T]

        # Save the dataset to a .pth file
        it_filename = filename.replace(
            ".pth", f"_{it_i+1}-{num_iterations}.pth")
        torch.save({
            "raw_sequences": raw_sequences,  # [size, T]
            "LLRs": LLRs,  # [size, T, 2, 2]
            "CLLRs": CLLRs,  # [size, T, 2, 2]
            "gsr_all": gsr_all,  # [size, T]
            "cusum_all": cusum_all,  # [size, T]
            "changepoints": cps[
                it_i * per_dataset: (it_i + 1) * per_dataset],  # [size]
        }, it_filename)
        print(f"Saved {it_filename}")


if __name__ == "__main__":
    torch.manual_seed(777)
    np.random.seed(777)
    torch.set_num_threads(4)

    data_dir = "./dataset"  # Directory to save the dataset

    for cp_method in ["geometric", "uniform"]:
        print(f"Generating dataset with cp_method={cp_method}...")
        if cp_method == "geometric":
            ls_p = [0.001, 0.1, 0.25]
        elif cp_method == "uniform":
            ls_p = [0.1, 0.5, 0.9]
        else:
            raise ValueError(f"Unknown cp_method={cp_method}.")

        for p in ls_p:

            save_pth_dataset_Poisson(
                size=10000,
                duration=1000,
                cp_method=cp_method,
                data_dir=data_dir,
                lambda1=1,
                lambda2=4,
                p=p,
                omega=0.0,
            )
