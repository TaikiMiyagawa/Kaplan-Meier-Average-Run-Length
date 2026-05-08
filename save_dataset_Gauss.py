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


def generate_random_LLRs(batch_size, duration, num_classes):
    LLRs = torch.randn(batch_size, duration, num_classes,
                       num_classes) * 10  # [B, T, K, K]
    LLRs_u = torch.triu(LLRs, diagonal=1)  # [B, T, K, K]
    LLRs_l = torch.transpose(LLRs_u, dim0=2, dim1=3)  # [B, T, K, K]
    LLRs = LLRs_u + LLRs_l  # [B, T, K, K]
    return LLRs


def generate_gaussian_LLRs(
    batch_size, duration, changepoint: Union[int, float, torch.Tensor],
        mu1, mu2, sigma1, sigma2, flag_output_raw_data: bool = False):
    """ Generate LLR sequence of i.i.d. Gaussian distributions.
    Supports univariate (scalar) and multivariate Gaussian.

    Args:
        batch_size (int): Batch size.
        duration (int): Duration of LLR sequence.
        changepoint (int, float, or Tensor): Changepoint index or tensor of indices [batch_size].
        mu1: Mean before change (scalar or 1D tensor of length D).
        mu2: Mean after change (scalar or 1D tensor of length D).
        sigma1: Std dev before change (scalar, 1D tensor, or DxD covariance matrix).
        sigma2: Std dev after change (scalar, 1D tensor, or DxD covariance matrix).
        flag_output_raw_data (bool): Whether to output raw data samples.
    Returns:
        If flag_output_raw_data:
            x: raw samples (shape [B, T] for univariate or [B, T, D] for multivariate)
            result: LLR matrices (shape [B, T, 2, 2])
        Else:
            result: LLR matrices (shape [B, T, 2, 2])
    """
    # Convert to tensors on correct device
    mu1 = torch.as_tensor(mu1, dtype=torch.float32)
    mu2 = torch.as_tensor(mu2,  dtype=torch.float32)
    sigma1 = torch.as_tensor(sigma1, dtype=torch.float32)
    sigma2 = torch.as_tensor(sigma2, dtype=torch.float32)

    # Determine dimensionality
    if mu1.ndim == 0:
        D = 1
        mu1_vec = mu1.view(1)
        mu2_vec = mu2.view(1)
        cov1 = (sigma1 ** 2).view(1, 1)
        cov2 = (sigma2 ** 2).view(1, 1)
    elif mu1.ndim == 1:
        D = mu1.shape[0]
        mu1_vec = mu1
        mu2_vec = mu2
        # Convert sigma to covariance
        if sigma1.ndim == 1 and sigma1.shape[0] == D:
            cov1 = torch.diag(sigma1 ** 2)
        elif sigma1.ndim == 2 and sigma1.shape == (D, D):
            cov1 = sigma1
        else:
            raise ValueError(
                "sigma1 must be a vector of length D or a DxD covariance matrix")
        if sigma2.ndim == 1 and sigma2.shape[0] == D:
            cov2 = torch.diag(sigma2 ** 2)
        elif sigma2.ndim == 2 and sigma2.shape == (D, D):
            cov2 = sigma2
        else:
            raise ValueError(
                "sigma2 must be a vector of length D or a DxD covariance matrix")
    else:
        raise ValueError("mu1 must be a scalar or 1D tensor")

    # Create multivariate distributions
    dist1 = torch.distributions.MultivariateNormal(
        mu1_vec, covariance_matrix=cov1)
    dist2 = torch.distributions.MultivariateNormal(
        mu2_vec, covariance_matrix=cov2)

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
    log_p1 = dist1.log_prob(x)
    log_p2 = dist2.log_prob(x)
    llr = log_p1 - log_p2  # [B, T]
    llr = torch.cumsum(llr, dim=1)  # iid Gaussian

    # Matrixize: [[0, llr], [-llr, 0]]
    result = torch.zeros(batch_size, duration, 2, 2,
                         dtype=llr.dtype)
    result[:, :, 0, 1] = llr  # = log(pre-change)  - log(post-change)
    result[:, :, 1, 0] = -llr  # = log(post-change) - log(pre-change)

    if flag_output_raw_data:
        # For univariate, squeeze last dim
        raw = x.squeeze(-1) if D == 1 else x
        return raw, result
    else:
        return result


def save_pth_dataset_Gauss(size, duration, cp_method, data_dir, mu1=0., mu2=0.1, sigma1=0.1, sigma2=0.1, p=0.01, omega=0.):
    """
    Save a dataset of raw sequence, LLRs, CLLRs, GSR statistics, and CUSUM statistics to a pth file.
    The dataset consists of LLRs generated from a Gaussian distribution with specified parameters.
    The changepoint is determined by the cp_method parameter.
    cp = torch.inf means no changepoint.
    The dataset is saved in a .pth file with the specified filename.
    Args:
        size (int): Number of samples in the dataset.
        duration (int): Duration of each sample.
        cp_method (str): Method for generating changepoints ('uniform' or 'geometric').
        data_dir (str): Directory to save the pth datasets.
        mu1 (float): Mean of the first Gaussian distribution.
        mu2 (float): Mean of the second Gaussian distribution.
        sigma1 (float): Standard deviation of the first Gaussian distribution.
        sigma2 (float): Standard deviation of the second Gaussian distribution.
        p (float): Probability parameter for geometric or uniform changepoint generation. Larger values lead to more changepoints in the dataset.
        omega (float): Non-negative parameter for GSR statistic calculation.
    Returns:
        None
    """
    if cp_method == "geometric":
        assert p in [0.001, 0.1, 0.25]
    elif cp_method == "uniform":
        assert p in [0.1, 0.5, 0.9]
    else:
        raise ValueError(
            f"cp_method must be 'uniform' or 'geometric'. Got {cp_method}.")
    per_dataset = 100  # Number of samples per dataset
    assert size % per_dataset == 0, "Size must be divisible by per_dataset."
    num_iterations = size // per_dataset

    # Define directory and filename
    dir_name = f"{data_dir}/Gauss_p{p}{cp_method}-dur{duration}"
    filename = f"{dir_name}/Gauss_p{p}{cp_method}-dur{duration}.pth"
    os.makedirs(dir_name, exist_ok=True)

    # Generate changepoints
    if cp_method == "uniform":
        dummy_duration = int(duration / p)
        cps = torch.randint(0, dummy_duration, size=[size])  # [size]
    elif cp_method == "geometric":
        m = torch.distributions.geometric.Geometric(torch.tensor([p] * size))
        cps = m.sample().int()  # [size]
    cps = torch.where(cps >= duration, torch.inf, cps)  # [size]

    # Generate LLRs, CLLRs, and GSR statistics
    sigma1 = torch.tensor(sigma1)
    sigma2 = torch.tensor(sigma2)
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
            # if i % 10 == 0:
            #     print(f"Generating {serial_num}/{size}...")

            if cps[serial_num] == torch.inf:
                changepoint = duration
            else:
                changepoint = int(cps[serial_num].numpy())

            raw_sequences[i], LLRs[i] = generate_gaussian_LLRs(
                batch_size=1,
                duration=duration,
                changepoint=changepoint,
                mu1=mu1,
                mu2=mu2,
                sigma1=sigma1,
                sigma2=sigma2,
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

            save_pth_dataset_Gauss(
                size=10000,
                duration=1000,
                cp_method=cp_method,
                data_dir=data_dir,
                mu1=0.,
                mu2=0.1,
                sigma1=0.1,
                sigma2=0.1,
                p=p,
                omega=0.0,
            )
