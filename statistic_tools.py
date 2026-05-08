# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional, Tuple, Union

import numpy as np
import torch
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.utils import resample
from tqdm import tqdm


def calc_logGSRstatistic_v2(CLLRs, omega=0., flag_log=True):
    """
    Calculate GSR statistic at a 'single' timestep = duration (not all timesteps) from conditional LLRs
    using a numerically stable log-sum-exp approach.
    Args:
        CLLRs (torch.Tensor): Conditional LLR sequence of shape (batch_size, duration, num_classes, num_classes).
        omega (float): Non-negative parameter.
        flag_log (bool): If True, apply log to GSR statistic.
    Returns:
        torch.Tensor: GSR statistic of shape (batch_size, num_classes, num_classes).

    What is omega?:
        Omega is a non-negative parameter called the head-start.
        - Bayesian: A prior belief that the change has already occurred before monitoring begins.
            If the change-point prior is geometric with parameter p, then omega=p/(1-p).
        - Evidence accumulator: Extra “evidence” in the statistic at n=0.
            A larger omega means the chart begins closer to the alarm threshold,
            so the first few observations can trigger an alarm more quickly (“fast initial response”).
        - Minimax design tool: A tuning knob that lets you shape the entire run-length distribution.
            Choosing omega appropriately (Pollak's rule) makes the worst-case conditional expected delay essentially equal to
            the stationary-average delay, achieving asymptotic minimax optimality.

    """
    if not flag_log:
        print("WARNING: GSR statistic is not in log-space, which will cause numerical instability.")

    duration = CLLRs.shape[1]
    # Expand and permute as before
    CLLRs = CLLRs[:, None, :, :, :]  # [B, 1, T, K, K]
    CLLRs = CLLRs.expand(-1, duration, -1, -1, -1)  # [B, T, T, K, K]
    CLLRs = CLLRs.permute(0, 3, 4, 1, 2)  # [B, K, K, T, T]
    CLLRs = torch.triu(CLLRs)  # [B, K, K, T, T]
    CLLRs = torch.sum(CLLRs, dim=4)  # [B, K, K, T]
    CLLRs = torch.flip(CLLRs, dims=[3])  # [B, K, K, T]

    # Compute stable cumulative sums in log-space
    logS = torch.empty_like(CLLRs)  # [B, K, K, T] (placeholder)
    logS[..., 0] = CLLRs[..., 0]
    for t in range(1, duration):
        logS[..., t] = torch.logaddexp(logS[..., t - 1], CLLRs[..., t])

    if flag_log:
        # log( sum_{τ=0..t} exp(LLR(τ)) + omega * exp(LLR(t)) )
        GSRstat = logS + torch.log1p(omega * torch.exp(CLLRs - logS))
    else:
        # sum_{τ=0..t} exp(LLR(τ)) + omega * exp(LLR(t))
        GSRstat = torch.exp(logS) + omega * torch.exp(CLLRs)

    GSRstat = GSRstat[..., -1]  # [B, K, K]

    return GSRstat


def calc_all_logGSRstatistics_v2(CLLRs: Union[torch.Tensor, np.ndarray], omega=0., pick_index=[1, 0]) -> Union[torch.Tensor, np.ndarray]:
    """ Calculate GSR statistic at all timesteps from conditional LLRs.
        This function is very time-consuming (duration^2) and not recommended.
        R_n = (1 + R_{n-1}) CLR_n (n >= 0),
        where R_-1 := omega, and CLR is the conditional likelihood ratio (CLLRs = log (CLRs)).
    Args:
        CLLRs (torch.Tensor): LLR sequence of shape (batch_size, duration, num_classes, num_classes) or (batch_size, duration).
        omega (float): Non-negative parameter.
        pick_index (array like): Index to pick the statistic from the CLLRs matrix. Not used when len(CLLRs.shape) == 2.
    Returns:
        torch.Tensor: GSR statistic of shape (batch_size, duration).
    """
    assert omega >= 0, "omega must be non-negative."
    assert len(pick_index) == 2, "pick_index must be of length 2."
    if len(CLLRs.shape) == 4:
        CLLRs = CLLRs[:, :, pick_index[0], pick_index[1]]  # [B, T]
    elif len(CLLRs.shape) == 2:
        pass
    else:
        raise ValueError("CLLRs must be of shape [B, T, K, K] or [B, T].")

    duration = CLLRs.shape[1]

    if isinstance(CLLRs, torch.Tensor):
        if not isinstance(omega, torch.Tensor):
            omega = torch.tensor(omega, dtype=CLLRs.dtype, device=CLLRs.device)

        logGSR_all = torch.empty_like(CLLRs)  # [B, T] (placeholder)

        it_logGSR = CLLRs[:, 0] + torch.log1p(omega)
        logGSR_all[:, 0] = it_logGSR

        for it_t in range(1, duration):
            it_CLLRs = CLLRs[:, it_t]  # [B,]
            it_logGSR = it_CLLRs + \
                torch.logaddexp(torch.zeros_like(it_logGSR), it_logGSR)  # [B,]
            logGSR_all[:, it_t] = it_logGSR

    elif isinstance(CLLRs, np.ndarray):
        if not isinstance(omega, np.ndarray):
            omega = np.array(omega, dtype=CLLRs.dtype)

        logGSR_all = np.empty_like(CLLRs)  # [B, T] (placeholder)

        it_logGSR = CLLRs[:, 0] + np.log1p(omega)
        logGSR_all[:, 0] = it_logGSR

        for it_t in range(1, duration):
            it_CLLRs = CLLRs[:, it_t]
            it_logGSR = it_CLLRs + np.logaddexp(
                np.zeros_like(it_logGSR), it_logGSR)
            logGSR_all[:, it_t] = it_logGSR
    else:
        raise ValueError(
            "CLLRs must be either a torch.Tensor or a numpy.ndarray.")

    return logGSR_all  # [B, T]


def calc_all_CUSUMstatistics(CLLRs: Union[torch.Tensor, np.ndarray], pick_index=[1, 0]) -> Union[torch.Tensor, np.ndarray]:
    """ Calculate general non-i.i.d. CUSUM statistic at all timesteps from conditional LLRs.
    Args:
        CLLRs (Union[torch.Tensor, np.ndarray]): LLR sequence of shape (batch_size, duration, num_classes, num_classes)
            or (batch_size, duration).
        pick_index (array like): Index to pick the statistic from the CLLRs matrix. Not used when len(CLLRs.shape) == 2.
    Returns:
        Union[torch.Tensor, np.ndarray]: CUSUM statistic of shape (batch_size, duration) or (batch_size, duration, num_classes, num_classes).
    """
    assert len(pick_index) == 2, "pick_index must be of length 2."

    if len(CLLRs.shape) == 4:
        CLLRs = CLLRs[:, :, pick_index[0], pick_index[1]]  # [B, T]
    elif len(CLLRs.shape) == 2:
        pass
    else:
        raise ValueError(
            f"CLLRs must be of shape [B, T, K, K] or [B, T]. Got shape {CLLRs.shape}.")

    duration = CLLRs.shape[1]

    if isinstance(CLLRs, torch.Tensor):
        cusum_all = torch.empty_like(CLLRs)  # [B, T] (placeholder)
        cusum_all[:, 0] = CLLRs[:, 0]

        for it_t in range(1, duration):
            cusum_all[:, it_t] = torch.maximum(
                torch.zeros_like(cusum_all[:, it_t-1]),
                cusum_all[:, it_t-1] + CLLRs[:, it_t]
            )

    elif isinstance(CLLRs, np.ndarray):
        cusum_all = np.empty_like(CLLRs)  # [B, T] (placeholder)
        cusum_all[:, 0] = CLLRs[:, 0]

        for it_t in range(1, duration):
            cusum_all[:, it_t] = np.maximum(
                np.zeros_like(cusum_all[:, it_t-1]),
                cusum_all[:, it_t-1] + CLLRs[:, it_t]
            )

    else:
        raise ValueError(
            "CLLRs must be either a torch.Tensor or a numpy.ndarray.")

    return cusum_all  # [B, T]


def logits_to_LLRs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to log-likelihood ratio matrices.
    Args:
        logits (torch.Tensor): Logits of shape (B, T, num_classes) or (B, num_classes).
    Returns:
        torch.Tensor: LLRs of shape (B, T, K, K) or (B, K, K).
    """
    # Calculate LLRs
    # [B, T, K, K] or [B, K, K]
    llrs = logits.unsqueeze(-1) - logits.unsqueeze(-2)

    return llrs


def LLRs_to_posterior(LLRs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert log-likelihood ratios (LLRs) to posterior probabilities.
    Flat prior is assumed. posterior = p(y=1|x) = p(x|y=1)/(p(x|y=0)+p(x|y=1)) = 1/(1+exp(-LLR)).
    Args:
        LLRs (Union[torch.Tensor, np.ndarray]): LLRs of any shape.
    Returns:
        Union[torch.Tensor, np.ndarray]: Posterior probabilities of the same shape as LLRs.
    """
    if isinstance(LLRs, np.ndarray):
        return 1 / (1 + np.exp(-LLRs))
    elif isinstance(LLRs, torch.Tensor):
        return torch.sigmoid(LLRs)


def LLRs_to_CLLRs(LLRs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert non-conditional LLRs to conditional LLRs.
    Args:
        LLRs (Union[torch.Tensor, np.ndarray]): LLRs of shape (batch_size, duration, num_classes, num_classes) or (batch_size, duration).
    Returns:
        Union[torch.Tensor, np.ndarray]: CLLRs of shape (batch_size, duration, num_classes, num_classes) or (batch_size, duration).
    """
    if isinstance(LLRs, np.ndarray):
        CLLRs = np.empty_like(LLRs)  # [B, T, K, K] or [B, T]
    elif isinstance(LLRs, torch.Tensor):
        CLLRs = torch.empty_like(LLRs)  # [B, T, K, K] or [B, T]

    if len(LLRs.shape) == 4:
        CLLRs[:, 0, :, :] = LLRs[:, 0, :, :]
        CLLRs[:, 1:, :, :] = LLRs[:, 1:, :, :] - LLRs[:, :-1, :, :]
    elif len(LLRs.shape) == 2:
        CLLRs[:, 0] = LLRs[:, 0]
        CLLRs[:, 1:] = LLRs[:, 1:] - LLRs[:, :-1]

    return CLLRs  # [B, T, ...]


def calc_LLRs_from_logits(logits: torch.Tensor):
    """
    Calculate the log-likelihood ratios (LLRs) from the logits and true labels.
    Args:
        logits: Tensor of shape (batch_size, num_classes) containing the logits from the model.
    Return:
        llrs: Tensor of shape (batch_size,) containing the flat-prior LLRs log p(x|y=1)/p(x|y=0) = log p(y=1|x)/p(y=0/x), where 1 is post-change and 0 is pre-change.
    """
    # Calculate LLRs
    llrs = logits[:, 1] - logits[:, 0]

    return llrs  # [B,]


def first_flip(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray], eps: float = 0.0) -> Union[torch.Tensor, np.ndarray]:
    """
    Find the first position where a >= b + eps occurs for each sequence in batch.

    Args:
        a, b : shape = [B, T] - Either torch.Tensor or numpy.ndarray with identical shapes
        eps  : Numerical error tolerance. The comparison is performed as (a - b) >= eps.

    Returns:
        first_pos : shape = [B] - The first index where a >= b + eps occurs.
        If no such position exists, return -1 (b > a in the sequence) or 0 (a > b in the sequence).
    """
    if a.shape != b.shape or len(a.shape) != 2:
        raise ValueError(
            "a, b must have identical shape [batch_size, duration].")
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # PyTorch implementation
        # 1. Obtain the comparison result as a boolean (True ⇔ a >= b + eps)
        gt = (a - b) >= eps          # shape [B, T]

        # 2. Detect where the comparison result changes between consecutive time steps
        flips = gt[:, 1:] ^ gt[:, :-1]     # XOR → shape [B, T-1]

        # 3. Extract the first position where True occurs for each batch
        has_flip = flips.any(dim=1)  # shape [B]  True/False
        preflip = gt[:, 0]  # shape [B]  True/False

        # argmax returns 0 even if there is no True in the row, so handle it as float
        # +1 to map back to the original time step (the gap came from XOR operation above)
        first_pos = torch.argmax(flips.float(), dim=1) + 1

        # 4. Replace batches with no flips with -1. Replace the preflip with 0.
        first_pos = torch.where(has_flip, first_pos,
                                torch.full_like(first_pos, -1))  # shape [B]
        first_pos = torch.where(preflip, torch.full_like(
            first_pos, 0), first_pos)  # shape [B]

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        # NumPy implementation
        # 1. Obtain the comparison result as a boolean (True ⇔ a >= b + eps)
        gt = (a - b) >= eps          # shape [B, T]

        # 2. Detect where the comparison result changes between consecutive time steps
        # XOR → shape [B, T-1]
        flips = np.logical_xor(gt[:, 1:], gt[:, :-1])

        # 3. Extract the first position where True occurs for each batch
        has_flip = np.any(flips, axis=1)  # shape [B]  True/False
        preflip = gt[:, 0]  # shape [B]  True/False

        # Add a small value to ensure argmax finds the first True
        # (otherwise it returns 0 for all-False rows)
        mask = flips.astype(np.float32)
        # Set the first position to a small value (less than 1) to avoid incorrect argmax for all-False rows
        mask[~has_flip] = -1e-10
        # +1 to map back to the original time step
        first_pos = np.argmax(mask, axis=1) + 1

        # 4. Replace batches with no flips with -1. Replace the preflip with 0.
        first_pos = np.where(has_flip, first_pos, -1)  # shape [B]
        first_pos = np.where(preflip, 0, first_pos)  # shape [B]

    else:
        raise ValueError(
            "Inputs a and b must be either both torch.Tensor or both numpy.ndarray.")

    return first_pos


def prediction_func(statistics: np.ndarray, thresh: float, batch_size: int, verbose: bool = True) -> np.ndarray:
    num_samples, duration = statistics.shape
    assert num_samples % batch_size == 0, f"Batch size must be a divisor of the number of samples. Got {num_samples} samples and batch size {batch_size}."
    num_iterations = num_samples // batch_size
    thresh_array = np.full((batch_size, duration), thresh)  # [B, T]

    ls_preds = []
    for i in tqdm(range(num_iterations), disable=not verbose):
        start = i * batch_size
        end = (i + 1) * batch_size
        it_stat = statistics[start:end]  # [B, T]

        predictions = first_flip(
            it_stat,
            thresh_array,
            eps=0.0)  # [B]

        ls_preds.append(predictions)

    preds_all = np.concatenate(ls_preds)  # [num_samples]
    return preds_all


def revealing(data: torch.Tensor, cps: torch.Tensor):
    """ Gradually revealing sequences.
    Args:
        data: shape (batch_size, duration, num_input_channels). No maskinig.
        cps: shape (batch_size, )
    Returns:
        data: shape (batch_size * duration, duration, num_input_channels). Gradulally revealing sequences.
        cps: shape (batch_size * duration, )
    """
    batch_size, duration, num_input_channels = data.shape

    # Shape (batch_size * duration, duration, num_input_channels)
    data = data.repeat_interleave(duration, dim=0)

    # Shape (batch_size * duration, )
    cps = cps.repeat_interleave(duration)

    # Reveal the sequences gradually, simulating the process of observing the sequences over time.
    for i in range(0, duration-1):
        # mask out the first i values of each sequence
        data[i::duration, i+1:] = 0.0

    return data, cps


# =========================================================
# Calc KME
# =========================================================


def bootstrap_rmst_sklearn(durations, events, tau, B=2000):
    """
    Uses resample in scikit-learn 
    """
    durations = np.asarray(durations)
    events = np.asarray(events)
    n = len(durations)

    def _compute_rmst(durations, events, tau, return_variance=False):
        """
        Returns
        - rmst: Restricted Mean Survival Time (RMST)
        - var_rmst: Variance of RMST (if return_variance is True)
        """
        kmf = KaplanMeierFitter().fit(durations, events)
        if return_variance:
            rmst, var_rmst = restricted_mean_survival_time(
                kmf, t=tau, return_variance=True)
            return rmst, var_rmst
        else:
            rmst = restricted_mean_survival_time(kmf, t=tau)
            return rmst

    rmst_orig, v = _compute_rmst(durations, events, tau, return_variance=True)
    se_rmst_orig = np.sqrt(v / n)

    samples = np.empty(B)
    for b in range(B):
        d_s, e_s = resample(
            durations, events,
            replace=True,
            n_samples=n,
        )
        samples[b] = _compute_rmst(d_s, e_s, tau)

    bs_rmst = samples.mean()  # bootstrap mean
    bias = bs_rmst - rmst_orig
    bc_rmst = rmst_orig - bias
    # unbiased standard error of bs_rmst: ddof=1
    se_bs = np.std(samples, ddof=1)
    # percentile confidence interval
    ci_lower, ci_upper = np.percentile(samples, [2.5, 97.5])

    return {
        'rmst_orig': rmst_orig,
        'bootstrap_rmst': bs_rmst,
        'bias_corrected_rmst': bc_rmst,
        'bias': bias,
        'se_rmst_orig': se_rmst_orig,
        'se_bootstrap_rmst': se_bs,
        'ci_95_percentile': (ci_lower, ci_upper)
    }


def calc_tildeTi_and_event_indicator_ARL(preds: np.ndarray, cps: np.ndarray, duration: int) -> Tuple[np.ndarray, np.ndarray]:
    """ duration is fixed.
    Calculate tildeT_i and event_indicator from preds and cps.
    Args:
        preds (np.ndarray): Predicted event indices, shape = [num_samples].
        cps (np.ndarray): Censoring indices, shape = [num_samples].
        duration (int): Duration of the monitoring period.
    Returns:
        tildeT_i (np.ndarray): Finite durations, shape = [num_samples].
        event_indicator (np.ndarray): 0 = censored, 1 = event, shape = [num_samples].
    """
    # T_i: [num_samples] possibly infinite event index
    # C_i: [num_samples] finite censoring index
    # tildeT_i: [num_samples] finite
    # event_indicator: [num_samples] 0 = censored, 1 = event

    T_i = np.where(preds == -1, np.inf, preds)
    C_i = np.where(cps == np.inf, duration-1, cps)
    tildeT_i = np.where(T_i > C_i, C_i, T_i)  # finite
    event_indicator = (T_i <= C_i).astype(int)

    return tildeT_i, event_indicator


def calc_tildeTi_and_event_indicator_from_durations_ARL(preds: np.ndarray, cps: np.ndarray, durations: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """ duration is fixed.
    Calculate tildeT_i and event_indicator from preds and cps.
    Args:
        preds (np.ndarray): Predicted event times, shape = [num_samples].
        cps (np.ndarray): Censoring times, shape = [num_samples].
        duration (int): Duration of the monitoring period.
    Returns:
        tildeT_i (np.ndarray): Finite durations, shape = [num_samples].
        event_indicator (np.ndarray): 0 = censored, 1 = event, shape = [num_samples].
    """
    # T_i: [num_samples] possibly infinite event index
    # C_i: [num_samples] finite censoring index
    # tildeT_i: [num_samples] finite
    # event_indicator: [num_samples] 0 = censored, 1 = event
    durations = np.asarray(durations)

    T_i = np.where(preds == -1, np.inf, preds)
    C_i = np.where(cps == np.inf, durations - 1, cps)
    tildeT_i = np.where(T_i > C_i, C_i, T_i)  # finite
    event_indicator = (T_i <= C_i).astype(int)

    return tildeT_i, event_indicator


def calc_KME(
        preds: np.ndarray, cps: np.ndarray, duration: Union[int, None], duration_array: Union[list, np.ndarray] = None,
        num_bs_samples: Optional[int] = None, flag_verbose: bool = True) -> Tuple[float, float, int]:
    """
    Calculate KME-based ARL (Average Run Length) using the Kaplan-Meier estimator.
    Args:
        preds (np.ndarray): Predicted event times, shape = [num_samples].
        cps (np.ndarray): Censoring times, shape = [num_samples].
        duration (int or None): Duration of the monitoring period. Ignored and reset to None if duration_array is provided.
        duration_array (Optional[Union[list, np.ndarray]]): Array of finite durations for each sample. If provided, it overrides the duration argument.
        num_bs_samples (Optional[int]): Number of bootstrap samples for correcting the finite-sample bias. If None, no bootstrap is performed. Note that the truncation bias is much larger than the finite-sample bias, so the bootstrap is not necessary in many cases when duration << true ARL.
    Returns:
        rmst (float): KME-ARL.
        se_rmst (float): Standard error of the RMST (restricted mean survival time).
        num_samples (int): Number of samples.
    """
    if duration_array is not None:
        assert len(duration_array) == len(
            preds), f"duration_array must have the same length as preds. Got {len(duration_array)} and {len(preds)}."
        duration = None
        duration_array = np.asarray(duration_array)
    preds = np.asarray(preds)
    cps = np.asarray(cps)

    if num_bs_samples is not None:
        assert num_bs_samples > 0, f"num_bs_samples must be a positive integer. Got {num_bs_samples}."

    num_samples = len(preds)

    # T_i: [num_samples] possibly infinite event time
    # C_i: [num_samples] finite censoring time
    # tildeT_i: [num_samples] finite durations
    # event_indicator: [num_samples] 0 = censored, 1 = event
    if duration_array is None:
        tildeT_i, event_indicator = calc_tildeTi_and_event_indicator_ARL(
            preds, cps, duration)
    else:
        tildeT_i, event_indicator = calc_tildeTi_and_event_indicator_from_durations_ARL(
            preds, cps, duration_array)
    max_index = int(np.max(tildeT_i))  # Cares of both tildeT_i = T_i and C_i

    if num_bs_samples is None:
        # Fit KME survival function
        kmf = KaplanMeierFitter()
        kmf.fit(durations=tildeT_i, event_observed=event_indicator,
                label="Survival Curve")

        # Calc the area under the survival curve (KME-ARL) for assertion
        timeline = np.arange(0, max_index + 1)
        surv_prob = kmf.predict(timeline)
        kme_arl = np.sum(surv_prob)

        # Calc RMST (= KME-ARL)
        restriction = max_index + 1
        rmst, var_rmst = restricted_mean_survival_time(
            kmf, t=restriction, return_variance=True)  # E[tau] and Var[tau] under restriction
        assert np.abs(
            rmst - kme_arl) < 1e-6, f"rmst and kme_arl should be the same. Got rmst={rmst}, kme_arl={kme_arl}."
        se_rmst = np.sqrt(var_rmst / num_samples)  # Standard error of RMST

    else:
        # Report the standard error of the original RMST to avoid double bootstrapping and reduce computation time. Note that the SE of the bias corrected RMST would be larger than the original RMST's.
        dc = bootstrap_rmst_sklearn(
            tildeT_i, event_indicator, max_index + 1, B=num_bs_samples)
        rmst = dc['bias_corrected_rmst']
        se_rmst = dc['se_rmst_orig']

        # dc keys and values:
        # 'rmst_orig': rmst_orig,
        # 'bootstrap_rmst': bs_rmst,
        # 'bias_corrected_rmst': bc_rmst,
        # 'bias': bias,
        # 'se_rmst_orig': se_rmst_orig,
        # 'se_bootstrap_rmst': se_bs,
        # 'ci_95_percentile': (ci_lower, ci_upper)

    # Verbose
    if flag_verbose:
        print(f"KME-ARL: {kme_arl} +/- {se_rmst}")

    return rmst, se_rmst, num_samples


def calc_KME_ADD(preds: np.ndarray, cps: np.ndarray, duration: int, duration_array: np.ndarray = None, flag_verbose: bool = True) -> Tuple[float, float, int]:
    """
    Calculate KME-based ADD (Average delay to detection) using the Kaplan-Meier estimator.
    Args:
        preds (np.ndarray): Predicted event times, shape = [num_samples].
        cps (np.ndarray): Censoring times, shape = [num_samples].
        duration (int): Duration of the monitoring period. Ignored and reset to None if duration_array is provided.
        duration_array (Optional[np.ndarray]): Array of finite durations for each sample. If provided, it overrides the duration argument.
        flag_verbose (bool): If True, print the KME-ADD and its standard error
    Returns:
        rmst (float): KME-ADD.
        se_rmst (float): Standard error of the RMST (restricted mean survival time; survival time := detection delay in calc_KME_ADD).
        effective_num_samples (int): Number of effective samples after filtering. This is the number of samples used for calculating KME.
    Note:
        cps == inf                  : not used by definition
        Under the condition cps < inf,
            - preds <= cps and preds != -1: not used by definition
            - preds >  cps and preds != -1: we need E[preds - cps] w/ the aid of censored seqs.
            - preds = -1                  : 'censored' at the end of the seq.
    """
    if duration_array is not None:
        assert len(
            duration_array) == len(preds), f"duration_array must have the same length as preds. Got {len(duration_array)} and {len(preds)}."
        duration = None
        duration_array = np.asarray(duration_array)
    preds = np.asarray(preds)
    cps = np.asarray(cps)

    if duration_array is None:
        # Remove nu = inf sequences
        flag_finite_nu = cps != np.inf  # num_finite_nu := sum(flag_finite_nu)
        preds_finite_nu = preds[flag_finite_nu]  # [num_finite_nu]
        cps_finite_nu = cps[flag_finite_nu]  # [num_finite_nu]

        # Remove 'preds <= cps and preds != -1' (tau \leq nu \cap not overrun) sequences
        flag_predsleqcps = preds_finite_nu <= cps_finite_nu  # [num_finite_nu]
        flag_underrun = preds_finite_nu != -1  # [num_finite_nu]
        flag_remove = flag_predsleqcps & flag_underrun  # [num_finite_nu]
        flag_filtered = ~flag_remove  # [num_finite_nu]
        preds_filtered = preds_finite_nu[flag_filtered]
        cps_filtered = cps_finite_nu[flag_filtered]

        duration_filtered = duration

    else:
        # Remove nu = inf sequences
        flag_finite_nu = cps != np.inf  # num_finite_nu := sum(flag_finite_nu)
        preds_finite_nu = preds[flag_finite_nu]  # [num_finite_nu]
        cps_finite_nu = cps[flag_finite_nu]  # [num_finite_nu]
        duration_finite_nu = duration_array[flag_finite_nu]  # [num_finite_nu]

        # Remove 'preds <= cps and preds != -1' (tau \leq nu \cap not overrun) sequences
        flag_predsleqcps = preds_finite_nu <= cps_finite_nu  # [num_finite_nu]
        flag_underrun = preds_finite_nu != -1  # [num_finite_nu]
        flag_remove = flag_predsleqcps & flag_underrun  # [num_finite_nu]
        flag_filtered = ~flag_remove  # [num_finite_nu]
        preds_filtered = preds_finite_nu[flag_filtered]
        cps_filtered = cps_finite_nu[flag_filtered]
        duration_filtered = duration_finite_nu[flag_filtered]

    effective_num_samples = preds_filtered.shape[0]

    # If no delayed alarms are found, return NaN
    if effective_num_samples == 0:
        print("WARNING: No delayed alarms found. ADD is set to np.nan.")
        return np.nan, np.nan, 0

    # Define T_i, C_i, tildeT_i, event_indicator
    # T_i: [effective_num_samples] possibly infinite event index
    # C_i: [effective_num_samples] finite censoring index
    # tildeT_i: [effective_num_samples] finite
    # event_indicator: [effective_num_samples] 0 = censored, 1 = event
    T_i = np.where(
        preds_filtered == -1,
        np.inf, preds_filtered - cps_filtered)  # [effective_num_samples]
    C_i = np.where(
        preds_filtered == -1,
        duration_filtered - 1 - cps_filtered, np.inf)  # [effective_num_samples]
    tildeT_i = np.where(T_i > C_i, C_i, T_i)  # finite
    event_indicator = (T_i <= C_i).astype(int)

    max_index = int(np.max(tildeT_i))  # Cares of both tildeT_i = T_i and C_i

    # Fit KME survival function
    kmf = KaplanMeierFitter()
    kmf.fit(durations=tildeT_i, event_observed=event_indicator,
            label="Survival Curve")

    # Calc the area under the survival curve (KME-ADD) for assertion
    timeline = np.arange(0, max_index + 1)
    surv_prob = kmf.predict(timeline)
    kme_add = np.sum(surv_prob)

    # Calc RMST (=KME=ADD)
    restriction = max_index + 1
    rmst, var_rmst = restricted_mean_survival_time(
        kmf, t=restriction, return_variance=True)  # E[tau] and Var[tau] under restriction

    # IntegrationWarning may be safely ignored
    assert np.abs(
        rmst - kme_add) < 1e-6, f"rmst and kme_add should be the same. Got rmst={rmst}, kme_add={kme_add}."

    # Standard error of RMST
    se_rmst = np.sqrt(var_rmst / effective_num_samples)

    # Verbose
    if flag_verbose:
        print(f"KME-ADD: {kme_add} +/- {se_rmst}")

    return rmst, se_rmst, effective_num_samples
