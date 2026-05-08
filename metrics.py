# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def calc_naive_ARL(preds_all: np.ndarray, cps_all: np.ndarray, flag_less_biased: bool = False, flag_verbose: bool = True):
    """ Calculate naive ARL from predictions and changepoints.
    Args:
        preds_all (np.ndarray): Predictions of shape [num_samples].
        cps_all (np.ndarray): Changepoints of shape [num_samples].
    Returns:
        naive_ARL (float): Naive ARL value. Can be np.nan if no false alarms are found or no cp=inf sequences are present (the condition depends on flag_less_biased).
        sterr (float): Standard error of the mean. Can be np.nan if no false alarms are found or no cp=inf sequences are present  (the condition depends on flag_less_biased).
        effective_num_samples (int): Number of samples used to calculate ARL.
    """
    preds_all = np.array(preds_all)
    cps_all = np.array(cps_all)
    assert np.prod(cps_all >= 0), "cps_all must be non-negative."
    assert preds_all.shape == cps_all.shape, "preds_all and cps_all must have the same shape."
    assert preds_all.ndim == 1, "preds_all must be 1D."
    assert cps_all.ndim == 1, "cps_all must be 1D."

    num_samples = preds_all.shape[0]
    num_overrun = np.sum(
        preds_all == -1)

    # Use all false alermed sequences to compute ARL.
    # Pro: More samples to compute ARL.
    # Con: Causes a negative bias in the ARL estimate.
    if not flag_less_biased:
        flags_alarmed = preds_all >= 0  # [num_samples].
        # No -1 in preds_alarmed
        preds_alarmed = preds_all[flags_alarmed]  # [num_alarmed]
        cps_alarmed = cps_all[flags_alarmed]  # [num_alarmed].

        flags_false_alarm = preds_alarmed < cps_alarmed  # [num_alarmed]
        preds_false_alarm = preds_alarmed[
            flags_false_alarm]  # [num_false_alarm]
        num_false_alarm = preds_false_alarm.shape[0]

        if len(preds_false_alarm) == 0:
            # No false alarms, so we have no choice but defining ARL is infinite...
            print("WARNING: No false alarms found. ARL is set to np.nan.")
            naive_ARL = np.nan
            sterr = np.nan
        else:
            naive_ARL = preds_false_alarm.mean()  # base 0
            # standard error of the mean
            sterr = np.std(preds_false_alarm) / np.sqrt(num_false_alarm)
            assert not np.isnan(
                naive_ARL), "Something went wrong with the ARL calculation."
            assert not np.isnan(
                sterr), "Something went wrong with the standard error calculation."

        effective_num_samples = num_samples - \
            num_overrun  # num of samples used to calc ARL

    # Use only cp = inf sequences to compute ARL.
    # Pro: Smaller bias in the ARL estimate.
    # Con: Less samples to compute ARL.
    else:
        flags_cpinf = cps_all == np.inf
        if len(flags_cpinf) == 0:
            return np.nan, np.nan, 0
        preds_cpinf = preds_all[flags_cpinf]
        preds_cpinf = preds_cpinf[preds_cpinf >= 0]  # [effective_num_samples]
        if len(preds_cpinf) == 0:
            return np.nan, np.nan, 0.
        naive_ARL = preds_cpinf.mean()
        sterr = np.std(preds_cpinf) / np.sqrt(len(preds_cpinf))
        effective_num_samples = len(preds_cpinf)

    # Verbose
    if flag_verbose:
        print(f"#samples:                   {num_samples}")
        print(f"#overruns:                  {num_overrun}")
        print(f"#samples for computing ARL: {effective_num_samples}")
        print(f"ARL±SE:                     {naive_ARL} ± {sterr}")

    return naive_ARL, sterr, effective_num_samples


def calc_naive_ADD(preds_all: np.ndarray, cps_all: np.ndarray, flag_verbose: bool = True):
    preds_all = np.array(preds_all)
    cps_all = np.array(cps_all)
    assert np.prod(cps_all >= 0), "cps_all must be non-negative."
    assert preds_all.shape == cps_all.shape, "preds_all and cps_all must have the same shape."
    assert preds_all.ndim == 1, "preds_all must be 1D."
    assert cps_all.ndim == 1, "cps_all must be 1D."
    assert np.any(cps_all >= 0)

    num_samples = preds_all.shape[0]
    num_no_alarm = np.sum(
        preds_all == -1)

    flags_alarmed = preds_all >= 0  # [num_samples].
    # There is no -1 in preds_alarmed.
    preds_alarmed = preds_all[flags_alarmed]  # [num_alarmed]
    cps_alarmed = cps_all[flags_alarmed]  # [num_alarmed].

    flags_delayed_alarm = preds_alarmed > cps_alarmed  # [num_alarmed]
    preds_delayed_alarm = preds_alarmed[
        flags_delayed_alarm]  # [num_delayed_alarm]
    cps_delayed_alarm = cps_alarmed[flags_delayed_alarm]  # [num_delayed_alarm]
    delay = preds_delayed_alarm - cps_delayed_alarm  # [num_delayed_alarm]
    num_delayed_alarm = preds_delayed_alarm.shape[0]

    if len(preds_delayed_alarm) == 0:
        # No delayed alarms (all alarms are false alarms)
        print("WARNING: No delayed alarms found. ADD is set to np.nan.")
        naive_ADD = np.nan
        sterr = np.nan
    else:
        naive_ADD = delay.mean()  # base 0
        sterr = np.std(delay) / np.sqrt(num_delayed_alarm)
        assert not np.isnan(
            naive_ADD), "Something went wrong with the ADD calculation."
        assert not np.isnan(
            sterr), "Something went wrong with the standard error calculation."

    effective_num_samples = delay.shape[0]  # num of samples used to calc ADD

    # Verbose
    if flag_verbose:
        print(f"#samples:                   {num_samples}")
        print(f"#no alarm samples:          {num_no_alarm}")
        print(f"#samples for computing ADD: {effective_num_samples}")
        print(f"ADD±SE:                     {naive_ADD} ± {sterr}")

    return naive_ADD, sterr, effective_num_samples


def calc_combined_mean_and_sem(means, sems, ns):
    """
    Calculate the combined mean and SEM from multiple groups.

    Parameters:
    means (array-like): Array of means for each group.
    sems (array-like): Array of SEMs for each group.
    ns (array-like): Array of sample sizes for each group.

    Returns:
    tuple: Combined mean and SEM.
    """
    # Step 1: Overall mean (weighted average)
    mu_total = np.sum(ns * means) / np.sum(ns)

    # Step 2: Restore SD for each group
    sds = sems * np.sqrt(ns)

    # Step 3: Pooled variance (within + between variance)
    within_var = np.sum((ns - 1) * sds**2)
    between_var = np.sum(ns * (means - mu_total)**2)
    pooled_var = (within_var + between_var) / (np.sum(ns) - 1)

    # Step 4: Overall standard error
    sem_total = np.sqrt(pooled_var) / np.sqrt(np.sum(ns))

    return mu_total, sem_total
