# Copyright 2026 Taiki Miyagawa

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Optional, Tuple, Union

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.utils import resample


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
