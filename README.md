<h1>🎉 KM-ARL and KM-ADD</h1>

# 1. 🎯 TL;DR

## 1.1. What Is It?

The KM-ARL and KM-ADD are estimators for the average run length (ARL) and the average detection delay (ADD), can be used for evaluating quickest changepoint detectors, and are **robust to irregular sequence lengths, which we often encounter in real-world datasets.**

## 1.2. Quick Start

See `/quick_start.ipynb`.

## 1.3. Our Paper

This is the official repository of our work published at **ICML2026**: **"[Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis](https://openreview.net/forum?id=LhGxRnGmGJ)" by Taiki Miyagawa & Akinori F. Ebihara**.

# 2. 🔖 Table of Contents

- [1. 🎯 TL;DR](#1--tldr)
  - [1.1. What Is It?](#11-what-is-it)
  - [1.2. Quick Start](#12-quick-start)
  - [1.3. Our Paper](#13-our-paper)
- [2. 🔖 Table of Contents](#2--table-of-contents)
- [3. ❓ What are the KM-ARL and KM-ADD?](#3--what-are-the-km-arl-and-km-add)
  - [3.1. Background](#31-background)
  - [3.2. Applications](#32-applications)
  - [3.3. Limitations](#33-limitations)
- [4. ✅ Quick Start](#4--quick-start)
- [5. 📝 Requirements](#5--requirements)
- [6. 📁 Directories and Files](#6--directories-and-files)
- [7. 📚 How to Cite This Work](#7--how-to-cite-this-work)

# 3. ❓ What are the KM-ARL and KM-ADD?

## 3.1. Background

The KM-ARL and KM-ADD are estimators for the average run length (ARL) and the average detection delay (ADD), can be used for evaluating quickest changepoint detectors.
They are inspired by the Kaplan-Meier estimator, a non-parametric estimator of the survival function (the probability that a subject (e.g., a person, machine, or system) will survive beyond a specific time.).
The KM-ARL and KM-ADD utilize all sequences with or without a changepoint, taking into account the probability that a detection occurs after the end of a sequence, and thus being robust against irregular sequence length.
Please check out our paper for more details.

## 3.2. Applications

The changepoint detection algorithms to be evaluated are arbitrary and may include online changepoint detection, offline changepoint detection, change detection in time-to-event data(survival analysis), temporal anomaly detection, network monitoring, biomedical signal processing, etc.

## 3.3. Limitations

- We do not support **multiple changepoint detection** (there are more than one changepoints in a sequence) or **changepoint isolation** (there are more than one type of changepoints and classification is required).
- All sequences are supposed to contain either (1) exclusively pre-change frames or (2) pre-change frames followed by post-change frames. Please adjust your data accordingly before feeding it into our algorithm.
- The KM-ARL and KM-ADD, as well as all other metrics, do not perform reliably when either the number of with-change or without-change sequences is very small. In such cases, consider using sampling methods such as bootstrapping.

# 4. ✅ Quick Start

Please have a look at `/quick_start.ipynb`, which gives an off-the-shelf, ready-to-use example of our estimators. Refer to this notebook for notes of practical relevance.

***The minimum requirement is [lifelines](https://pypi.org/project/lifelines/) 0.30.0 and [scikit-learn](https://pypi.org/project/scikit-learn/) 1.6.1, which require Python >=3.9*** (scikit-learn >= 1.7.0 requires Python >=3.10).
lifelines' dependencies include numpy, scipy, pandas, matplotlib, autograd, autograd-gamma, formulaic, etc. (see `/reqs` in the [repository of lifelines](https://github.com/CamDavidsonPilon/lifelines)), while scikit-learn additionally requires joblib, threadpoolctl, etc.
They will be installed automatically via tools such as pip.

# 5. 📝 Requirements

We have confirmed that our code runs under the following enviroment. GPUs are not required nor supported.

- Python 3.11.9
- Numpy 1.26.4
- lifelines 0.30.0
- scikit-learn 1.6.1
- changepoint-online 1.2.1
- ruptures 1.1.9
- ocpdet 0.0.6
  - The code snippets used for our paper are already included in the directory `/ocpdet`. Manual installation is unnecessary and would otherwise require TensorFlow.
- PyTorch 2.6.0
  - This is used only for saving and loading the Gaussian and Poisson `.pth` datasets and calculating statistics such as the generalized Shiryaev-Roberts and CUSUM. Otherwise, this is not strictly required at all.

# 6. 📁 Directories and Files

- `/dataset`: Dataset files (Gaussian process, Poisson process, and preprocessed WISDM Actitracker datasets) will be saved here after running `/save_dataset_*.py` or `/WISDMactitracker.ipynb`. Currently, the preprocessed WISDM "labeled" dataset is included due to storage limitations. Downloading [the original WISDM Actitracker dataset](https://www.cis.fordham.edu/wisdm/dataset.php) is required to reproduce our paper's results.
- `/ocpdet`: Snippets taken from `ocpdet` are contained, and you do not have to install `ocpdet` manually. Note that if you try to fully install the original `ocpdet` via `pip`, TensorFlow will be required, although it is not used in our code.
- `/results`: Result files will be saved here after running `/calc_*.py`. Currently, the results for the preprocessed WISDM "labeled" dataset is included due to storage limitations.
- `/calc_{gt/es}{ARL/ADD}_{Gauss/Posson/WISDM/Gauss_cpmodels/WISDM_cpmodels}.py`: The evaluation codes for our results given in the paper. They are given for reproducibility. `gt` and `es` stand for ground-truth and estimation. `_cpmodels` means the statistics (charts) used in the code is not derived from the ground-truth distribution. The statistics are defined by each detection model.
- `/LICENSE`: License file.
- `/metrics.py`: Utility codes including the functions that compute the LB-ARL and LB-ADD.
- `/quick_start.ipynb`: The quick start notebook with minimum requirements.
- `/quick_start_toolkit.py`: The functions for `/quick_start.ipynb`, including the functions that compute the KM-ARL and the KM-ADD. The functions therein are identical to those in `metrics.py` and `statistic_tools.py`.
- `/README.md`: Readme file.
- `/save_dataset_{Gauss/Poisson}.py`: Create and save the Gaussian and Poisson datasets to `/dataset` directory.
- `/statistic_tools.py`: Utility functions including the functions that compute the KM-ARL and KM-ADD.
- `/WISDMactitracker.ipynb`: All preprocesses used for the WISDM Actitracker dataset in our paper are described. We generated `/dataset/WISDM_labeled_common.pkl` and `/dataset/WISDM_labeled_common.pkl` datasets using this file.

# 7. 📚 How to Cite This Work

```
@inproceedings{miyagawa2026accurate,
title={Accurate Evaluation of Quickest Changepoint Detectors via Non-parametric Survival Analysis},
author={Taiki Miyagawa and Akinori F. Ebihara},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=LhGxRnGmGJ}
}
```
