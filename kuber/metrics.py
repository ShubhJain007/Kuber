"""Canonical evaluation metrics for Kuber surrogates.

Pure NumPy, no model/IO dependencies, so they are trivially testable and reusable. They
implement exactly the per-channel error definitions that ``train_simshift.evaluate``
accumulates batch-wise during evaluation:

    nRMSE_c      = sqrt( mean_i (pred_i,c - gt_i,c)^2 )       on z-normalized targets
    relL2_c      = sqrt( sum_i (pred-gt)^2 / sum_i gt^2 )     scale-free relative L2
    RMSE_phys_c  = sqrt( mean_i ((pred-gt) * ystd_c)^2 )      de-normalized to physical units

`pred` and `gt` are the z-normalized targets used in training; multiplying the error by the
per-channel training std (`ystd`) recovers physical units (K, m/s, Pa).
"""
from __future__ import annotations

import numpy as np


def _flat(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, a.shape[-1])


def per_channel_nrmse(pred, gt):
    """Per-channel normalized RMSE on z-normalized targets. pred/gt: [..., C] -> [C]."""
    e = _flat(pred) - _flat(gt)
    return np.sqrt((e ** 2).mean(axis=0))


def rel_l2(pred, gt, eps=1e-12):
    """Per-channel relative L2 error: ||pred-gt|| / ||gt||. pred/gt: [..., C] -> [C]."""
    e = _flat(pred) - _flat(gt)
    g = _flat(gt)
    return np.sqrt((e ** 2).sum(axis=0) / ((g ** 2).sum(axis=0) + eps))


def rmse_physical(pred, gt, ystd):
    """Per-channel RMSE in physical units (error de-normalized by ystd). pred/gt: [...,C] -> [C]."""
    e = (_flat(pred) - _flat(gt)) * np.asarray(ystd, dtype=np.float64)[None, :]
    return np.sqrt((e ** 2).mean(axis=0))


def mean_nrmse(pred, gt):
    """Scalar mean nRMSE across channels — the headline number reported in the paper."""
    return float(per_channel_nrmse(pred, gt).mean())
