"""PDE-Refiner (Lippe, Perdikaris, Brandstetter, Cranmer et al., NeurIPS 2023),
faithfully adapted to our steady field-prediction setting.

Instead of one-shot MSE (which is spectrally biased — it drops low-amplitude high-freq
modes), the field is produced by an initial prediction + K denoising refinement steps at
exponentially decreasing noise amplitudes, so the network is forced to model all scales.

Unified formulation (same network for all steps, conditioned on step k):
  noise schedule:  sigma_k = sigma_min ** (k / K)          for k = 1..K   (sigma_0 := 0)
  step k = 0  -> predict the SIGNAL (the field) from the conditioning        [MSE to target]
  step k >= 1 -> predict the NOISE added to the current estimate             [MSE to eps]

Training (per-sample k ~ Uniform{0..K}):
  k = 0 :  noised_field = 0,                 regress target -> field
  k >= 1:  noised_field = target + sigma_k*eps,  regress target -> eps
Inference:
  u = net(cond, ., 0)                        # initial signal prediction
  for k = 1..K:  u = (u + sigma_k*eps) - sigma_k * net(cond, u + sigma_k*eps, k)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDERefiner(nn.Module):
    def __init__(self, net, num_steps=3, sigma_min=1e-2, out_dim=5):
        """net: a model with signature net(cond, coords, surf_pts, surf_normals,
        noised_field=[B,N,out_dim], step=[B]) -> [B,N,out_dim] (refiner_inputs=True)."""
        super().__init__()
        assert num_steps >= 1 and 0 < sigma_min < 1
        self.net, self.K, self.sigma_min, self.out_dim = net, num_steps, sigma_min, out_dim

    def sigma(self, k):
        """Noise amplitude at step k (tensor or int); sigma(0)=0, sigma(K)=sigma_min."""
        k = torch.as_tensor(k, dtype=torch.float32)
        return torch.where(k == 0, torch.zeros_like(k), self.sigma_min ** (k / self.K))

    def loss(self, cond, coords, surf_pts, surf_normals, target):
        """PDE-Refiner training loss (per-sample random refinement step)."""
        B, N, C = target.shape
        dev = target.device
        k = torch.randint(0, self.K + 1, (B,), device=dev)                 # [B] in {0..K}
        sig = self.sigma(k).to(dev)[:, None, None]                         # [B,1,1]
        eps = torch.randn_like(target)
        is0 = (k == 0)[:, None, None]
        noised = torch.where(is0, torch.zeros_like(target), target + sig * eps)
        regress = torch.where(is0, target, eps)                           # signal if k=0 else noise
        pred = self.net(cond, coords, surf_pts, surf_normals,
                        noised_field=noised, step=k.float())
        return F.mse_loss(pred, regress)

    @torch.no_grad()
    def predict(self, cond, coords, surf_pts, surf_normals, sample=True):
        """Initial prediction + K denoising refinement steps -> [B,N,out_dim]."""
        B, N = coords.shape[:2]
        dev = coords.device
        zeros = torch.zeros(B, N, self.out_dim, device=dev)
        u = self.net(cond, coords, surf_pts, surf_normals,
                     noised_field=zeros, step=torch.zeros(B, device=dev))   # signal (k=0)
        for k in range(1, self.K + 1):
            sig = float(self.sigma_min ** (k / self.K))
            eps = torch.randn_like(u) if sample else torch.zeros_like(u)
            noised = u + sig * eps
            eps_hat = self.net(cond, coords, surf_pts, surf_normals,
                               noised_field=noised, step=torch.full((B,), k, device=dev).float())
            u = noised - sig * eps_hat
        return u
