"""AB-UPT-inspired local surface-geometry conditioning (clean-room, Apache-safe).

Two modules:
  SurfaceGeometryEncoder      : surface (points+normals) -> geometry tokens via MLP + self-attn
  LocalSurfaceCrossAttention  : each volume point cross-attends to its K-nearest surface
                                tokens (with RELATIVE positions) -> per-point geometry descriptor

The per-point descriptor is a learned, local, orientation-aware replacement for the scalar
SDF. Locality + relative positions make it compositional -> the intended OOD-generalization
lever (a 10-fin heatsink is locally "more of the same" fin-gap patches seen at 5-8 fins).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class SurfaceGeometryEncoder(nn.Module):
    """(points[B,Ns,3], normals[B,Ns,3]) -> geometry tokens [B,Ns,d]. Permutation-equivariant."""

    def __init__(self, d=128, n_layers=2, n_head=4):
        super().__init__()
        self.embed = _mlp(6, d, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_head, dim_feedforward=2 * d,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d = d

    def forward(self, points, normals):
        assert points.shape == normals.shape and points.shape[-1] == 3, \
            f"expected points/normals [B,Ns,3], got {points.shape}/{normals.shape}"
        x = self.embed(torch.cat([points, normals], dim=-1))     # [B,Ns,d]
        return self.enc(x)                                        # [B,Ns,d]


def knn_idx(query, ref, k):
    """k nearest `ref` points for each `query` point (squared-L2).
    query[B,N,3], ref[B,M,3] -> idx[B,N,k] (into M)."""
    d2 = torch.cdist(query, ref)                                 # [B,N,M]
    k = min(k, ref.shape[1])
    return d2.topk(k, dim=-1, largest=False).indices             # [B,N,k]


class LocalSurfaceCrossAttention(nn.Module):
    """Volume points cross-attend to their K-nearest surface tokens (relative-position keys).
    (vol_coords[B,N,3], surf_pts[B,Ns,3], surf_tok[B,Ns,d]) -> descriptor [B,N,d_out]."""

    def __init__(self, d_tok=128, d_out=128, n_head=4, k=16):
        super().__init__()
        assert d_out % n_head == 0, "d_out must be divisible by n_head"
        self.k, self.n_head, self.dh = k, n_head, d_out // n_head
        self.q = _mlp(3, d_out, d_out)                           # volume coord -> query
        self.rel = _mlp(3, d_out, d_out)                         # relative position -> key/val bias
        self.kproj = nn.Linear(d_tok, d_out)
        self.vproj = nn.Linear(d_tok, d_out)
        self.out = nn.Linear(d_out, d_out)

    def forward(self, vol_coords, surf_pts, surf_tok):
        B, N, _ = vol_coords.shape
        Ns = surf_pts.shape[1]
        assert surf_tok.shape[:2] == (B, Ns), f"token/point mismatch {surf_tok.shape} vs {surf_pts.shape}"
        idx = knn_idx(vol_coords, surf_pts, self.k)              # [B,N,k]
        k = idx.shape[-1]
        gp = idx.unsqueeze(-1).expand(-1, -1, -1, 3)             # [B,N,k,3]
        neigh_pts = torch.gather(surf_pts.unsqueeze(1).expand(-1, N, -1, -1), 2, gp)   # [B,N,k,3]
        gt = idx.unsqueeze(-1).expand(-1, -1, -1, surf_tok.shape[-1])
        neigh_tok = torch.gather(surf_tok.unsqueeze(1).expand(-1, N, -1, -1), 2, gt)   # [B,N,k,dtok]

        rel = vol_coords.unsqueeze(2) - neigh_pts                # [B,N,k,3] relative pos (local frame)
        rbias = self.rel(rel)                                    # [B,N,k,d_out]
        q = self.q(vol_coords)                                   # [B,N,d_out]
        keys = self.kproj(neigh_tok) + rbias                     # [B,N,k,d_out]
        vals = self.vproj(neigh_tok) + rbias                     # [B,N,k,d_out]

        # multi-head local attention over the k neighbors
        q = q.view(B, N, self.n_head, self.dh)                   # [B,N,H,dh]
        keys = keys.view(B, N, k, self.n_head, self.dh)
        vals = vals.view(B, N, k, self.n_head, self.dh)
        logits = torch.einsum("bnhd,bnkhd->bnhk", q, keys) / (self.dh ** 0.5)   # [B,N,H,k]
        attn = logits.softmax(dim=-1)
        ctx = torch.einsum("bnhk,bnkhd->bnhd", attn, vals).reshape(B, N, -1)    # [B,N,d_out]
        return self.out(ctx)                                     # [B,N,d_out]
