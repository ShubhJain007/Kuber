"""SurfaceGeoTransolver: surface-geometry branch + GeoTransolver core.

Geometry wiring (`geom_wiring`):
  "concat" — per-point geometry descriptor concatenated into GeoTransolver's local_embedding
             (shallow: geometry enters once at the input).
  "deep"   — descriptor fed as GeoTransolver's `geometry` input, which the core projects to
             context and cross-attends in EVERY GALE block (AB-UPT-style deep conditioning).

Refiner inputs (`refiner_inputs=True`, used by PDE-Refiner): the model additionally ingests a
per-node noised field [B,N,out_dim] and a refinement-step scalar (sinusoidal-embedded), so the
same network serves as both the signal predictor (step 0) and the denoiser (steps >=1).
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

from kuber.surface_model import SurfaceGeometryEncoder, LocalSurfaceCrossAttention


def _import_geotransolver(pnemo_path=None):
    if pnemo_path and pnemo_path not in sys.path:
        sys.path.insert(0, pnemo_path)
    from physicsnemo.experimental.models.geotransolver.geotransolver import GeoTransolver
    return GeoTransolver


def sinusoidal_embedding(t, dim):
    """Standard diffusion-style timestep embedding. t:[B] float -> [B,dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1))
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SurfaceGeoTransolver(nn.Module):
    def __init__(self, cond_dim, out_dim=5, geom_wiring="concat", refiner_inputs=False,
                 step_emb_dim=16,
                 # surface branch
                 d_surf=128, n_surf_layers=2, n_surf_head=4, d_geo=64, geo_head=4, k=16,
                 # GeoTransolver core
                 n_hidden=256, n_layers=12, n_head=8, slice_num=64,
                 include_local_features=True, n_hidden_local=32,
                 radii=(0.05, 0.25), neighbors=(8, 32),
                 pnemo_path=None):
        super().__init__()
        assert geom_wiring in ("concat", "deep")
        GeoTransolver = _import_geotransolver(pnemo_path)
        self.cond_dim, self.d_geo, self.out_dim = cond_dim, d_geo, out_dim
        self.geom_wiring, self.refiner_inputs = geom_wiring, refiner_inputs
        self.step_emb_dim = step_emb_dim if refiner_inputs else 0

        # "concat" builds our own shallow surface encoder + cross-attention (uses normals);
        # "deep" uses GeoTransolver's NATIVE geometry branch on the raw surface point cloud
        # (per-block cross-attention, AB-UPT-style) — no custom modules needed.
        if geom_wiring == "concat":
            self.encoder = SurfaceGeometryEncoder(d=d_surf, n_layers=n_surf_layers, n_head=n_surf_head)
            self.cross = LocalSurfaceCrossAttention(d_tok=d_surf, d_out=d_geo, n_head=geo_head, k=k)
        if refiner_inputs:
            self.step_mlp = nn.Sequential(nn.Linear(step_emb_dim, step_emb_dim), nn.GELU(),
                                          nn.Linear(step_emb_dim, step_emb_dim))

        # local_embedding = cond + (descriptor if concat) + (noised field + step emb if refiner)
        extra = (out_dim + self.step_emb_dim) if refiner_inputs else 0
        func_dim = cond_dim + extra + (d_geo if geom_wiring == "concat" else 0)
        func_dim = max(func_dim, 1)             # GeoTransolver needs functional_dim >= 1
        # geometry is always 3D coords: volume coords (concat) or surface cloud (deep)
        self.core = GeoTransolver(
            functional_dim=func_dim, out_dim=out_dim, geometry_dim=3,
            n_layers=n_layers, n_hidden=n_hidden, n_head=n_head, slice_num=slice_num,
            use_te=False, include_local_features=include_local_features,
            n_hidden_local=n_hidden_local, radii=list(radii), neighbors_in_radius=list(neighbors))

    def forward(self, cond, vol_coords, surf_pts, surf_normals, noised_field=None, step=None):
        B, N, _ = vol_coords.shape
        parts = []
        if cond is not None and cond.shape[-1] > 0:
            parts.append(cond)

        if self.geom_wiring == "concat":
            Zg = self.encoder(surf_pts, surf_normals)          # [B,Ns,d_surf]
            geo = self.cross(vol_coords, surf_pts, Zg)         # [B,N,d_geo] learned descriptor
            parts.append(geo)
            geom_in = vol_coords                               # core geometry = volume coords
        else:  # deep: raw surface cloud drives the core's native geometry cross-attention
            geom_in = surf_pts                                 # [B,Ns,3]

        if self.refiner_inputs:
            assert noised_field is not None and step is not None, "refiner needs noised_field + step"
            parts.append(noised_field)                         # [B,N,out_dim]
            se = self.step_mlp(sinusoidal_embedding(step, self.step_emb_dim))   # [B,step_emb_dim]
            parts.append(se[:, None, :].expand(-1, N, -1))     # broadcast per node

        le = torch.cat(parts, dim=-1) if parts else \
            torch.ones(B, N, 1, device=vol_coords.device, dtype=vol_coords.dtype)
        return self.core(local_embedding=le, local_positions=vol_coords, geometry=geom_in)
