"""The numerical-stability harness (edge_proof) must return the documented summary structure,
and its gradient-magnitude estimator must be correct on fields with a known slope."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kuber.edge_proof import grad_mag, analyze


def test_grad_mag_shape_and_linear_slope():
    # f = 2*x on a uniform line -> local slope 2 everywhere
    n = 200
    coords = np.zeros((n, 3), np.float64)
    coords[:, 0] = np.linspace(0.0, 1.0, n)
    g = grad_mag(2.0 * coords[:, 0], coords, k=4)
    assert g.shape == (n,)
    assert np.isfinite(g).all()
    assert abs(float(np.median(g)) - 2.0) < 0.3


def test_grad_mag_constant_field_is_zero():
    coords = np.random.default_rng(0).standard_normal((100, 3))
    assert np.allclose(grad_mag(np.ones(100), coords, k=6), 0.0)


def test_analyze_returns_expected_structure():
    C = 5
    B, N = 2, 120
    rng = np.random.default_rng(1)
    split = dict(
        coords=rng.standard_normal((B, N, 3)).astype(np.float32),
        feats=rng.standard_normal((B, N, 4)).astype(np.float32),
        y=rng.standard_normal((B, N, C)).astype(np.float32),
        sdf=rng.standard_normal((B, N)).astype(np.float32),
        surf_pts=rng.standard_normal((B, 8, 3)).astype(np.float32),
        surf_normals=rng.standard_normal((B, 8, 3)).astype(np.float32),
    )

    class Stub(torch.nn.Module):
        def forward(self, cond, pos, sp, sn):
            return torch.zeros(pos.shape[0], pos.shape[1], C)

    summ = analyze(Stub(), split, geom_mode="surface", dev="cpu", ystd=np.ones(C), tag="test")

    expected = {"edge_ratio", "bulk_ratio", "steep_p999_ratio", "max_ratio", "explosion_frac",
                "value_overshoot_frac", "value_worst_overshoot", "edge_gp", "edge_gg",
                "T_rmse_edge", "nan_inf_total", "cases"}
    assert expected.issubset(summ.keys()), expected - set(summ.keys())
    for k, v in summ.items():
        assert np.isfinite(v), (k, v)
    assert summ["cases"] == B
    assert summ["nan_inf_total"] == 0          # a finite (zero) prediction has no NaN/Inf


def test_analyze_none_split_returns_none():
    assert analyze(None, None, "surface", "cpu", np.ones(5), "x") is None
