"""nRMSE / relL2 / physical-RMSE against hand-computed values, and a check that the metrics
module equals what the training loop's `evaluate` accumulates batch-wise (guards drift)."""
import numpy as np
import pytest

from kuber.metrics import per_channel_nrmse, rel_l2, rmse_physical, mean_nrmse


def test_nrmse_hand_computed():
    # 4 points, 2 channels. ch0 errors [3,4,0,0] -> mean sq = 25/4 = 6.25 -> 2.5
    #                        ch1 errors [1,1,1,1] -> mean sq = 1.0        -> 1.0
    gt = np.zeros((4, 2))
    pred = np.array([[3., 1.], [4., 1.], [0., 1.], [0., 1.]])
    assert np.allclose(per_channel_nrmse(pred, gt), [2.5, 1.0])


def test_nrmse_zero_on_identical():
    x = np.random.default_rng(0).standard_normal((3, 100, 5))
    assert np.allclose(per_channel_nrmse(x, x), 0.0)


def test_nrmse_flattens_batch_and_point_dims():
    pred = np.random.default_rng(1).standard_normal((2, 10, 5))
    gt = np.random.default_rng(2).standard_normal((2, 10, 5))
    assert np.allclose(per_channel_nrmse(pred, gt),
                       per_channel_nrmse(pred.reshape(-1, 5), gt.reshape(-1, 5)))


def test_rel_l2_hand_computed():
    gt = np.array([[1., 1.], [1., 1.]])
    pred = np.array([[2., 1.], [1., 1.]])          # ch0 err [1,0] -> sqrt(1/2); ch1 err 0
    assert np.allclose(rel_l2(pred, gt), [np.sqrt(0.5), 0.0])


def test_rmse_physical_denormalizes_by_ystd():
    gt = np.zeros((2, 1))
    pred = np.array([[1.], [-1.]])                 # err [1,-1] * ystd 3 -> [3,-3] -> rmse 3
    assert np.allclose(rmse_physical(pred, gt, ystd=[3.0]), [3.0])


def test_mean_nrmse_is_channel_mean():
    gt = np.zeros((4, 2))
    pred = np.array([[3., 1.], [4., 1.], [0., 1.], [0., 1.]])
    assert np.isclose(mean_nrmse(pred, gt), (2.5 + 1.0) / 2)


def test_matches_training_loop_evaluate():
    """`evaluate` (batch-wise accumulation) must agree with the metrics module on the same data.
    Uses a deterministic stub model + geom_mode='surface' so no GeoTransolver/physicsnemo is needed."""
    torch = pytest.importorskip("torch")
    try:
        from kuber.train_simshift import evaluate
    except Exception as e:                          # pragma: no cover - env-dependent
        pytest.skip(f"train_simshift import unavailable: {e}")

    rng = np.random.default_rng(3)
    B, N, C = 3, 40, 5
    y = rng.standard_normal((B, N, C)).astype(np.float32)
    split = dict(
        coords=rng.standard_normal((B, N, 3)).astype(np.float32),
        feats=rng.standard_normal((B, N, 4)).astype(np.float32),
        y=y,
        surf_pts=rng.standard_normal((B, 8, 3)).astype(np.float32),
        surf_normals=rng.standard_normal((B, 8, 3)).astype(np.float32),
        sdf=rng.standard_normal((B, N)).astype(np.float32),
    )

    class Stub(torch.nn.Module):                    # predicts zeros, deterministic
        def forward(self, cond, pos, sp, sn):
            return torch.zeros(pos.shape[0], pos.shape[1], C)

    channels = ["U_x", "U_y", "U_z", "T", "p_rgh"]
    m = evaluate(Stub(), split, dev="cpu", bs=2, n_points=N, ymean=np.zeros(C),
                 ystd=np.ones(C), channels=channels, geom_mode="surface")

    expect = per_channel_nrmse(np.zeros((B, N, C)), y)   # error = -y
    got = np.array([m["nRMSE"][c] for c in channels])
    assert np.allclose(got, expect, atol=1e-5), (got, expect)
