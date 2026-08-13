"""Shape contracts for the model on synthetic batches.

The surface branch (SurfaceGeometryEncoder + LocalSurfaceCrossAttention) is pure PyTorch and
always runs. The full SurfaceGeoTransolver needs the GeoTransolver core (physicsnemo) and is
skipped where that is unavailable (e.g. bare CI)."""
import pytest

torch = pytest.importorskip("torch")

from kuber.surface_model import SurfaceGeometryEncoder, LocalSurfaceCrossAttention, knn_idx


def test_surface_encoder_shape():
    enc = SurfaceGeometryEncoder(d=32, n_layers=1, n_head=2).eval()
    B, Ns = 2, 50
    with torch.no_grad():
        out = enc(torch.randn(B, Ns, 3), torch.randn(B, Ns, 3))
    assert out.shape == (B, Ns, 32)
    assert torch.isfinite(out).all()


def test_cross_attention_shape():
    cross = LocalSurfaceCrossAttention(d_tok=32, d_out=16, n_head=2, k=8).eval()
    B, N, Ns = 2, 40, 50
    with torch.no_grad():
        out = cross(torch.randn(B, N, 3), torch.randn(B, Ns, 3), torch.randn(B, Ns, 32))
    assert out.shape == (B, N, 16)
    assert torch.isfinite(out).all()


def test_cross_attention_k_exceeds_num_surface_points():
    # k larger than the available surface points must not crash (knn clamps k)
    cross = LocalSurfaceCrossAttention(d_tok=8, d_out=8, n_head=2, k=16).eval()
    B, N, Ns = 1, 10, 4
    with torch.no_grad():
        out = cross(torch.randn(B, N, 3), torch.randn(B, Ns, 3), torch.randn(B, Ns, 8))
    assert out.shape == (B, N, 8)


def test_knn_idx_shape_and_range():
    idx = knn_idx(torch.randn(2, 10, 3), torch.randn(2, 7, 3), k=5)
    assert idx.shape == (2, 10, 5)
    assert int(idx.min()) >= 0 and int(idx.max()) < 7


def test_full_surface_geotransolver_forward():
    """End-to-end shape contract for the full model (requires the GeoTransolver core)."""
    pytest.importorskip("physicsnemo")
    from kuber.surface_geotransolver import SurfaceGeoTransolver

    model = SurfaceGeoTransolver(
        cond_dim=3, out_dim=5, geom_wiring="concat",
        d_surf=16, n_surf_layers=1, n_surf_head=2, d_geo=8, geo_head=2, k=4,
        n_hidden=16, n_layers=1, n_head=2, slice_num=4,
        n_hidden_local=8, radii=(0.1, 0.3), neighbors=(4, 8),
    ).eval()
    B, N, Ns = 1, 32, 24
    with torch.no_grad():
        out = model(torch.randn(B, N, 3), torch.rand(B, N, 3),
                    torch.rand(B, Ns, 3), torch.randn(B, Ns, 3))
    assert out.shape == (B, N, 5)
    assert torch.isfinite(out).all()
