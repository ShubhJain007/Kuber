"""The data_sample/ npz files must match the contract documented in docs/DATASET.md."""
import glob
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = sorted(glob.glob(os.path.join(ROOT, "data_sample", "*.npz")))

N_NODES, N_SURF = 16384, 2048
REQUIRED = {
    "coords": (N_NODES, 3), "U": (N_NODES, 3), "T": (N_NODES, 1), "p_rgh": (N_NODES, 1),
    "surf_pts": (N_SURF, 3), "surf_normals": (N_SURF, 3),
}
OPTIONAL = {"sdf": (N_NODES,), "sdf_grad": (N_NODES, 3)}   # heatsink-only
# operating-condition scalars present for every device class (heatsink + cold plate)
UNIVERSAL_COND_KEYS = {"Cp", "Pr", "envTemp", "mu", "rho", "solidTemp", "u_in"}


def test_data_sample_present():
    assert SAMPLES, "no npz files found in data_sample/"


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_npz_matches_documented_contract(path):
    name = os.path.basename(path)
    z = np.load(path, allow_pickle=True)

    for key, shape in REQUIRED.items():
        assert key in z.files, f"{name}: missing required key '{key}'"
        assert z[key].shape == shape, f"{name}: {key} shape {z[key].shape} != {shape}"
        assert np.issubdtype(z[key].dtype, np.floating), f"{name}: {key} not float"
        assert np.isfinite(z[key]).all(), f"{name}: {key} has non-finite values"

    for key, shape in OPTIONAL.items():
        if key in z.files:
            assert z[key].shape == shape, f"{name}: {key} shape {z[key].shape} != {shape}"

    # surface normals are documented as outward UNIT normals
    norms = np.linalg.norm(z["surf_normals"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"{name}: normals not unit ({norms.min()}..{norms.max()})"

    # conditions is a JSON object carrying the physics the model is conditioned on
    assert "conditions" in z.files
    cond = json.loads(str(z["conditions"]))
    assert isinstance(cond, dict) and cond, f"{name}: conditions not a non-empty dict"
    missing = UNIVERSAL_COND_KEYS - set(cond)
    assert not missing, f"{name}: conditions missing {missing}"
