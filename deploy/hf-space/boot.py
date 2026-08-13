#!/usr/bin/env python3
"""Container entrypoint for the Kuber live-inference Space.

Fetches the model + corpus + splits from a Hugging Face Hub repo (set KUBER_HF_REPO),
points the app at them, and launches the FastAPI server on the platform port. The
assets are NOT bundled in the image — they live in your own HF Model/Dataset repo.

Your HF assets repo must contain:
  multigeo.pt      the trained checkpoint
  cases/           the .npz corpus (needed for correct training normalizers)
  splits.json      the split file (difficulty -> {src,tgt} -> {train,val,test})
"""
import os
from pathlib import Path

ASSETS = Path(os.environ.get("KUBER_ASSETS", "assets"))
repo = os.environ.get("KUBER_HF_REPO", "").strip()

if repo and not (ASSETS / "multigeo.pt").exists():
    from huggingface_hub import snapshot_download
    print(f"[boot] downloading assets from {repo} -> {ASSETS}", flush=True)
    snapshot_download(repo_id=repo, repo_type=os.environ.get("KUBER_HF_REPO_TYPE", "dataset"),
                      local_dir=str(ASSETS), token=os.environ.get("HF_TOKEN"))

os.environ.setdefault("KUBER_CKPT", str(ASSETS / "multigeo.pt"))
os.environ.setdefault("KUBER_DATA", str(ASSETS / "cases"))
os.environ.setdefault("KUBER_SPLITS", str(ASSETS / "splits.json"))

import uvicorn
uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
