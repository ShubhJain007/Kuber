---
title: Kuber Live Demo
emoji: 🌡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: other
---

# Kuber — live inference demo (Hugging Face Docker Space)

Runs the **full interactive workflow live**: the editable-geometry UI (`static/index.html`)
served by a FastAPI backend that runs **SurfaceGeoTransolver on CPU** (~1.5 s/case) and returns
the predicted field beside the CFD ground truth. This is the same app as
[github.com/ShubhJain007/Kuber](https://github.com/ShubhJain007/Kuber), packaged for a Space.

## One-time setup

The image is code-only; the **model + cases** live in your own HF **Dataset** repo (kept out of the
image so git stays small). The bundle has this layout — assemble it with the helper (copies only the
~2.3k `.npz` the split references, not the full 1.4 GB corpus):

```
multigeo.pt      # the trained checkpoint
splits.json      # the split file
cases/<id>.npz   # only the cases the split names (train cases fit the normalizers; test cases are the presets)
```

```bash
# assemble locally, then upload to your dataset repo (needs: huggingface-cli login)
python prepare_assets.py \
    --ckpt   ~/cfd_thermal_mvp/outputs/multigeo.pt \
    --corpus ~/cfd_thermal_mvp/demo_data/multigeo_corpus \
    --splits ~/cfd_thermal_mvp/demo_data/multigeo_splits_demo.json \
    --out    ~/kuber-assets \
    --push   <you>/kuber-assets
```

Then, in the Space **Settings → Variables and secrets**, set:

| name | value |
|---|---|
| `KUBER_HF_REPO` | `you/kuber-assets` |
| `KUBER_HF_REPO_TYPE` | `dataset` |
| `HF_TOKEN` *(secret)* | a read token, if the assets repo is private |

On boot, `boot.py` downloads the assets and launches the server; the app loads the model once and is
then interactive.

## Deploy

```bash
# create a Docker Space at huggingface.co/new-space (SDK: Docker), then:
git clone https://huggingface.co/spaces/you/kuber-live && cd kuber-live
cp -r /path/to/Kuber/deploy/hf-space/* .
git add . && git commit -m "kuber live demo" && git push
```

## Notes

- **CPU-only.** No GPU needed; the model runs on CPU. `nvidia-physicsnemo` provides the GeoTransolver
  core (Warp falls back to its CPU backend). If the CPU build of `physicsnemo` is troublesome, a
  scale-to-zero **GPU serverless** host (e.g. Modal) is the robust alternative — same `app.py`.
- **First boot** downloads the corpus and fits normalizers (~1–2 min); subsequent requests are fast.
  A future optimization is to ship precomputed normalizers so only the model + a few preset cases are
  needed (drops the corpus download).
- Free HF CPU Spaces sleep when idle and cold-start on the next visit.
