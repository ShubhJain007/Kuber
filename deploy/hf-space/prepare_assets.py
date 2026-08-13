#!/usr/bin/env python3
"""Assemble the asset bundle for the Kuber live Space and (optionally) push it to HF Hub.

Copies the trained checkpoint, the splits file, and ONLY the ``.npz`` cases referenced
by the split into ``<out>/`` — the layout ``boot.py`` expects:

    <out>/multigeo.pt
    <out>/splits.json
    <out>/cases/<id>.npz

The referenced medium split is ~2.3k cases (~1 GB); the full corpus (3.4k files, 1.4 GB)
is not needed. The engine fits its normalizers on the training split, so those cases must
be present — this copies exactly the set the split names, nothing more.

Examples
--------
  # 1) assemble the bundle locally
  python prepare_assets.py \
      --ckpt   ~/cfd_thermal_mvp/outputs/multigeo.pt \
      --corpus ~/cfd_thermal_mvp/demo_data/multigeo_corpus \
      --splits ~/cfd_thermal_mvp/demo_data/multigeo_splits_demo.json \
      --out    ~/kuber-assets

  # 2) assemble AND upload to your HF *dataset* repo (needs `huggingface-cli login`)
  python prepare_assets.py ... --out ~/kuber-assets --push <you>/kuber-assets
"""
import argparse
import json
import shutil
from pathlib import Path


def referenced_ids(node):
    """Collect every case id (leaf string) anywhere under a splits (sub)tree."""
    ids = set()

    def walk(x):
        if isinstance(x, str):
            ids.add(x)
        elif isinstance(x, list):
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            for e in x.values():
                walk(e)

    walk(node)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="path to multigeo.pt")
    ap.add_argument("--corpus", required=True, help="dir of <id>.npz cases")
    ap.add_argument("--splits", required=True, help="splits json")
    ap.add_argument("--out", required=True, help="output bundle dir")
    ap.add_argument("--difficulty", default=None,
                    help="limit to one difficulty key (default: every id in the file)")
    ap.add_argument("--push", default=None,
                    help="HF dataset repo id to upload the bundle to, e.g. you/kuber-assets")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    cases = out / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    corpus = Path(a.corpus).expanduser()

    sp = json.loads(Path(a.splits).expanduser().read_text())
    node = sp[a.difficulty] if a.difficulty else sp
    ids = referenced_ids(node)
    print(f"[assets] {len(ids)} cases referenced by the split")

    shutil.copy2(Path(a.ckpt).expanduser(), out / "multigeo.pt")
    shutil.copy2(Path(a.splits).expanduser(), out / "splits.json")

    tot = miss = copied = 0
    for i in sorted(ids):
        src = corpus / f"{i}.npz"
        if not src.exists():
            miss += 1
            continue
        dst = cases / f"{i}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
        tot += dst.stat().st_size
        copied += 1

    ck_mb = (out / "multigeo.pt").stat().st_size / 1e6
    print(f"[assets] bundle ready at {out}")
    print(f"           multigeo.pt   {ck_mb:.0f} MB")
    print(f"           splits.json")
    print(f"           cases/        {copied} files, {tot / 1e6:.0f} MB"
          + (f"   (WARNING: {miss} referenced npz missing from corpus)" if miss else ""))

    if a.push:
        from huggingface_hub import HfApi, create_repo
        create_repo(a.push, repo_type="dataset", exist_ok=True)
        print(f"[push] uploading {out} -> dataset:{a.push}  (large; can take a while)")
        HfApi().upload_folder(folder_path=str(out), repo_id=a.push, repo_type="dataset")
        print(f"[push] done -> https://huggingface.co/datasets/{a.push}")
        print("[push] set the Space secret  KUBER_HF_REPO = " + a.push)


if __name__ == "__main__":
    main()
