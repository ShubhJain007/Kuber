# Contributing to Kuber

Thanks for your interest. Kuber is a young, honest benchmark — contributions that add
rigor (new baselines, harder splits, better data) or catch over-claims are all welcome.

## Ways to contribute

- **Submit a leaderboard result.** Train any model on the SIMSHIFT split (or our corpus) and
  open a PR adding a row. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md#submitting-a-result) for the
  required disclosures (UDA? external pretraining? parameter count?) and the metric script.
- **Add a baseline model.** A clean, self-contained model + a one-command train/eval recipe.
- **Extend the dataset.** New fluids, shapes, or a new device class via `datagen/`. Include the
  generation command and a convergence check.
- **Report a bug or an over-claim.** Open an issue. We would rather fix a number than defend it.

## Ground rules

- **No licensed or scraped data.** Everything must be self-generated or from a source whose license
  permits redistribution. Our corpus is OpenFOAM-generated with clean provenance (no licensed/scraped data).
- **Reproducibility.** A result must come with the exact command and environment that produced it.
- **Honesty about domain adaptation and pretraining.** State plainly whether a number used UDA or
  any external data. The leaderboard tracks these columns for a reason.

## Development

```bash
pip install -r requirements.txt
python -m py_compile kuber/*.py     # syntax check
pip install pytest && pytest -q     # run the test suite in tests/
```

The suite (`tests/`) covers the model's forward shape contract, the nRMSE/relL2 metric
definitions, the `data_sample/` data contract, and the numerical-stability harness. Tests that
need the GeoTransolver core (physicsnemo) self-skip where it is not installed, so the suite runs
on a plain CPU box and in CI.

Keep PRs focused. For anything large (a new device class, a hosted leaderboard), open an issue first
so we can align on the design.
