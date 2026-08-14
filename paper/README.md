# Paper

**KuberNet: A Geometry-General Surrogate for Conjugate Heat Transfer with Boundary-Layer Attention** — formatted as an
**IEEE conference (ICRA) paper** (two-column, Times, `IEEEtran`).

| file | what it is |
|---|---|
| [`kuber.pdf`](kuber.pdf) | the paper (read this) — renders inline on GitHub |
| `kuber.tex` | LaTeX source, `\documentclass[conference]{IEEEtran}` — compiles with `pdflatex` or on [Overleaf](https://overleaf.com) |
| `kuber.html` | the IEEE-styled HTML build used to render the PDF (via `chromium --print-to-pdf`) |

Figures are pulled from [`../assets/sim/`](../assets/sim). Every number in the paper is measured and
reproducible with the code in this repository; see [`../docs/RESULTS.md`](../docs/RESULTS.md) and the
machine-readable [`../results/`](../results).

Build the PDF from LaTeX:

```bash
cd paper && pdflatex kuber.tex && pdflatex kuber.tex
```

or from the HTML (no LaTeX needed):

```bash
chromium --headless --no-pdf-header-footer --print-to-pdf=kuber.pdf paper/kuber.html
```

## Cite

```bibtex
@techreport{kuber2026,
  title  = {KuberNet: A Geometry-General Surrogate for Conjugate Heat Transfer with Boundary-Layer Attention},
  author = {Jain, Shubh},
  year   = {2026},
  institution = {Kuber.ai},
  url    = {https://github.com/ShubhJain007/Kuber}
}
```
