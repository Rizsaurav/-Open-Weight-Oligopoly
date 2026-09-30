# The Open-Weight Oligopoly

An IEEE-format audit of the Hugging Face model ecosystem via its public API: 20,000 models, trailing-30-day downloads.

**Status:** independent, unpublished, not peer-reviewed.

## Headline findings

- 2.84B trailing-30-day downloads across 20,000 models
- Download Gini 0.918: top 100 models take 48.4%, top 1,000 take 81.2%
- Publisher HHI only 375 (organizationally unconcentrated)
- Top-10 org download share fell from 78.6% (2022) to 53.7% (2026): decentralizing
- Median model size grew from 124M to 12B parameters

## Reproduce

Requires `IEEEtran.cls` (standard IEEE class, not vendored) alongside the `.tex` source.

```bash
pip install -r pipeline/requirements.txt   # if present; else: requests pandas numpy scipy scikit-learn matplotlib
python pipeline/01_collect.py        # pull model list from HF API
python pipeline/02_collect_detail.py # per-model detail pages
python pipeline/03_clean.py          # assertion-gated cleaning
python pipeline/04_analyses.py      # Gini, HHI, regressions, clustering
python pipeline/05_figures.py       # figures/
python pipeline/06_tables.py        # tables/
pdflatex hf-oligopoly-ieee.tex
```

Every pipeline stage exits nonzero on assertion mismatch. Raw API dumps (`data/raw_detail/`, `data/raw_pages/`) are excluded from version control; `data/models_clean.csv` is the analysis-ready dataset.

## Layout

- `hf-oligopoly-ieee.pdf` — the compiled 5-page paper
- `hf-oligopoly-ieee.tex` — LaTeX source
- `pipeline/` — 01-06 reproducible pipeline
- `figures/` — paper figures
- `tables/` — LaTeX tables
- `results/hf_results.json` — headline numbers
- `data/` — cleaned dataset + cleaning report
