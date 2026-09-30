# The Open-Weight Oligopoly

An empirical audit of concentration in the Hugging Face model ecosystem, built
from the Hub's public API. Sample: the **top 20,000 models by trailing-30-day
downloads**.

**Status:** independent research. Not peer-reviewed, not submitted for
publication.

## Headline findings

| Finding | Value |
|---|---|
| Models audited | 20,000 |
| Trailing-30-day downloads | 2.84B |
| Download Gini | 0.918 |
| Share of top 100 models | 48.4% |
| Share of top 1,000 models | 81.2% |
| Publisher HHI (download shares) | 375 |
| Top-10 org download share, 2022 cohort | 78.6% |
| Top-10 org download share, 2026 cohort | 53.7% |
| Median model size, 2022 cohort | 124M params |
| Median model size, 2026 cohort | 12B params |

Reading: downloads are brutally concentrated (Gini 0.918) but *organizationally*
the ecosystem is unconcentrated (HHI 375, far below the 1,500 "moderate"
threshold) and decentralizing over time.

## Repository layout

```
hf-oligopoly-ieee.tex        LaTeX source of the 5-page IEEE-format paper
hf-oligopoly-ieee.pdf        Compiled paper (build artifact, committed for convenience)
IEEEtran.cls                 IEEE LaTeX class (v1.8b, LPPL; vendored so the paper builds as-is)
pipeline/
  01_collect.py              Pull the model listing from the HF API -> data/raw_pages/
  02_collect_detail.py       Per-model detail (params, license tags) -> data/raw_detail/
  03_clean.py                Parse + feature-engineer -> data/models_clean.csv
  04_analyses.py             All nine analyses (A1-A9) -> results/hf_results.json
  05_figures.py              Render figures/fig1-fig5.png from results
  06_tables.py               Render tables/t1-t9.tex from results
figures/                     5 publication figures (300 dpi PNG)
tables/                      9 LaTeX table snippets, \input by the .tex source
results/hf_results.json      Every headline number, keyed by analysis block
data/
  models_clean.csv           Analysis-ready dataset: 20,000 rows, one per model
  models_list.json           Raw stage-1 listing (lets you skip 01_collect.py)
  cleaning_report.json       Coverage stats emitted by 03_clean.py
  detail_failures.json       Per-model fetch failures from 02_collect_detail.py
```

`data/raw_pages/` and `data/raw_detail/` (the raw API dumps, ~50 MB) are
excluded from version control by `.gitignore`; they regenerate via steps 01-02.

## Reproduce

### Prerequisites

- Python 3.10+
- `pip install pandas numpy scipy scikit-learn matplotlib statsmodels`
  (steps 01-02 use only the standard library)
- A LaTeX distribution with `pdflatex` (e.g. TeX Live) for the paper

### Full run, from zero

```bash
python3 pipeline/01_collect.py         # ~5 min; 200 pages, 0.65 s between requests
python3 pipeline/02_collect_detail.py  # ~30 min; top 2000 models, 0.75 s between requests
python3 pipeline/03_clean.py           # seconds
python3 pipeline/04_analyses.py        # a few minutes (random forest CV dominates)
python3 pipeline/05_figures.py         # seconds
python3 pipeline/06_tables.py          # seconds
pdflatex hf-oligopoly-ieee.tex         # the 5-page paper
```

### Fast path (no API calls)

`data/models_clean.csv` is committed, so steps 04-06 and the LaTeX build run
without touching the network:

```bash
python3 pipeline/04_analyses.py
python3 pipeline/05_figures.py
python3 pipeline/06_tables.py
pdflatex hf-oligopoly-ieee.tex
```

### What each step does

**01_collect.py** — `GET https://huggingface.co/api/models?limit=100&sort=downloads&direction=-1`,
following the `Link: rel="next"` cursor. The `downloads` field is the trailing
30-day count (distinct from all-time). Sleeps 0.65 s between requests
(~460 per 300 s window, under the 500 limit) and backs off 300 s on HTTP 429.
Writes one JSON file per page to `data/raw_pages/`, then consolidates to
`data/models_list.json`. Assertion gates: every page is a JSON list, every
item has an `id` and a non-negative integer `downloads`, the final frame is
duplicate-free and sorted non-increasing in downloads (the API's stated sort
contract). Flags: `--target` (default 20000), `--limit` (default 100).

**02_collect_detail.py** — `GET https://huggingface.co/api/models/{id}` for the
top 2000 models by downloads. Adds `safetensors.total` (parameter count),
author, and card metadata. Deleted/renamed repos are expected and recorded in
`data/detail_failures.json`; the run fails only if the failure rate exceeds
5%. Assertion gates: returned `id` matches the request, parameter counts are
positive integers when present. Flags: `--top` (default 2000), `--offset`,
`--sleep`.

**03_clean.py** — Parses the list + detail payloads into `data/models_clean.csv`
(one row per model) and writes `data/cleaning_report.json` with coverage
stats. Feature engineering, all rules explicit in the script:
- `org`: the `id` prefix before `/` (or the whole id for legacy single-component ids).
- `org_type`: author-coded, case-insensitive curated lists (`CORPORATE` /
  `RESEARCH` in the script); everything else is `community`.
- `license_class`: from `license:*` tags, first regex match wins, in order:
  `restrictive` (openrail/llama/gemma/non-commercial/…), `copyleft` (gpl/agpl/lgpl/cc-by-sa),
  `permissive` (apache-2.0/mit/bsd/cc-by/…), `other`, `unspecified` (no license tag).
- `age_days` vs the collection timestamp; `year`/`quarter` from `createdAt`;
  `params` from `safetensors.total` with a `has_params` flag.
Assertion gates: row count matches the manifest, ids unique, downloads are
non-negative ints, every `createdAt` parses with year in [2015, 2026],
`age_days >= 0`, params positive where present, categorical columns take only
declared values.

**04_analyses.py** — Nine analysis blocks on `data/models_clean.csv`, writing
`results/hf_results.json`:
- **A1** descriptives: Gini (sorted-formula), top-100 / top-1,000 download shares.
- **A2** org concentration: HHI on download shares, CR1/CR4/CR10, top-10 org table.
- **A3** parameters on the detail subsample: coverage, median params by year, params Gini.
- **A4** license-class shares by creation year.
- **A5** OLS with HC3 robust SE: `log(downloads)` on org type, license class,
  `log(age)`, pipeline-tag dummies; second spec on the detail subsample adds `log(params)`.
- **A6** logistic: P(breakout), where breakout = downloads at or above the
  90th percentile; odds ratios, 95% CIs, pseudo-R².
- **A7** random forest (400 trees) vs logistic, repeated stratified 5-fold CV
  (10 repeats, 50 folds): mean ± sd AUC vs the 0.5 chance baseline.
- **A8** k-means archetypes on standardized numeric features; silhouette for
  k = 2..6, honest choice of the silhouette maximizer.
- **A9** temporal: models per quarter; top-10 org download share by creation year.

Every block ends with assertion gates; the script exits nonzero on failure.

**05_figures.py** — Renders `figures/fig1_lorenz.png` (Lorenz curve),
`fig2_orgs.png` (top-10 orgs), `fig3_rf.png` (RF importance + breakout by org
type), `fig4_time.png` (concentration and median params over time),
`fig5_licenses.png` (license mix by year). Single palette, IEEE two-column
friendly.

**06_tables.py** — Renders `tables/t1_descriptives.tex` … `t9_license.tex`,
one `\begin{table}` block each, `\input` by the paper source. Every number
comes from `results/hf_results.json`; nothing is hand-typed.

### Where the paper's numbers come from

All headline numbers live in `results/hf_results.json`, keyed by block:
Gini and top-k shares under `A1_descriptives`; HHI and concentration ratios
under `A2_org_concentration`; the 78.6% → 53.7% decentralization series under
`A9_temporal.top10_org_share_by_year`; median params by year under
`A3_params.median_params_by_year`. The `.tex` paper and all tables/figures
are generated from this file, so re-running the pipeline reproduces the
paper's numbers exactly.

### `data/models_clean.csv` columns

`id, org, org_type, has_org_prefix, downloads, likes, pipeline_tag,
license_class, created_at, year, quarter, age_days, params, has_params`

`downloads` is the trailing-30-day count at collection time; `params` is null
unless the model was in the top-2000 detail fetch (`has_params = 1`).

## Design decisions and caveats

- The sample is the top 20,000 models **by recent downloads**, not a random
  draw of the Hub: findings describe where attention goes, not the long tail
  of 2M+ repos.
- `downloads` is a 30-day trailing window. Rankings move; re-running 01 later
  gives a different snapshot.
- Parameter counts exist only for the top-2000 detail subsample (A3, and the
  second A5 spec). Everything else uses the full 20,000.
- `org_type` is author-coded from curated org lists in `03_clean.py`
  (corporate vs research vs community) — a judgment call, documented in code.
- License classes come from regexes over `license:*` tags; dual-licensed and
  custom licenses land in `other`/`unspecified`. See `license_class()` in
  `03_clean.py` for the exact rules.
- The 2022-vs-2026 decentralization comparison groups models by *creation*
  year cohort, not by a longitudinal panel.

## License of this repo

Code and analysis are shared for research transparency. `IEEEtran.cls` is
under the LaTeX Project Public License. Model metadata © their respective
Hugging Face Hub publishers.
