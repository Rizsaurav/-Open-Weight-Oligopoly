"""04_analyses.py — Concentration and adoption analyses on the cleaned HF sample.

Reads data/models_clean.csv (one row per model; see 03_clean.py).
Writes results/hf_results.json.

A1  Download descriptives: Gini, top-100 / top-1,000 shares, median/IQR.
A2  Org concentration: top-10 orgs, CR1/CR4/CR10, HHI on download shares.
A3  Parameters (detail subsample): coverage, median params by year, params Gini.
A4  License-class shares by creation year.
A5  OLS (HC3): log(downloads) ~ org_type + license_class + log(age) +
    pipeline_tag (top tags + other); second spec on detail subsample adds
    log(params).
A6  Logistic: P(breakout) ~ same features; odds ratios, p-values, pseudo-R2.
    breakout = downloads >= 90th percentile of the sample.
A7  Random forest vs logistic, repeated stratified 5-fold CV (10 repeats):
    AUC mean +/- sd vs chance baseline.
A8  K-means archetypes on standardized numeric features; silhouette for
    k = 2..6, honest choice of k.
A9  Temporal: models per quarter; top-10 org download share by creation
    year (centralization since 2022).

Every block ends with assertion gates; the script exits nonzero on failure.
"""
import json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
rng = np.random.RandomState(7)

def gini(x):
    """Gini coefficient via the O(n log n) sorted formula (no O(n^2) matrix)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    assert n > 0 and (x >= 0).all() and x.sum() > 0
    return float((2 * np.sum((np.arange(1, n + 1)) * x)) / (n * x.sum()) - (n + 1) / n)

df = pd.read_csv(os.path.join(DATA, "models_clean.csv"))
res = {"n": int(len(df))}
assert len(df) > 15000, f"sample too small: {len(df)}"
assert df.id.nunique() == len(df), "duplicate ids in cleaned frame"
assert (df.downloads >= 0).all(), "negative downloads"

# ================= A1: download descriptives =================
d = df.downloads.sort_values(ascending=False).to_numpy(dtype=float)
n = len(d)
tot = d.sum()
assert tot > 0, "zero total downloads"
gini_dl = gini(d)
top100_share = d[:100].sum() / tot
top1000_share = d[:1000].sum() / tot
res["A1_descriptives"] = {
    "n": n, "total_downloads": int(tot),
    "median": float(np.median(d)), "q25": float(np.percentile(d, 25)),
    "q75": float(np.percentile(d, 75)), "mean": float(d.mean()),
    "max": int(d.max()), "max_id": df.loc[df.downloads.idxmax(), "id"],
    "gini": round(float(gini_dl), 4),
    "top100_share": round(float(top100_share), 4),
    "top1000_share": round(float(top1000_share), 4),
    "pct_zero": round(float((d == 0).mean()), 4)}
assert 0 <= gini_dl <= 1, f"Gini out of range: {gini_dl}"
print(f"A1: n={n}, Gini={gini_dl:.4f}, top100={top100_share:.1%}, top1000={top1000_share:.1%}")

# ================= A2: org concentration =================
g = df.groupby("org").agg(models=("id", "size"), downloads=("downloads", "sum"),
                          likes=("likes", "sum")).sort_values("downloads", ascending=False)
shares = g.downloads / g.downloads.sum()
hhi = float((shares ** 2).sum() * 10000)
top10 = g.head(10)
res["A2_org_concentration"] = {
    "n_orgs": int(len(g)),
    "hhi_downloads": round(hhi, 0),
    "cr1": round(float(shares.iloc[0]), 4), "cr4": round(float(shares.iloc[:4].sum()), 4),
    "cr10": round(float(shares.iloc[:10].sum()), 4),
    "top10": [{"org": o, "models": int(r.models), "downloads": int(r.downloads),
               "share": round(float(r.downloads) / float(g.downloads.sum()), 4)}
              for o, r in top10.iterrows()]}
assert abs(shares.sum() - 1) < 1e-9, "org shares do not sum to 1"
print(f"A2: {len(g)} orgs, HHI={hhi:.0f}, CR1={shares.iloc[0]:.1%}, CR10={shares.iloc[:10].sum():.1%}")

# ================= A3: parameters (detail subsample) =================
import glob as _glob
detail_n = len(_glob.glob(os.path.join(DATA, "raw_detail", "*.json")))
det = df[df.has_params == 1].copy()
cov = len(det) / len(df)
cov_detail = len(det) / detail_n if detail_n else 0.0
res["A3_params"] = {"detail_n": int(detail_n),
                    "params_coverage_full": round(float(cov), 4),
                    "params_coverage_detail": round(float(cov_detail), 4)}
assert len(det) > 500, f"too few models with params: {len(det)}"
assert (det.params > 0).all(), "non-positive params"
p = det.params.sort_values(ascending=False).to_numpy(dtype=float)
pgini = gini(p)
med_by_year = det.groupby("year").params.median()
res["A3_params"].update({
    "n_params": int(len(det)),
    "median_params": float(np.median(p)), "q25": float(np.percentile(p, 25)),
    "q75": float(np.percentile(p, 75)), "max_params": int(p.max()),
    "max_params_id": det.loc[det.params.idxmax(), "id"],
    "gini_params": round(float(pgini), 4),
    "median_params_by_year": {int(k): float(v) for k, v in med_by_year.items()},
    "top10_params_share": round(float(p[:10].sum() / p.sum()), 4)})
print(f"A3: detail_n={detail_n}, params coverage full={cov:.1%} / detail={cov_detail:.1%}, median={np.median(p)/1e6:.1f}M, Gini={pgini:.4f}")

# ================= A4: license trends =================
lic = pd.crosstab(df.year, df.license_class, normalize="index")
res["A4_license_by_year"] = {int(y): {c: round(float(v), 4) for c, v in row.items()}
                             for y, row in lic.iterrows()}
print("A4 license shares by year:"); print(lic.round(3).to_string())

# ================= design matrices =================
df["log_age"] = np.log(df.age_days + 1)
top_tags = df.pipeline_tag.value_counts().head(8).index.tolist()
df["tag_group"] = df.pipeline_tag.fillna("unspecified")
df["tag_group"] = df.tag_group.where(df.tag_group.isin(top_tags + ["unspecified"]), "other")
# explicit reference levels (most common categories) for interpretable coefs
df["org_type"] = pd.Categorical(df.org_type, categories=["community", "corporate", "research"])
df["license_class"] = pd.Categorical(df.license_class,
    categories=["permissive", "copyleft", "restrictive", "other", "unspecified"])
tag_base = df.tag_group.value_counts().index[0]
df["tag_group"] = pd.Categorical(df.tag_group,
    categories=[tag_base] + [c for c in df.tag_group.unique() if c != tag_base])
def dummies(frame, cols):
    return pd.get_dummies(frame[cols], columns=cols, drop_first=True, dtype=float)
X_base = dummies(df, ["org_type", "license_class", "tag_group"])
X_base["log_age"] = df.log_age
y_log = np.log(df.downloads + 1)
assert X_base.shape[0] == len(df) and y_log.notna().all()

# ================= A5: OLS =================
def ols_fit(X, y, name):
    Xc = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, Xc).fit(cov_type="HC3")
    out = {"n": int(len(X)), "r2": round(float(fit.rsquared), 3),
           "adj_r2": round(float(fit.rsquared_adj), 3),
           "params": {k: {"coef": round(float(v), 4), "se": round(float(fit.bse[k]), 4),
                          "p": round(float(fit.pvalues[k]), 4)}
                      for k, v in fit.params.items()}}
    assert 0 <= fit.rsquared <= 1
    return out, fit
res["A5_ols_full"], fit_full = ols_fit(X_base, y_log, "full")
print(f"A5 full: n={len(X_base)}, R2={res['A5_ols_full']['r2']}")
# subsample with params (re-selected after design features were added to df)
ds = df[df.has_params == 1].copy()
Xs = dummies(ds, ["org_type", "license_class", "tag_group"])
Xs["log_age"] = np.log(ds.age_days + 1)
Xs["log_params"] = np.log(ds.params)
ys = np.log(ds.downloads + 1)
res["A5_ols_params"], _ = ols_fit(Xs, ys, "params")
print(f"A5 params-spec: n={len(Xs)}, R2={res['A5_ols_params']['r2']}")

# ================= A6: logistic breakout =================
thr = df.downloads.quantile(0.90)
df["breakout"] = (df.downloads >= thr).astype(int)
assert 0.08 < df.breakout.mean() < 0.12, f"breakout rate off: {df.breakout.mean()}"
Xl = sm.add_constant(X_base, has_constant="add")
logit = sm.Logit(df.breakout, Xl).fit(disp=0, maxiter=500)
assert logit.mle_retvals["converged"], "logistic did not converge"
res["A6_logistic"] = {
    "n": int(len(Xl)), "threshold": int(thr), "breakout_rate": round(float(df.breakout.mean()), 4),
    "pseudo_r2": round(float(logit.prsquared), 3),
    "odds_ratios": {k: {"or": round(float(np.exp(v)), 3),
                        "ci95": [round(float(np.exp(logit.conf_int().loc[k, 0])), 3),
                                 round(float(np.exp(logit.conf_int().loc[k, 1])), 3)],
                        "p": round(float(logit.pvalues[k]), 4)}
                    for k, v in logit.params.items()}}
print(f"A6: breakout>={thr:,.0f} (rate {df.breakout.mean():.1%}), pseudo-R2={logit.prsquared:.3f}")

# ================= A7: RF vs logistic, repeated CV =================
Xa = X_base.to_numpy(float)
ya = df.breakout.to_numpy()
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=7)
rf = RandomForestClassifier(n_estimators=400, random_state=7, n_jobs=-1)
lr = LogisticRegression(max_iter=5000)
rf_auc = cross_val_score(rf, Xa, ya, cv=cv, scoring="roc_auc")
lr_auc = cross_val_score(lr, Xa, ya, cv=cv, scoring="roc_auc")
base_acc = max(ya.mean(), 1 - ya.mean())
res["A7_cv"] = {
    "rf_auc_mean": round(float(rf_auc.mean()), 3), "rf_auc_sd": round(float(rf_auc.std()), 3),
    "lr_auc_mean": round(float(lr_auc.mean()), 3), "lr_auc_sd": round(float(lr_auc.std()), 3),
    "chance_auc": 0.5, "majority_acc": round(float(base_acc), 3), "n_folds": 50}
rf.fit(Xa, ya)
imp = dict(sorted(zip(X_base.columns, rf.feature_importances_), key=lambda x: -x[1]))
res["A7_cv"]["rf_importance_top8"] = {k: round(float(v), 3) for k, v in list(imp.items())[:8]}
assert rf_auc.mean() > 0.5, "RF not above chance AUC"
print(f"A7: RF AUC={rf_auc.mean():.3f}+/-{rf_auc.std():.3f} | LR AUC={lr_auc.mean():.3f}+/-{lr_auc.std():.3f} | chance AUC=0.5, majority acc={base_acc:.3f}")

# ================= A8: k-means archetypes =================
df["log_org_models"] = np.log(df.groupby("org")["org"].transform("size"))
F = ["log_downloads", "log_likes", "age_years", "log_org_models"]
df["log_downloads"] = np.log(df.downloads + 1)
df["log_likes"] = np.log(df.likes + 1)
df["age_years"] = df.age_days / 365.25
Xm = df[F].copy()
assert Xm.notna().all().all(), "NaN in k-means features"
Xs_ = StandardScaler().fit_transform(Xm)
sil = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=7, n_init=20).fit(Xs_)
    sil[k] = float(silhouette_score(Xs_, km.labels_))
    assert len(set(km.labels_)) == k, f"k={k} collapsed"
res["A8_kmeans"] = {"silhouette": {int(k): round(v, 3) for k, v in sil.items()}}
best_k = max(sil, key=sil.get)
# honest choice: the silhouette maximizer, reported with all k=2..6 scores
chosen_k = best_k
km = KMeans(n_clusters=chosen_k, random_state=7, n_init=20).fit(Xs_)
df["archetype"] = km.labels_
prof = df.groupby("archetype")[F + ["breakout"]].median().round(3)
sizes = df.archetype.value_counts().sort_index()
res["A8_kmeans"].update({
    "chosen_k": int(chosen_k), "best_k": int(best_k),
    "sizes": {int(k): int(v) for k, v in sizes.items()},
    "medians": {int(k): {c: float(v) for c, v in prof.loc[k].items()} for k in prof.index}})
print("A8 silhouette:", {k: round(v, 3) for k, v in sil.items()}, "-> chosen", chosen_k)
print(prof.to_string())

# ================= A9: temporal centralization =================
q = df.groupby("quarter").agg(models=("id", "size"), downloads=("downloads", "sum"))
share_top10_by_year = {}
for y, sub in df.groupby("year"):
    s = sub.groupby("org").downloads.sum()
    share_top10_by_year[int(y)] = round(float(s.sort_values(ascending=False).head(10).sum() / s.sum()), 4)
res["A9_temporal"] = {
    "models_per_quarter": {str(k): int(v) for k, v in q.models.items()},
    "top10_org_share_by_year": share_top10_by_year,
    "median_downloads_by_year": {int(k): float(v) for k, v in df.groupby("year").downloads.median().items()}}
print("A9 top-10 org share by year:", share_top10_by_year)

with open(os.path.join(RES, "hf_results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote results/hf_results.json")
