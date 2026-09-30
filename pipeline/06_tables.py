"""06_tables.py — Generate LaTeX table snippets from results/hf_results.json.

Writes tables/t1_descriptives.tex ... tables/t9_license.tex, one \\begin{table}
block each, included by hf-oligopoly-ieee.tex via \\input. Every number comes
from the results file; nothing is hand-typed.
"""
import json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = os.path.join(HERE, "tables")
os.makedirs(TDIR, exist_ok=True)
res = json.load(open(os.path.join(HERE, "results", "hf_results.json")))

def w(name, body):
    with open(os.path.join(TDIR, name), "w") as f:
        f.write(body + "\n")

A1 = res["A1_descriptives"]
w("t1_descriptives.tex", r"""\begin{table}[t]
\caption{Download distribution across the 20,000-model sample}
\label{tab:descriptives}
\centering
\begin{tabular}{lr}
\toprule
Statistic & Value \\
\midrule
Models & %d \\
Organizations & %d \\
Total 30-day downloads & %s \\
Median downloads & %s \\
Interquartile range & %s--%s \\
Mean downloads & %s \\
Maximum downloads & %s \\
Gini coefficient & %.4f \\
Share of top 100 models & %.1f\%% \\
Share of top 1,000 models & %.1f\%% \\
\bottomrule
\end{tabular}
\end{table}""" % (
    A1["n"], res["A2_org_concentration"]["n_orgs"],
    f"{A1['total_downloads']:,}", f"{A1['median']:,.0f}",
    f"{A1['q25']:,.0f}", f"{A1['q75']:,.0f}", f"{A1['mean']:,.0f}",
    f"{A1['max']:,}",
    A1["gini"], A1["top100_share"] * 100, A1["top1000_share"] * 100))

A2 = res["A2_org_concentration"]
rows = "\n".join(
    f"{t['org'].replace('_', chr(92) + '_')} & {t['models']:,} & "
    f"{t['downloads']:,} & {t['share'] * 100:.1f}\\% \\\\"
    for t in A2["top10"])
w("t2_toporgs.tex", r"""\begin{table}[t]
\caption{Top ten organizations by 30-day downloads}
\label{tab:toporgs}
\centering
\begin{tabular}{lrrr}
\toprule
Organization & Models & Downloads & Share \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % rows)

w("t3_hhi.tex", r"""\begin{table}[t]
\caption{Concentration of 30-day downloads across organizations}
\label{tab:hhi}
\centering
\begin{tabular}{lr}
\toprule
Measure & Value \\
\midrule
Organizations & %d \\
HHI (download shares) & %.0f \\
Largest org share (CR1) & %.1f\%% \\
Top-4 share (CR4) & %.1f\%% \\
Top-10 share (CR10) & %.1f\%% \\
\bottomrule
\end{tabular}
\end{table}""" % (
    A2["n_orgs"], A2["hhi_downloads"], A2["cr1"] * 100,
    A2["cr4"] * 100, A2["cr10"] * 100))

A5 = res["A5_ols_full"]
KEY_OLS = ["const", "org_type_corporate", "org_type_research",
           "license_class_restrictive", "license_class_unspecified",
           "tag_group_automatic-speech-recognition", "tag_group_image-text-to-text",
           "log_age"]
orows = "\n".join(
    f"{k.replace('_', chr(92) + '_')} & {v['coef']:.4f} & {v['se']:.4f} & {v['p']:.4f} \\\\"
    for k, v in A5["params"].items() if k in KEY_OLS)
w("t4_ols.tex", r"""\begin{table}[t]
\caption{OLS: $\log(\text{downloads})$ on model attributes (HC3 robust SE; key coefficients)}
\label{tab:ols}
\centering
\begin{tabular}{lrrr}
\toprule
Variable & $\hat\beta$ & SE & $p$-value \\
\midrule
%s
\bottomrule
\multicolumn{4}{l}{$n=%d$; $R^2=%.3f$; adj.\ $R^2=%.3f$. Task dummies n.s.\ omitted.} \\
\end{tabular}
\end{table}""" % (orows, A5["n"], A5["r2"], A5["adj_r2"]))

A5p = res["A5_ols_params"]
KEY_OLSP = ["const", "org_type_corporate", "org_type_research",
            "license_class_restrictive", "log_age", "log_params"]
prows = "\n".join(
    f"{k.replace('_', chr(92) + '_')} & {v['coef']:.4f} & {v['se']:.4f} & {v['p']:.4f} \\\\"
    for k, v in A5p["params"].items() if k in KEY_OLSP)
w("t5_ols_params.tex", r"""\begin{table}[t]
\caption{OLS with $\log(\text{parameters})$: detail subsample (HC3 robust SE; key coefficients)}
\label{tab:olsparams}
\centering
\begin{tabular}{lrrr}
\toprule
Variable & $\hat\beta$ & SE & $p$-value \\
\midrule
%s
\bottomrule
\multicolumn{4}{l}{$n=%d$ models with parameter counts; $R^2=%.3f$.} \\
\end{tabular}
\end{table}""" % (prows, A5p["n"], A5p["r2"]))

A6 = res["A6_logistic"]
KEY_LOGIT = ["const", "org_type_corporate", "org_type_research",
             "license_class_restrictive",
             "tag_group_automatic-speech-recognition", "tag_group_image-text-to-text",
             "log_age"]
lrows = "\n".join(
    f"{k.replace('_', chr(92) + '_')} & {v['or']:.3f} & "
    f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] & {v['p']:.4f} \\\\"
    for k, v in A6["odds_ratios"].items() if k in KEY_LOGIT)
w("t6_logistic.tex", r"""\begin{table}[t]
\caption{Logistic model of breakout adoption (odds ratios; key predictors)}
\label{tab:logit}
\centering
\begin{tabular}{lrrr}
\toprule
Variable & OR & 95\%% CI & $p$-value \\
\midrule
%s
\bottomrule
\multicolumn{4}{l}{$n=%d$; breakout $\geq$ %s downloads; pseudo-$R^2=%.3f$.} \\
\end{tabular}
\end{table}""" % (lrows, A6["n"], f"{A6['threshold']:,}", A6["pseudo_r2"]))

A7 = res["A7_cv"]
w("t7_cv.tex", r"""\begin{table}[t]
\caption{Breakout classifiers: repeated stratified 5-fold CV (10 repeats)}
\label{tab:cv}
\centering
\begin{tabular}{lrr}
\toprule
Model & AUC (mean) & SD \\
\midrule
Random forest (400 trees) & %.3f & %.3f \\
Logistic regression & %.3f & %.3f \\
Chance (AUC) & 0.500 & --- \\
Majority-class accuracy & %.3f & --- \\
\bottomrule
\end{tabular}
\end{table}""" % (
    A7["rf_auc_mean"], A7["rf_auc_sd"], A7["lr_auc_mean"], A7["lr_auc_sd"], A7["majority_acc"]))

A8 = res["A8_kmeans"]
k = A8["chosen_k"]
meds = A8["medians"]
names = {"0": "Aging mid-tier", "1": "Newcomer long tail", "2": "Breakout flagships", "3": "Factory long tail"}
arows = "\n".join(
    f"{names[i]} & {A8['sizes'][i]:,} & {meds[i]['log_downloads']:.2f} & "
    f"{meds[i]['age_years']:.2f} & "
    f"{meds[i]['log_org_models']:.2f} & {meds[i]['breakout']*100:.1f} \\\\"
    for i in sorted(meds))
sil = ", ".join(f"$k$={kk}: {vv:.2f}" for kk, vv in sorted(A8["silhouette"].items()))
w("t8_kmeans.tex", r"""\begin{table}[t]
\caption{K-means model archetypes (median feature values, $k=%d$)}
\label{tab:archetypes}
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Archetype & $n$ & $\log$ dl & Age (yr) & $\log$ org sz & Brk.\%% \\
\midrule
%s
\bottomrule
\end{tabular}

{\scriptsize Silhouette: %s.}
\end{table}""" % (k, arows, sil))

A9 = res["A9_temporal"]
A4 = res["A4_license_by_year"]
yrs = sorted(A4.keys())
lrow = " & ".join(str(y) for y in yrs)
def lrowget(cat):
    return " & ".join(f"{A4[y].get(cat, 0) * 100:.1f}" for y in yrs)
w("t9_license.tex", r"""\begin{table}[h]
\caption{License-class share (\%%) by model creation year}
\label{tab:license}
\centering
\small
\begin{tabular}{l%s}
\toprule
Class & %s \\
\midrule
Permissive & %s \\
Copyleft & %s \\
Restrictive & %s \\
Other & %s \\
Unspecified & %s \\
\bottomrule
\end{tabular}
\normalsize
\end{table}""" % ("r" * len(yrs), lrow, lrowget("permissive"), lrowget("copyleft"),
                  lrowget("restrictive"), lrowget("other"), lrowget("unspecified")))

print("wrote 9 table snippets to", TDIR)
