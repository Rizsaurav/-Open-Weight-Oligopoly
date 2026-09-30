"""05_figures.py — Publication figures for the HF oligopoly paper.

Reads data/models_clean.csv and results/hf_results.json.
Writes figures/fig1_lorenz.png ... fig5_licenses.png (300 dpi).

Style: clean IEEE two-column friendly; single palette throughout.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

C = {"corp": "#7a1f1f", "res": "#2f5d62", "com": "#b98a2f", "gray": "#8a8a8a",
     "dark": "#222222", "light": "#d9d9d9"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(DATA, "models_clean.csv"))
res = json.load(open(os.path.join(HERE, "results", "hf_results.json")))
assert len(df) == res["n"], "figure/data row mismatch"
df["breakout"] = (df.downloads >= res["A6_logistic"]["threshold"]).astype(int)

# ---------- Fig 1: Lorenz curve of downloads ----------
d = np.sort(df.downloads.to_numpy(float))
cum = np.cumsum(d) / d.sum()
x = np.arange(1, len(d) + 1) / len(d)
fig, ax = plt.subplots(figsize=(3.3, 2.8))
ax.plot(x, cum, color=C["corp"], lw=1.6, label="Downloads")
ax.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1, label="Equality")
ax.set_xlabel("Cumulative share of models", fontsize=8)
ax.set_ylabel("Cumulative share of 30-day downloads", fontsize=8)
ax.set_title(f"Lorenz curve (Gini = {res['A1_descriptives']['gini']:.3f})", fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(pad=0.6); fig.savefig(os.path.join(FIG, "fig1_lorenz.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: top-10 orgs by downloads ----------
top10 = res["A2_org_concentration"]["top10"]
orgs = [t["org"] for t in top10][::-1]
shares = [t["share"] * 100 for t in top10][::-1]
fig, ax = plt.subplots(figsize=(3.3, 3.2))
ax.barh(orgs, shares, color=C["corp"], edgecolor="none")
ax.set_xlabel("Share of sample 30-day downloads (%)", fontsize=8)
ax.set_title("Top 10 organizations by downloads", fontsize=9)
ax.tick_params(labelsize=7)
for i, v in enumerate(shares):
    ax.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=7)
ax.set_xlim(0, max(shares) * 1.28)
fig.tight_layout(pad=0.6); fig.savefig(os.path.join(FIG, "fig2_orgs.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: RF importance + breakout by org_type (single-column, stacked) ----------
imp = res["A7_cv"]["rf_importance_top8"]
names = [k.replace("tag_group_", "").replace("_", " ") for k in imp.keys()][::-1]
vals = [imp[k] for k in imp.keys()][::-1]
br = df.groupby("org_type").breakout.mean().reindex(["corporate", "research", "community"])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 3.2))
a1.barh(names, vals, color=C["res"], edgecolor="none")
a1.set_xlabel("Gini importance", fontsize=8)
a1.set_title(f"Top RF features (AUC={res['A7_cv']['rf_auc_mean']:.2f})", fontsize=9)
a1.tick_params(labelsize=7)
for i, v in enumerate(vals):
    a1.text(v + max(vals) * 0.02, i, f"{v:.2f}", va="center", fontsize=7)
a1.set_xlim(0, max(vals) * 1.22)
a2.bar(br.index, br.values * 100, color=[C["corp"], C["res"], C["com"]], edgecolor="none")
a2.set_ylabel("Breakout rate (%)", fontsize=8)
a2.set_title("Breakout rate by organization type", fontsize=9)
a2.tick_params(labelsize=7)
a2.set_ylim(0, br.values.max() * 100 * 1.3)
for i, v in enumerate(br.values * 100):
    a2.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=7)
fig.tight_layout(pad=0.8, h_pad=1.2); fig.savefig(os.path.join(FIG, "fig3_rf.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: centralization + params over time (single-column, stacked) ----------
share_y = res["A9_temporal"]["top10_org_share_by_year"]
med_p = res["A3_params"]["median_params_by_year"]
ys = sorted(share_y.keys())
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.3, 3.2))
a1.plot(ys, [share_y[y] * 100 for y in ys], marker="o", color=C["corp"], lw=1.8)
a1.set_xlabel("Model creation year", fontsize=8); a1.set_ylabel("Top-10 org download share (%)", fontsize=8)
a1.set_title("Download concentration by cohort year", fontsize=9)
a1.tick_params(labelsize=7)
a1.set_ylim(0, 100)
py = sorted(med_p.keys())
a2.plot(py, [med_p[y] / 1e9 for y in py], marker="s", color=C["res"], lw=1.8)
a2.set_xlabel("Model creation year", fontsize=8); a2.set_ylabel("Median parameters (B)", fontsize=8)
a2.set_title("Median model size by cohort year", fontsize=9)
a2.tick_params(labelsize=7)
a2.set_yscale("log")
fig.tight_layout(pad=0.8, h_pad=1.2); fig.savefig(os.path.join(FIG, "fig4_time.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 5: license class shares by year ----------
lic = res["A4_license_by_year"]
years = sorted(lic.keys())
cats = ["permissive", "copyleft", "restrictive", "other", "unspecified"]
cols = {"permissive": C["res"], "copyleft": C["com"], "restrictive": C["corp"],
        "other": C["gray"], "unspecified": C["light"]}
fig, ax = plt.subplots(figsize=(3.3, 2.8))
bottom = np.zeros(len(years))
for c in cats:
    v = np.array([lic[y].get(c, 0) for y in years]) * 100
    ax.bar(years, v, bottom=bottom, label=c, color=cols[c], edgecolor="white", lw=0.5)
    bottom += v
ax.set_xlabel("Model creation year", fontsize=8); ax.set_ylabel("Share of models (%)", fontsize=8)
ax.set_title("License class by cohort year", fontsize=9)
ax.tick_params(labelsize=7)
ax.legend(frameon=False, ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.tight_layout(pad=0.6); fig.savefig(os.path.join(FIG, "fig5_licenses.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

print("figures written to", FIG)
for f in ["fig1_lorenz.png", "fig2_orgs.png", "fig3_rf.png", "fig4_time.png", "fig5_licenses.png"]:
    p = os.path.join(FIG, f)
    assert os.path.exists(p) and os.path.getsize(p) > 5000, f"figure missing/small: {f}"
print("all 5 figures OK")
