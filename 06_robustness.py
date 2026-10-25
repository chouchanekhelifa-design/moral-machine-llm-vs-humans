"""
06_robustness.py
Is the "which countries are closest to which model" result robust?

Checks, for each model, whether the ranking of countries changes when we:
  (1) compare only the PATTERN of the 9 effects (Pearson correlation), ignoring size;
  (2) standardise each effect by how much countries differ on it (so no effect dominates);
  (3) drop each of the 9 effects in turn (leave-one-out);
and shows WHICH effects drive the distance for Algeria and a few other countries.

Inputs: same folders as 05_models_vs_countries.py
Run (from D:\\New folder (2)):
    python 06_robustness.py --human out_human_first --llm out_llm --out out_robust
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--human", default="out_human_first")
ap.add_argument("--llm", default="out_llm")
ap.add_argument("--out", default="out_robust")
ap.add_argument("--min-people", type=int, default=300, help="smallest country kept (people, every effect)")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

EFFECTS = ["Intervention", "Relation to AV", "Law", "Gender", "Fitness", "Social Status",
           "Age", "No. Characters (diff 4)", "Species"]
SHORT = ["Interv", "RelAV", "Law", "Gender", "Fitness", "Status", "Age", "NoChar", "Species"]
MODELS = ["GPT-3.5", "GPT-4", "PaLM 2", "Llama 2"]
FOCAL = ["DZA", "JOR", "SAU", "NOR", "DEU", "BRA", "JPN"]

hc = pd.read_csv(Path(args.human) / "human_effects_by_country.csv")
hg = pd.read_csv(Path(args.human) / "human_effects_global.csv")
llm = pd.read_csv(Path(args.llm) / "llm_effects_all.csv")

sel = hc[hc.effect_name.isin(EFFECTS)]
H = sel.pivot(index="group", columns="effect_name", values="estimate")
HS = sel.pivot(index="group", columns="effect_name", values="se")
NP = sel.pivot(index="group", columns="effect_name", values="n_people")
ok = H.dropna().index.intersection(HS.dropna().index)
ok = [c for c in ok if NP.loc[c, EFFECTS].min() >= args.min_people]
H, HS = H.loc[ok, EFFECTS], HS.loc[ok, EFFECTS]
print(f"{len(ok)} countries with all 9 effects and >= {args.min_people} people per effect")

W = hg[hg.effect_name.isin(EFFECTS)].set_index("effect_name").loc[EFFECTS]
M = llm[llm.effect_name.isin(EFFECTS)].pivot(index="model", columns="effect_name", values="estimate").loc[MODELS, EFFECTS]
MS = llm[llm.effect_name.isin(EFFECTS)].pivot(index="model", columns="effect_name", values="boot_se").loc[MODELS, EFFECTS]

sd_across = H.std(ddof=1).values            # how much countries differ on each effect


def d2_terms(h, hse, m, mse):
    """Per-effect noise-corrected squared differences."""
    return (h - m) ** 2 - hse ** 2 - mse ** 2


rows = []
for c in H.index:
    h, hse = H.loc[c].values, HS.loc[c].values
    for mod in MODELS:
        m, mse = M.loc[mod].values, MS.loc[mod].values
        t = d2_terms(h, hse, m, mse)
        r = {"country": c, "model": mod,
             "dist": np.sqrt(max(t.sum(), 0)),
             "dist_std": np.sqrt(max((t / sd_across ** 2).sum(), 0)),
             "pattern_r": np.corrcoef(h, m)[0, 1]}
        for k, name in enumerate(SHORT):
            r[f"dist_without_{name}"] = np.sqrt(max(np.delete(t, k).sum(), 0))
        rows.append(r)
D = pd.DataFrame(rows)

# ranks: 1 = closest
for col in [c for c in D.columns if c.startswith("dist")]:
    D["rank_" + col] = D.groupby("model")[col].rank()
D["rank_pattern_r"] = D.groupby("model")["pattern_r"].rank(ascending=False)
D.to_csv(out / "robustness_rankings.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== 1. HOW STABLE IS THE RANKING? (Spearman correlation with the main distance ranking; 1 = identical) ===")
stab = []
for mod in MODELS:
    s = D[D.model == mod]
    base = s["rank_dist"]
    loo = {n: base.corr(s[f"rank_dist_without_{n}"], method="spearman") for n in SHORT}
    worst = min(loo, key=loo.get)
    top = lambda col: set(s.nsmallest(10, col)["country"])
    overlap_r = len(top("rank_dist") & set(s.nsmallest(10, "rank_pattern_r")["country"]))
    overlap_z = len(top("rank_dist") & top("rank_dist_std"))
    stab.append({"model": mod,
                 "pattern_only": round(base.corr(s["rank_pattern_r"], method="spearman"), 2),
                 "standardised": round(base.corr(s["rank_dist_std"], method="spearman"), 2),
                 "worst_drop_one": round(loo[worst], 2), "worst_effect_dropped": worst,
                 "top10_overlap_pattern": overlap_r, "top10_overlap_standardised": overlap_z})
print(pd.DataFrame(stab).set_index("model").to_string())
print("(top-10 overlap: how many of the 10 closest countries stay in the top 10)")

print("\n=== 2. CLOSEST COUNTRIES BY PATTERN ONLY (correlation of the 9 effects) ===")
for mod in MODELS:
    s = D[D.model == mod].sort_values("pattern_r", ascending=False)
    print(f"{mod:8s}: " + " | ".join(f"{r.country} {r.pattern_r:.2f}" for r in s.head(8).itertuples()))
    print(f"{'':8s}  farthest: " + " | ".join(f"{r.country} {r.pattern_r:.2f}" for r in s.tail(5).itertuples()))
wr = {mod: np.corrcoef(W["estimate"].values, M.loc[mod].values)[0, 1] for mod in MODELS}
print("Pattern correlation of the world average with each model:", {k: round(v, 2) for k, v in wr.items()})

print("\n=== 3. WHERE DOES ALGERIA RANK UNDER EACH VARIANT? (of", len(ok), "countries; 1 = closest) ===")
if "DZA" in set(D.country):
    a = D[D.country == "DZA"].set_index("model")
    print(a[["rank_dist", "rank_dist_std", "rank_pattern_r"]].astype(int)
          .rename(columns={"rank_dist": "main", "rank_dist_std": "standardised", "rank_pattern_r": "pattern"})
          .to_string())
    print("pattern correlation of Algeria with each model:", a["pattern_r"].round(2).to_dict())
else:
    print("Algeria is not in the kept set.")

print("\n=== 4. WHICH EFFECTS DRIVE THE DISTANCE? (share of the squared difference, %) ===")
share_rows = []
focal = [c for c in FOCAL if c in set(H.index)]
for mod in MODELS:
    for c in ["WORLD"] + focal:
        h = W["estimate"].values if c == "WORLD" else H.loc[c].values
        sq = (h - M.loc[mod].values) ** 2
        share_rows.append([mod, c] + list(np.round(100 * sq / sq.sum()).astype(int)))
S = pd.DataFrame(share_rows, columns=["model", "country"] + SHORT)
print(S.to_string(index=False))
S.to_csv(out / "distance_drivers_percent.csv", index=False)

# chart: pattern vs distance
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, mod in zip(axes, MODELS):
    s = D[D.model == mod]
    ax.scatter(s.dist, s.pattern_r, s=14, color="gray")
    for c, col in [("DZA", "red"), ("JOR", "green"), ("NOR", "blue")]:
        if c in set(s.country):
            r = s[s.country == c].iloc[0]
            ax.scatter(r.dist, r.pattern_r, s=40, color=col)
            ax.annotate(c, (r.dist, r.pattern_r), fontsize=8, color=col, xytext=(3, 3), textcoords="offset points")
    ax.set_title(mod)
    ax.set_xlabel("distance (smaller = closer)")
axes[0].set_ylabel("pattern correlation (larger = more similar)")
fig.suptitle("Two ways to be 'close' to a model: small distance and similar pattern", fontsize=10)
fig.tight_layout()
fig.savefig(out / "distance_vs_pattern.png", dpi=150)
print("\nSaved in", out.resolve(), ": robustness_rankings.csv, distance_drivers_percent.csv, distance_vs_pattern.png")
