"""
05_models_vs_countries.py
Which countries' human preferences is each language model closest to?

Inputs (small CSV files written by the earlier scripts):
    out_human_first\\human_effects_by_country.csv   (from 04_human_effects.py)
    out_human_first\\human_effects_global.csv
    out_llm\\llm_effects_all.csv                    (from 03_llm_effects.py)

Run (from D:\\New folder (2)):
    python 05_models_vs_countries.py --human out_human_first --llm out_llm --out out_compare

Distance = Euclidean distance between the vector of 9 effects of a model and that of a
country (same measure family as Takemoto 2024). Smaller = closer. It is corrected for
noise: small countries have noisier estimates, which would inflate their raw distance.
Uncertainty: both sides are redrawn 2000 times from their estimates and standard errors.
The raw (uncorrected) distance is also saved in the CSV.
IMPORTANT: for 'No. Characters' every model and country uses the 4-extra-people effect
(the human data only supports that one for small countries), so PaLM 2 here differs
from its published figure (which uses the 1-person effect).
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
ap.add_argument("--out", default="out_compare")
ap.add_argument("--draws", type=int, default=2000)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

EFFECTS = ["Intervention", "Relation to AV", "Law", "Gender", "Fitness", "Social Status",
           "Age", "No. Characters (diff 4)", "Species"]
MODELS = ["GPT-3.5", "GPT-4", "PaLM 2", "Llama 2"]
ARAB = ["DZA", "MAR", "TUN", "LBY", "EGY", "SAU", "ARE", "QAT", "KWT", "BHR", "OMN", "YEM",
        "IRQ", "SYR", "JOR", "LBN", "PSE"]
rng = np.random.default_rng(7)

# ---------------- read ----------------
hc = pd.read_csv(Path(args.human) / "human_effects_by_country.csv")
hg = pd.read_csv(Path(args.human) / "human_effects_global.csv")
llm = pd.read_csv(Path(args.llm) / "llm_effects_all.csv")

H = hc[hc.effect_name.isin(EFFECTS)].pivot(index="group", columns="effect_name", values="estimate")
HS = hc[hc.effect_name.isin(EFFECTS)].pivot(index="group", columns="effect_name", values="se")
NP = hc[hc.effect_name.isin(EFFECTS)].pivot(index="group", columns="effect_name", values="n_people")
complete = H.dropna().index.intersection(HS.dropna().index)
print(f"Countries with all {len(EFFECTS)} effects estimated: {len(complete)} "
      f"(of {H.shape[0]} with at least one)")
H, HS, NP = H.loc[complete, EFFECTS], HS.loc[complete, EFFECTS], NP.loc[complete, EFFECTS]

world = hg[hg.effect_name.isin(EFFECTS)].set_index("effect_name").loc[EFFECTS]
M = llm[llm.effect_name.isin(EFFECTS)].pivot(index="model", columns="effect_name", values="estimate").loc[MODELS, EFFECTS]
MS = llm[llm.effect_name.isin(EFFECTS)].pivot(index="model", columns="effect_name", values="boot_se").loc[MODELS, EFFECTS]


def dist_with_ci(h, hse, m, mse):
    """
    Noise-corrected Euclidean distance. The raw distance of two noisy vectors is inflated
    by the sum of their variances (more for small countries), so we subtract it from the
    squared distance. Returns (corrected distance, ci_low, ci_high, raw distance).
    """
    var = (hse ** 2 + mse ** 2).sum()
    d2_obs = ((h - m) ** 2).sum()
    point = np.sqrt(max(d2_obs - var, 0.0))
    dh = rng.normal(h, hse, size=(args.draws, len(h)))
    dm = rng.normal(m, mse, size=(args.draws, len(m)))
    d2_draw = ((dh - dm) ** 2).sum(axis=1) - 2 * var       # centred on the corrected value
    lo, hi = np.sqrt(np.clip(np.percentile(d2_draw, [2.5, 97.5]), 0, None))
    return point, min(lo, point), max(hi, point), np.sqrt(d2_obs)


rows = []
for c in H.index:
    for mod in MODELS:
        d, lo, hi, raw = dist_with_ci(H.loc[c].values, HS.loc[c].values, M.loc[mod].values, MS.loc[mod].values)
        rows.append((c, mod, d, lo, hi, raw, int(NP.loc[c].min())))
D = pd.DataFrame(rows, columns=["country", "model", "distance", "ci_low", "ci_high", "raw_distance", "min_people"])
D["rank_closest_first"] = D.groupby("model")["distance"].rank()

# reference: the world average
ref = []
for mod in MODELS:
    d, lo, hi, raw = dist_with_ci(world["estimate"].values, world["se"].values, M.loc[mod].values, MS.loc[mod].values)
    ref.append((mod, d, lo, hi, raw))
REF = pd.DataFrame(ref, columns=["model", "distance", "ci_low", "ci_high", "raw_distance"]).set_index("model")
D.to_csv(out / "distance_country_model.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== Distance between each model and the WORLD average (smaller = closer) ===")
print(REF.round(3).to_string())

for mod in MODELS:
    s = D[D.model == mod].sort_values("distance")
    print(f"\n=== {mod}: 8 closest and 8 farthest countries (distance, 95% interval) ===")
    fmt = lambda r: f"{r.country}  {r.distance:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    print("closest :", " | ".join(fmt(r) for r in s.head(8).itertuples()))
    print("farthest:", " | ".join(fmt(r) for r in s.tail(8).itertuples()))

print("\n=== ALGERIA and Arab countries: rank among", len(H), "countries (1 = closest to the model) ===")
ar = D[D.country.isin(ARAB)].pivot(index="country", columns="model", values="rank_closest_first")[MODELS].astype(int)
print(ar.sort_values("GPT-4").to_string())

best = D.loc[D.groupby("country")["distance"].idxmin()]
print("\nClosest model per country (number of countries):")
print(best.model.value_counts().to_string())
print("\nCountries closest to each model by distance rank correlation across models:")
print(D.pivot(index="country", columns="model", values="distance")[MODELS].corr(method="spearman").round(2).to_string())

# ---------------- chart ----------------
P = D.pivot(index="country", columns="model", values="distance")[MODELS]
P = P.loc[P.mean(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(7, max(6, 0.17 * len(P))))
im = ax.imshow(P.values, aspect="auto", cmap="viridis_r")
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS)
ax.set_yticks(range(len(P)))
ax.set_yticklabels(P.index, fontsize=6)
ax.xaxis.tick_top()
for i, c in enumerate(P.index):
    if c == "DZA":
        ax.get_yticklabels()[i].set_color("red")
        ax.get_yticklabels()[i].set_fontweight("bold")
fig.colorbar(im, ax=ax, label="Distance to the model (smaller = closer)", shrink=0.5)
ax.set_title("Human preferences by country vs language models\n(countries sorted by average distance; Algeria in red)", fontsize=9, pad=28)
fig.tight_layout()
fig.savefig(out / "countries_vs_models.png", dpi=170)
print("\nSaved in", out.resolve(), ": distance_country_model.csv, countries_vs_models.png")
