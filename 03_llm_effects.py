"""
03_llm_effects.py
Effects of the 4 language models (GPT-3.5, GPT-4, PaLM 2, Llama 2) with bootstrap
95% intervals, compared with the published human values.

Needs mm_estimators.py in the same folder.
Run (from the folder that contains mmllm_data):
    python 03_llm_effects.py --data "mmllm_data\\data" --out out_llm
Needs: pip install pandas numpy matplotlib

What it checks first: our numbers must equal the published ones in
summary_overall_preferences.csv. If they do not, do not trust the rest.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mm_estimators import effects, ORDER

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"mmllm_data\data", help="folder with the model CSV files")
ap.add_argument("--out", default="out_llm")
ap.add_argument("--boot", type=int, default=500, help="bootstrap repetitions")
args = ap.parse_args()

data = Path(args.data)
out = Path(args.out)
out.mkdir(exist_ok=True)

PATTERNS = {"GPT-3.5": "shared_responses_gpt-3*.csv", "GPT-4": "shared_responses_gpt-4*.csv",
            "PaLM 2": "shared_responses_palm2*.csv", "Llama 2": "shared_responses_llama*.csv"}
published = pd.read_csv(data / "summary_overall_preferences.csv", encoding="utf-8-sig").set_index("Label")
human = published["Human"]

rows = []
for model, pattern in PATTERNS.items():
    path = next(data.glob(pattern))
    df = pd.read_csv(path)
    # a scenario = the two rows sharing the same ResponseID prefix (..._1 and ..._2)
    df["scenario"] = df["ResponseID"].str.rsplit("_", n=1).str[0]
    print(f"{model}: {len(df):,} rows, {df['scenario'].nunique():,} scenarios  ({path.name})")
    res = effects(df, cluster="scenario", n_boot=args.boot).set_index("effect_name")

    # "No. Characters" as published = the difference (1..4) with the largest absolute effect
    diffs = [f"No. Characters (diff {k})" for k in range(1, 5)]
    best = res.loc[diffs, "estimate"].abs().idxmax()
    print(f"   No. Characters uses the difference with the largest effect: {best}")
    res.loc["No. Characters"] = res.loc[best]
    res["model"] = model
    rows.append(res.reset_index())

allres = pd.concat(rows, ignore_index=True)
allres.to_csv(out / "llm_effects_all.csv", index=False)
main = allres[allres["effect_name"].isin(ORDER)].copy()

# ---- 1. CHECK: do we reproduce the published values? ----
chk = main.pivot(index="effect_name", columns="model", values="estimate").loc[ORDER]
pub = published.loc[ORDER, list(PATTERNS)]
maxdiff = (chk - pub).abs().max().max()
print(f"\nCHECK vs published values: largest absolute difference = {maxdiff:.6f}")
print("  (should be ~0.00000x; if it is bigger, something is wrong)")

# ---- 2. Table: estimates with intervals ----
main["text"] = main.apply(lambda r: f"{r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
table = main.pivot(index="effect_name", columns="model", values="text").loc[ORDER, list(PATTERNS)]
table["Human (published)"] = human.loc[ORDER].map(lambda v: f"{v:+.3f}")
print("\nEffects with bootstrap 95% intervals (resampling scenarios):")
print(table.to_string())
table.to_csv(out / "llm_effects_table.csv")

# ---- 3. Closeness to humans ----
est = main.pivot(index="effect_name", columns="model", values="estimate").loc[ORDER, list(PATTERNS)]
summ = pd.DataFrame({
    "mean_abs_diff_vs_human": (est.sub(human.loc[ORDER], axis=0)).abs().mean().round(3),
    "same_sign_as_human_(of_9)": (np.sign(est).eq(np.sign(human.loc[ORDER]), axis=0)).sum(),
}).sort_values("mean_abs_diff_vs_human")
print("\nCloseness to the published human values (smaller = closer):")
print(summ.to_string())
summ.to_csv(out / "llm_closeness_to_human.csv")

# ---- 4. Chart ----
fig, ax = plt.subplots(figsize=(11, 7))
colors = {"GPT-3.5": "#1f77b4", "GPT-4": "#2ca02c", "PaLM 2": "#ff7f0e", "Llama 2": "#9467bd"}
ypos = np.arange(len(ORDER))[::-1]
for i, (model, c) in enumerate(colors.items()):
    m = main[main.model == model].set_index("effect_name").loc[ORDER]
    off = (i - 1.5) * 0.15
    ax.errorbar(m.estimate, ypos + off, xerr=[m.estimate - m.ci_low, m.ci_high - m.estimate],
                fmt="o", ms=5, color=c, label=model, capsize=2, lw=1)
ax.scatter(human.loc[ORDER], ypos, marker="|", s=500, color="red", label="Human (published)", zorder=5)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(ORDER)
ax.set_xlabel("Effect on probability of being saved (positive: inaction, pedestrians, lawful, female, fit,\n"
              "high status, young, more characters, humans)")
ax.set_title("Language models vs published human preferences (bootstrap 95% intervals)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(out / "llm_vs_human.png", dpi=150)
print(f"\nSaved to {out.resolve()}: llm_effects_table.csv, llm_effects_all.csv, "
      "llm_closeness_to_human.csv, llm_vs_human.png")
