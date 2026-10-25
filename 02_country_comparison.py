"""
02_country_comparison.py
Compares every country with the REST OF THE WORLD, using the results table
written by 01_explore_human_data.py. It reads only a small CSV (runs in seconds).

Run (from the folder that contains out_first):
    python 02_country_comparison.py --folder out_first

IMPORTANT
  * Use it on the FullFirstSessions results (out_first): there every person has
    exactly one session, so "sessions" = independent people and the intervals
    are valid. Do not use it on the full-file results (people have many sessions).
  * "Significant" = still different after correcting for the number of
    comparisons (Benjamini-Hochberg false discovery rate, 5%). It says the
    difference is unlikely to be chance. It does NOT say why it exists.
  * Players are volunteers on a website, not a representative sample of a country.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="out_first", help="folder with crude_effects_by_country.csv")
ap.add_argument("--min-sessions", type=int, default=300,
                help="countries with fewer people (per dimension) are skipped")
ap.add_argument("--fdr", type=float, default=0.05)
args = ap.parse_args()

folder = Path(args.folder)
res = pl.read_csv(folder / "crude_effects_by_country.csv")

world = res.filter(pl.col("country") == "ALL").select(
    "dimension",
    pl.col("n_rows").alias("N_w"), pl.col("effect").alias("eff_w"),
    pl.col("ci_low").alias("lo_w"), pl.col("ci_high").alias("hi_w"))
cty = res.filter(pl.col("country") != "ALL")

df = cty.join(world, on="dimension")

# effect = 2P - 1. Work on the "sum of successes" scale S = P * N, where the
# variances of different countries simply add up (independent people).
se_c = (pl.col("ci_high") - pl.col("ci_low")) / (2 * 1.96)          # SE of effect, country
se_w = (pl.col("hi_w") - pl.col("lo_w")) / (2 * 1.96)               # SE of effect, world
P_c = (pl.col("effect") + 1) / 2
P_w = (pl.col("eff_w") + 1) / 2
df = df.with_columns(
    S_c=P_c * pl.col("n_rows"), S_w=P_w * pl.col("N_w"),
    varS_c=(se_c / 2 * pl.col("n_rows")) ** 2, varS_w=(se_w / 2 * pl.col("N_w")) ** 2,
).with_columns(
    N_r=pl.col("N_w") - pl.col("n_rows"),
    S_r=pl.col("S_w") - pl.col("S_c"),
    varS_r=(pl.col("varS_w") - pl.col("varS_c")).clip(lower_bound=1e-9),
).with_columns(
    P_r=pl.col("S_r") / pl.col("N_r"),
    se_P_r=pl.col("varS_r").sqrt() / pl.col("N_r"),
).with_columns(
    effect_rest=2 * pl.col("P_r") - 1,
    se_rest=2 * pl.col("se_P_r"),
    se_country=se_c,
).with_columns(
    diff=pl.col("effect") - pl.col("effect_rest"),
    se_diff=(pl.col("se_country") ** 2 + pl.col("se_rest") ** 2).sqrt(),
).with_columns(
    diff_low=pl.col("diff") - 1.96 * pl.col("se_diff"),
    diff_high=pl.col("diff") + 1.96 * pl.col("se_diff"),
    z=pl.col("diff") / pl.col("se_diff"),
)

df = df.filter(pl.col("n_sessions") >= args.min_sessions)
z = df["z"].to_numpy()
p = np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])

# Benjamini-Hochberg false discovery rate
m = len(p)
order = np.argsort(p)
q_sorted = p[order] * m / (np.arange(m) + 1)
q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
q = np.empty(m)
q[order] = np.minimum(q_sorted, 1.0)

out = (df.with_columns(pl.Series("p_value", p), pl.Series("q_value", q))
         .with_columns((pl.col("q_value") < args.fdr).alias("significant"))
         .select("country", "dimension", "n_sessions", "effect", "effect_rest",
                 "diff", "diff_low", "diff_high", "p_value", "q_value", "significant")
         .sort("country", "dimension"))
out.write_csv(folder / "country_vs_world.csv")

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
show = ["country", "dimension", "n_sessions", "effect", "effect_rest", "diff",
        "diff_low", "diff_high", "q_value", "significant"]
rnd = lambda d: d.with_columns(pl.col(["effect", "effect_rest", "diff", "diff_low", "diff_high", "q_value"]).round(3))

print(f"Tests run: {m} (countries with >= {args.min_sessions} people, 6 dimensions)")
print(f"Significant after correction (FDR {args.fdr:.0%}): {int(out['significant'].sum())}")

print("\n=== ALGERIA vs rest of the world ===")
print(rnd(out.filter(pl.col("country") == "DZA")).select(show))

arab = ["DZA", "MAR", "TUN", "LBY", "EGY", "SAU", "ARE", "QAT", "KWT", "BHR", "OMN",
        "YEM", "IRQ", "SYR", "JOR", "LBN", "PSE"]
print("\n=== ARAB COUNTRIES: difference from the rest of the world (q < 0.05 marked True) ===")
print(rnd(out.filter(pl.col("country").is_in(arab))).select(show).sort("dimension", "diff"))

print("\n=== 25 LARGEST SIGNIFICANT DIFFERENCES (any country) ===")
big = out.filter(pl.col("significant")).with_columns(pl.col("diff").abs().alias("abs_diff")) \
         .sort("abs_diff", descending=True).head(25)
print(rnd(big).select(show))

print("\nSaved:", (folder / "country_vs_world.csv").resolve())
