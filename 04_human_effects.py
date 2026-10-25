"""
04_human_effects.py
Human effects with the SAME method used for the language models
(weighted differences, original Moral Machine / Awad et al. 2018 estimator),
globally and by country, with cluster-robust 95% intervals (cluster = person).

Run (from the folder that contains mm_estimators.py):
    python 04_human_effects.py --parquet "outputs\\shared_responses_full.parquet" --out out_human_full
    python 04_human_effects.py --parquet "out_first\\SharedResponsesFullFirstSessions.parquet" --out out_human_first

Needs: pip install polars pyarrow numpy
Only READS the parquet file. It does one pass over the data (a few minutes).

Step 1 check: the global estimates (computed on the FULL file) should be close to the
published human values (Age 0.497, Fitness 0.160, Gender 0.119, ...). Small gaps are
possible because the published numbers come from the authors' own cleaning.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-clusters", type=int, default=300, help="smallest country (people) kept")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

PUBLISHED_HUMAN = {"Intervention": 0.061, "Relation to AV": 0.097, "Law": 0.353, "Gender": 0.119,
                   "Fitness": 0.160, "Social Status": 0.345, "Age": 0.497,
                   "No. Characters": 0.651, "Species": 0.585}
ORDER = list(PUBLISHED_HUMAN)

lf = pl.scan_parquet(args.parquet)
cols = lf.collect_schema().names()
for c in ["Intervention", "Barrier", "PedPed", "CrossingSignal", "ScenarioType", "ScenarioTypeStrict",
          "AttributeLevel", "Saved", "DiffNumberOFCharacters", "UserCountry3", "ExtendedSessionID"]:
    assert c in cols, f"missing column {c}"


def run(lazy):
    try:
        return lazy.collect(engine="streaming")
    except TypeError:
        return lazy.collect(streaming=True)


# ------------------------------------------------------------------ diagnostics
print("=== DIAGNOSTICS ===")
print("Rows:", f"{run(lf.select(pl.len())).item():,}")
for c in ["Intervention", "Barrier", "PedPed", "CrossingSignal", "ScenarioType"]:
    print(f"\n{c}:")
    print(run(lf.group_by(c).len().sort("len", descending=True)))

# ------------------------------------------------------------------ definitions
I, B, P, C = (pl.col(c) for c in ["Intervention", "Barrier", "PedPed", "CrossingSignal"])
ST = pl.col("ScenarioType").replace({"Social Status": "Social Value"})
STS = pl.col("ScenarioTypeStrict").replace({"Social Status": "Social Value"})
AL = pl.col("AttributeLevel")
DN = pl.col("DiffNumberOFCharacters")

mult_a = pl.when(C == 0).then(0.48).when(C == 1).then(0.2).otherwise(0.32)
mult_b = pl.when(C == 0).then(0.48).when(C == 1).then(0.32).otherwise(0.2)
p_design = (
    pl.when((I == 0) & (B == 0)).then(pl.when(P == 1).then(0.48).otherwise(0.32) * mult_a)
      .when((I == 0) & (B != 0)).then(0.2)
      .when((B == 0) & (P == 1)).then(0.48 * mult_b)
      .when(B == 0).then(0.2 * mult_a)
      .otherwise(0.32)
)
w = 1.0 / p_design
y = pl.col("Saved").cast(pl.Float64)

DIMS = [("Gender", "Gender", "Male", "Female"), ("Fitness", "Fitness", "Fat", "Fit"),
        ("Social Status", "Social Value", "Low", "High"), ("Age", "Age", "Old", "Young"),
        ("No. Characters", "Utilitarian", "Less", "More"), ("Species", "Species", "Pets", "Hoomans")]

specs = [("Intervention", pl.lit(True), I == 1),
         ("Relation to AV", (C == 0) & (P == 0), B == 0),
         ("Law", (C != 0) & (P == 1), C == 1)]
for name, stype, l1, l2 in DIMS:
    base = (ST == stype) & (STS == stype) & AL.is_in([l1, l2])
    if name == "No. Characters":
        for k in range(1, 5):
            specs.append((f"No. Characters (diff {k})", base & (DN == k), AL == l2))
    else:
        specs.append((name, base, AL == l2))

aggs = []
for i, (name, sub, g2) in enumerate(specs):
    m1 = (sub & ~g2).fill_null(False)
    m2 = (sub & g2).fill_null(False)
    aggs += [pl.when(m1).then(w).otherwise(0.0).sum().alias(f"s{i}_w1"),
             pl.when(m1).then(w * y).otherwise(0.0).sum().alias(f"s{i}_wy1"),
             pl.when(m2).then(w).otherwise(0.0).sum().alias(f"s{i}_w2"),
             pl.when(m2).then(w * y).otherwise(0.0).sum().alias(f"s{i}_wy2"),
             (m1 | m2).sum().cast(pl.Int64).alias(f"s{i}_n")]

person = (pl.coalesce(pl.col("UserID").cast(pl.Utf8), pl.col("ExtendedSessionID").cast(pl.Utf8))
          if "UserID" in cols else pl.col("ExtendedSessionID").cast(pl.Utf8))
print("\nAggregating per person and country (one pass, a few minutes)...")
T = run(lf.group_by(pl.col("UserCountry3").alias("country"), person.alias("cluster")).agg(aggs))
T = T.with_columns(pl.lit("ALL").alias("grp"))
print(f"Done: {T.height:,} person-country groups.")


# ------------------------------------------------------------------ estimation
def estimate(T, i, key):
    """Estimate for spec i by `key`, with cluster-robust (delta-method) standard errors."""
    a = T.select(key, pl.col(f"s{i}_w1").alias("w1"), pl.col(f"s{i}_wy1").alias("wy1"),
                 pl.col(f"s{i}_w2").alias("w2"), pl.col(f"s{i}_wy2").alias("wy2"),
                 pl.col(f"s{i}_n").alias("n"))
    S = a.group_by(key).agg(W1=pl.col("w1").sum(), Y1=pl.col("wy1").sum(), W2=pl.col("w2").sum(),
                            Y2=pl.col("wy2").sum(), n_rows=pl.col("n").sum(),
                            G=((pl.col("w1") + pl.col("w2")) > 0).sum())
    S = S.with_columns(m1=pl.col("Y1") / pl.col("W1"), m2=pl.col("Y2") / pl.col("W2"))
    a = a.join(S.select(key, "W1", "W2", "m1", "m2"), on=key)
    a = a.with_columns(infl=((pl.col("wy2") - pl.col("m2") * pl.col("w2")) / pl.col("W2"))
                            - ((pl.col("wy1") - pl.col("m1") * pl.col("w1")) / pl.col("W1")))
    V = a.group_by(key).agg(v=(pl.col("infl") ** 2).sum())
    res = (S.join(V, on=key)
             .with_columns(estimate=pl.col("m2") - pl.col("m1"),
                           se=(pl.col("v") * pl.col("G") / (pl.col("G") - 1)).sqrt())
             .select(pl.col(key).alias("group"), "estimate", "se", "n_rows", pl.col("G").alias("n_people")))
    return res.with_columns(pl.lit(specs[i][0]).alias("effect_name"))


def table(key):
    parts = [estimate(T, i, key) for i in range(len(specs))]
    return pl.concat(parts).with_columns(
        (pl.col("estimate") - 1.96 * pl.col("se")).alias("ci_low"),
        (pl.col("estimate") + 1.96 * pl.col("se")).alias("ci_high"))


glob = table("grp")
bycty = table("country")

# "No. Characters" as published: the difference (1-4) with the largest absolute effect (global)
nc = glob.filter(pl.col("effect_name").str.starts_with("No. Characters ("))
best = nc.sort(pl.col("estimate").abs(), descending=True)["effect_name"][0]
print(f"\nNo. Characters: using '{best}' (largest effect, as in the published figure)")


def with_nc(df):
    nc_rows = df.filter(pl.col("effect_name") == best).with_columns(pl.lit("No. Characters").alias("effect_name"))
    return pl.concat([df, nc_rows])


glob = with_nc(glob)
bycty = with_nc(bycty)
glob.write_csv(out / "human_effects_global.csv")
bycty = bycty.filter(pl.col("n_people") >= args.min_clusters)
bycty.write_csv(out / "human_effects_by_country.csv")

# ------------------------------------------------------------------ report
g = glob.filter(pl.col("effect_name").is_in(ORDER))
rows = []
for name in ORDER:
    r = g.filter(pl.col("effect_name") == name).row(0, named=True)
    rows.append({"effect": name, "ours": round(r["estimate"], 3), "ci_low": round(r["ci_low"], 3),
                 "ci_high": round(r["ci_high"], 3), "published": PUBLISHED_HUMAN[name],
                 "gap": round(r["estimate"] - PUBLISHED_HUMAN[name], 3), "n_people": r["n_people"]})
print("\n=== GLOBAL HUMAN EFFECTS vs PUBLISHED ===")
print(pl.DataFrame(rows))
print(f"\nCountries kept (>= {args.min_clusters} people per effect): "
      f"{bycty['group'].n_unique()}. Saved files in {out.resolve()}")
print("DZA:")
print(bycty.filter((pl.col("group") == "DZA") & pl.col("effect_name").is_in(ORDER))
          .select("effect_name", "estimate", "ci_low", "ci_high", "n_people")
          .with_columns(pl.col(["estimate", "ci_low", "ci_high"]).round(3)))
