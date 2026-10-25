"""
01_explore_human_data.py
First look at the real Moral Machine human data (one row per scenario).

What it does
  1. Converts the big CSV to Parquet once (fast to reuse afterwards).
  2. Prints the structure: columns, rows, sessions, missing values.
  3. Counts responses and sessions per country (checks Algeria / Arab countries).
  4. Computes CRUDE preference effects (global and per country) with 95% CIs
     clustered by session (ExtendedSessionID).

Run:  python 01_explore_human_data.py --csv "SharedResponses.csv" --out out_all
      (use a different --out folder for each CSV file)
Needs: pip install polars pyarrow numpy

IMPORTANT
  * Crude effect = 2*P - 1, where P = share of scenarios where the preferred
    level was saved. It is NOT the published AMCE (that needs the full
    conjoint coding). It is a first look and a format check.
  * Never open the CSV in Excel and save it: Excel corrupts long IDs.
"""
from pathlib import Path
import numpy as np
import polars as pl

# ----------------------------- SETTINGS -----------------------------------
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="SharedResponses.csv", help="path to the CSV file")
ap.add_argument("--out", default="outputs", help="folder for results (one folder per CSV!)")
args = ap.parse_args()

CSV = args.csv
OUT = Path(args.out)
MIN_SESSIONS_COUNTRY = 500                       # countries shown in the table
ALWAYS_SHOW = ["DZA"]                            # shown even if small
# Preferred level in each single-dimension scenario type (AttributeLevel value)
PREFERRED = {
    "Species": "Hoomans",
    "Age": "Young",
    "Fitness": "Fit",
    "Gender": "Female",
    "Social Status": "High",
    "Utilitarian": "More",
}
# --------------------------------------------------------------------------

pl.Config.set_tbl_rows(60)      # show full tables instead of "..."
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)

OUT.mkdir(exist_ok=True)
PARQUET = OUT / (Path(CSV).stem + ".parquet")   # named after the CSV


def run(lazy):
    """Collect a lazy query, using the streaming engine to save memory."""
    try:
        return lazy.collect(engine="streaming")
    except TypeError:                      # older polars versions
        return lazy.collect(streaming=True)


# --------------------- 1. CSV -> Parquet (once) ---------------------------
if not PARQUET.exists():
    header = pl.read_csv(CSV, n_rows=5)
    # keep IDs as text: long numbers lose precision otherwise
    overrides = {c: pl.Utf8 for c in ["ResponseID", "ExtendedSessionID", "UserID"]
                 if c in header.columns}
    print("Converting CSV to Parquet (a few minutes for a big file)...")
    pl.scan_csv(CSV, schema_overrides=overrides,
                infer_schema_length=10_000).sink_parquet(PARQUET)
    print("Saved", PARQUET)

lf = pl.scan_parquet(PARQUET)
cols = lf.collect_schema().names()

# --------------------- 2. Structure ---------------------------------------
print("\n=== STRUCTURE ===")
print("Columns:", cols)
print(lf.collect_schema())
n_rows = run(lf.select(pl.len())).item()
n_sess = run(lf.select(pl.col("ExtendedSessionID").n_unique())).item()
print(f"Rows: {n_rows:,} | Sessions: {n_sess:,}")

print("\nMissing values per column:")
print(run(lf.select(pl.all().null_count())).transpose(
    include_header=True, column_names=["missing"]).filter(pl.col("missing") > 0))

# Rows per scenario / per session, on a 1% sample of sessions (cheap even for huge files)
print("\n=== ROW STRUCTURE (1% sample of sessions) ===")
samp = lf.filter(pl.col("ExtendedSessionID").hash() % 100 == 0)
if "ScenarioOrder" in cols:
    rows_per_scn = run(samp.group_by("ExtendedSessionID", "ScenarioOrder").len()
                           .group_by("len").agg(pl.len().alias("n_scenarios")).sort("len")
                           .rename({"len": "rows_per_scenario"}))
    print("How many rows describe one scenario (session x ScenarioOrder)?")
    print(rows_per_scn)
rows_per_sess = run(samp.group_by("ExtendedSessionID").len()
                        .select(pl.col("len").min().alias("min"),
                                pl.col("len").median().alias("median"),
                                pl.col("len").mean().round(1).alias("mean"),
                                pl.col("len").max().alias("max")))
print("Rows per session (summary):")
print(rows_per_sess)

for c in ["ScenarioType", "Saved", "Intervention", "PedPed", "Barrier",
          "CrossingSignal", "DiffNumberOFCharacters", "AttributeLevel"]:
    if c in cols:
        print(f"\n{c}:")
        print(run(lf.group_by(c).len().sort("len", descending=True)))

# Survey columns (only present in SharedResponsesSurvey.csv)
review = [c for c in cols if c.startswith("Review_")]
if review:
    print("\n=== SURVEY (demographic) COLUMNS ===")
    for c in review:
        print(f"\n{c}:")
        print(run(lf.group_by(c).len().sort("len", descending=True)).head(15))

# --------------------- 3. Countries ---------------------------------------
print("\n=== COUNTRIES ===")
country = run(
    lf.group_by("UserCountry3")
      .agg(pl.len().alias("rows"), pl.col("ExtendedSessionID").n_unique().alias("sessions"))
      .sort("rows", descending=True)
)
country.write_csv(OUT / "country_counts.csv")
print(country.head(30))
print(f"Number of countries: {country.height}")
arab = ["DZA", "MAR", "TUN", "LBY", "EGY", "SDN", "SAU", "ARE", "QAT", "KWT", "BHR",
        "OMN", "YEM", "IRQ", "SYR", "JOR", "LBN", "PSE", "MRT"]
print("\nArab-region countries (rows, sessions):")
print(country.filter(pl.col("UserCountry3").is_in(arab)))

# --------------------- 4. Crude effects -----------------------------------
print("\n=== CRUDE PREFERENCE EFFECTS (cluster = session) ===")
# indicator: was the PREFERRED level saved?
parts = []
for st, level in PREFERRED.items():
    parts.append(
        pl.when(pl.col("ScenarioType") == st)
          .then(pl.when(pl.col("AttributeLevel") == level)
                  .then(pl.col("Saved")).otherwise(1 - pl.col("Saved")))
    )
ps = pl.coalesce(parts).alias("pref_saved")

per_session = run(
    lf.filter(pl.col("ScenarioType").is_in(list(PREFERRED)))
      .with_columns(ps)
      .group_by("UserCountry3", "ScenarioType", "ExtendedSessionID")
      .agg(pl.col("pref_saved").sum().alias("s"), pl.len().alias("n"))
)


def crude(s, n):
    """Ratio estimate P = sum(s)/sum(n) with session-clustered 95% CI for 2P-1."""
    s, n = np.asarray(s, float), np.asarray(n, float)
    G, N = len(s), n.sum()
    P = s.sum() / N
    resid = s - P * n
    se = np.sqrt(G / max(G - 1, 1) * (resid ** 2).sum()) / N
    eff = 2 * P - 1
    return eff, 2 * 1.96 * se, int(N), G


rows = []
# global
for (st,), g in per_session.group_by("ScenarioType"):
    eff, ci, N, G = crude(g["s"], g["n"])
    rows.append(("ALL", st, N, G, eff, eff - ci, eff + ci))
# by country
for (cty, st), g in per_session.group_by("UserCountry3", "ScenarioType"):
    eff, ci, N, G = crude(g["s"], g["n"])
    rows.append((cty, st, N, G, eff, eff - ci, eff + ci))

res = pl.DataFrame(rows, orient="row",
                   schema=["country", "dimension", "n_rows", "n_sessions",
                           "effect", "ci_low", "ci_high"])
res.write_csv(OUT / "crude_effects_by_country.csv")

print("\nGlobal:")
print(res.filter(pl.col("country") == "ALL").sort("effect", descending=True)
         .with_columns(pl.col(["effect", "ci_low", "ci_high"]).round(3)))

keep = (country.filter((pl.col("sessions") >= MIN_SESSIONS_COUNTRY) |
                       pl.col("UserCountry3").is_in(ALWAYS_SHOW))["UserCountry3"].to_list())
wide = (res.filter(pl.col("country").is_in(keep))
           .pivot(on="dimension", index="country", values="effect")
           .with_columns(pl.exclude("country").round(2)))
print(f"\nCrude effect by country (sessions >= {MIN_SESSIONS_COUNTRY}, plus {ALWAYS_SHOW}):")
with pl.Config(tbl_rows=100, tbl_cols=20):
    print(wide.sort("country"))

print("\nDone. Files written to:", OUT.resolve())
