"""
check_parquets.py  -  tells you what each Parquet file contains.
It only READS the files. It does not change or delete anything.

Run from the folder that contains the 'outputs' folder:
    python check_parquets.py
Needs: pip install polars pyarrow
"""
from pathlib import Path
import polars as pl

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)

FILES = sorted(Path("outputs").glob("*.parquet"))
print("Parquet files found:", [f.name for f in FILES])


def run(lazy):
    try:
        return lazy.collect(engine="streaming")
    except TypeError:
        return lazy.collect(streaming=True)


for f in FILES:
    lf = pl.scan_parquet(f)
    cols = lf.collect_schema().names()
    print("\n" + "=" * 70)
    print(f.name, f"({f.stat().st_size / 1e9:.2f} GB)")
    print("=" * 70)

    rows = run(lf.select(pl.len())).item()
    sessions = run(lf.select(pl.col("ExtendedSessionID").n_unique())).item()
    users = run(lf.select(pl.col("UserID").n_unique())).item() if "UserID" in cols else None
    print(f"Rows: {rows:,}")
    print(f"Distinct sessions: {sessions:,}")
    print(f"Distinct users (UserID): {users:,}" if users is not None else "No UserID column")
    print(f"Rows per session (average): {rows / sessions:.1f}")
    print(f"Number of columns: {len(cols)}")
    review = [c for c in cols if c.startswith("Review_")]
    print("Has survey (Review_*) columns:", bool(review))
    nonnull_age = run(lf.select(pl.col("Review_age").is_not_null().sum())).item() if "Review_age" in cols else None
    if nonnull_age is not None:
        print(f"Rows with a survey age filled in: {nonnull_age:,}")

    # How many rows per session, on a 1% sample of sessions
    samp = lf.filter(pl.col("ExtendedSessionID").hash() % 100 == 0)
    dist = run(samp.group_by("ExtendedSessionID").len()
                   .group_by("len").agg(pl.len().alias("n_sessions")).sort("n_sessions", descending=True)
                   .head(6).rename({"len": "rows_in_session"}))
    print("\nMost common session sizes (1% sample):")
    print(dist)

    if "ScenarioOrder" in cols:
        per_scn = run(samp.group_by("ExtendedSessionID", "ScenarioOrder").len()
                          .group_by("len").agg(pl.len().alias("n")).sort("len")
                          .rename({"len": "rows_per_scenario"}))
        print("Rows per scenario (1% sample):")
        print(per_scn)

    if "UserCountry3" in cols:
        c = run(lf.group_by("UserCountry3").agg(pl.col("ExtendedSessionID").n_unique().alias("sessions"))
                  .sort("sessions", descending=True))
        print(f"\nCountries: {c.height}. Top 5:")
        print(c.head(5))
        print("DZA sessions:", c.filter(pl.col("UserCountry3") == "DZA")["sessions"].to_list())

print("\nDone.")
