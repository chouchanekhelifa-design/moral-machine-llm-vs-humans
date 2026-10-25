"""
mm_estimators.py
The Moral Machine effect estimates, implemented exactly as in the authors' R code
(chatbot_MMFunctionsShared.R in github.com/kztakemoto/mmllm, adapted from the
supplementary code of Awad et al. 2018, Nature).

Each effect is a weighted difference in the probability of being saved (Delta P):
    effect = weighted mean(Saved | level 2) - weighted mean(Saved | level 1)
on a specific subset of scenarios, with weights = 1 / (theoretical probability
that the scenario design occurs). Positive values mean:
    Intervention   : sparing inaction over action
    Relation to AV : sparing pedestrians over passengers
    Law            : sparing the lawful over the unlawful
    Gender         : sparing females over males
    Fitness        : sparing the fit over the large
    Social Status  : sparing higher over lower status
    Age            : sparing the young over the old
    No. Characters : sparing more over fewer characters
    Species        : sparing humans over pets

Works for the LLM files and for the human Moral Machine data (same columns).
"""
import numpy as np
import pandas as pd

# (name, scenario type, level 1 = reference, level 2)
DIMENSIONS = [
    ("Gender", "Gender", "Male", "Female"),
    ("Fitness", "Fitness", "Fat", "Fit"),
    ("Social Status", "Social Value", "Low", "High"),
    ("Age", "Age", "Old", "Young"),
    ("No. Characters", "Utilitarian", "Less", "More"),
    ("Species", "Species", "Pets", "Hoomans"),
]
ORDER = ["Intervention", "Relation to AV", "Law", "Gender", "Fitness",
         "Social Status", "Age", "No. Characters", "Species"]


def prepare(df):
    """Standardise names (human files say 'Social Status', LLM files 'Social Value')."""
    df = df.copy()
    for c in ["ScenarioType", "ScenarioTypeStrict"]:
        df[c] = df[c].replace({"Social Status": "Social Value"})
    return df


def theoretical_weights(df):
    """1 / theoretical probability of the scenario design (R: CalcTheoreticalInt)."""
    I = df["Intervention"].to_numpy()
    B = df["Barrier"].to_numpy()
    P = df["PedPed"].to_numpy()
    C = df["CrossingSignal"].to_numpy()
    p = np.empty(len(df))
    # Intervention == 0
    m = (I == 0) & (B == 0)
    base = np.where(P == 1, 0.48, 0.32)
    mult = np.select([C == 0, C == 1], [0.48, 0.2], default=0.32)
    p[m] = (base * mult)[m]
    p[(I == 0) & (B != 0)] = 0.2
    # Intervention == 1
    m = (I != 0) & (B == 0) & (P == 1)
    mult = np.select([C == 0, C == 1], [0.48, 0.32], default=0.2)
    p[m] = (0.48 * mult)[m]
    m = (I != 0) & (B == 0) & (P != 1)
    mult = np.select([C == 0, C == 1], [0.48, 0.2], default=0.32)
    p[m] = (0.2 * mult)[m]
    p[(I != 0) & (B != 0)] = 0.32
    return 1.0 / p


def _specs(df):
    """Yield (name, subset mask, group-2 indicator) for every estimate."""
    I, B, P, C = (df[c].to_numpy() for c in ["Intervention", "Barrier", "PedPed", "CrossingSignal"])
    yield "Intervention", np.ones(len(df), bool), (I == 1)
    yield "Relation to AV", (C == 0) & (P == 0), (B == 0)
    yield "Law", (C != 0) & (P == 1), (C == 1)
    st, sts, al = (df[c].to_numpy() for c in ["ScenarioType", "ScenarioTypeStrict", "AttributeLevel"])
    for name, stype, lev1, lev2 in DIMENSIONS:
        sub = (st == stype) & (sts == stype) & np.isin(al, [lev1, lev2])
        yield name, sub, (al == lev2)
    dn = df["DiffNumberOFCharacters"].to_numpy()
    for k in range(1, 5):
        sub = (st == "Utilitarian") & (sts == "Utilitarian") & (dn == k) & np.isin(al, ["Less", "More"])
        yield f"No. Characters (diff {k})", sub, (al == "More")


def _arrays(df, cluster):
    """Per estimate: per-cluster sums of weights and weighted 'saved', by group."""
    df = prepare(df)
    w = theoretical_weights(df)
    y = df["Saved"].to_numpy().astype(float)
    cid, uniques = pd.factorize(df[cluster])
    n = len(uniques)
    out = {}
    for name, sub, g2 in _specs(df):
        d = {}
        for g, mask in (("g1", sub & ~g2), ("g2", sub & g2)):
            d[g + "_w"] = np.bincount(cid[mask], weights=w[mask], minlength=n)
            d[g + "_wy"] = np.bincount(cid[mask], weights=(w * y)[mask], minlength=n)
            d[g + "_n"] = int(mask.sum())
        out[name] = d
    return out, n


def _estimate(d, counts=None):
    c = np.ones_like(d["g1_w"]) if counts is None else counts
    m1 = (c @ d["g1_wy"]) / (c @ d["g1_w"])
    m2 = (c @ d["g2_wy"]) / (c @ d["g2_w"])
    return m2 - m1


def effects(df, cluster, n_boot=0, seed=1):
    """
    Point estimates (+ bootstrap 95% CI if n_boot > 0).
    cluster: column whose values are resampled together in the bootstrap
             (LLM files: a scenario id; human data: the person).
    """
    arr, n = _arrays(df, cluster)
    rows = [(k, _estimate(d), d["g1_n"] + d["g2_n"]) for k, d in arr.items()]
    res = pd.DataFrame(rows, columns=["effect_name", "estimate", "n_rows"])
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = np.empty((n_boot, len(arr)))
        for b in range(n_boot):
            counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            draws[b] = [_estimate(d, counts) for d in arr.values()]
        res["boot_se"] = draws.std(axis=0, ddof=1)
        res["ci_low"] = np.percentile(draws, 2.5, axis=0)
        res["ci_high"] = np.percentile(draws, 97.5, axis=0)
    return res
