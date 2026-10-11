"""Shared helpers for the W06 / W07 / capstone notebooks (Lane 2: refresh scoring).

Everything here is small on purpose. The notebooks explain *why*; this file only
keeps the same logic from being copy-pasted three times (and drifting apart).

Timeline used throughout (all windows end at the export date, "now"):

    days 61-90 back | days 31-60 back | days 1-30 back
    --------------- | --------------- | --------------
      feature only  |  feature + label |  LABEL ONLY
                      denominator        (impressions_last_30d)

The label (`trend_direction == "down"`) compares days 1-30 with days 31-60.
A model that is meant to *forecast* a decline may only see information that
existed before days 1-30 started. So the "honest" feature set pretends the
decision is made 30 days ago ("T-30") and uses only days 31-90 + static metadata.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS = 5
N_SEEDS = 5                 # 5 seeds x 5 folds = 25 held-out client groups
MEASURABLE_FLOOR = 30       # min impressions in the label's denominator window (~1/day)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.append(str(_REPO / "scripts"))
from ml_utils import precision_at_k  # noqa: E402  (repo's own implementation)

# Columns that must never be features (label source, target, ids, product metadata)
BANNED = {"trend_direction", "trend_pct", "is_declining_label", "y",
          "content_id", "client_id", "provider_used", "model_used"}

# W05 feature set, kept verbatim so the before/after comparison is fair
W05_NUMERIC = ["days_since_last_update", "ctr", "avg_position", "impressions_90d",
               "clicks_90d", "engagement_rate", "scroll_rate", "content_age_days",
               "word_count"]
W05_CATEGORICAL = ["position_tier", "freshness_tier", "content_type", "main_intent"]

POSITION_ORDER = ["top_3", "page_1", "striking", "page_3_5", "deep"]


def load_data() -> pd.DataFrame:
    """Starter CSV + label + the days 61-90 sub-window (derived, not new data)."""
    for p in [_REPO / "data/raw/content_refresh_anonymized.csv",
              Path("data/raw/content_refresh_anonymized.csv")]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("content_refresh_anonymized.csv not found")
    df["y"] = df["trend_direction"].str.lower().eq("down").astype(int)
    for m in ["impressions", "clicks", "sessions"]:
        df[f"{m}_61_90d"] = (df[f"{m}_90d"] - df[f"{m}_last_30d"]
                             - df[f"{m}_prev_30d"]).clip(lower=0)
    return df


def measurable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where a 20% drop means something: >= MEASURABLE_FLOOR impressions in days 31-60."""
    return df[df["impressions_prev_30d"] >= MEASURABLE_FLOOR].copy()


def w05_features(f: pd.DataFrame) -> pd.DataFrame:
    num = f[W05_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0)
    cat = pd.get_dummies(f[W05_CATEGORICAL].fillna("unknown").astype(str),
                         prefix=W05_CATEGORICAL)
    return pd.concat([num, cat], axis=1).astype(float)


def honest_features(f: pd.DataFrame, as_of: str = "T-30") -> pd.DataFrame:
    """Features knowable at decision time.

    as_of="T-30": training/validation. "Recent month" = days 31-60, "month before" = 61-90.
    as_of="now":  scoring the live queue. "Recent month" = days 1-30, "month before" = 31-60.
    Same columns, same meaning, shifted 30 days -> the model forecasts the NEXT 30 days.
    """
    if as_of == "T-30":
        recent, before, age_shift = "prev_30d", "61_90d", 30
    elif as_of == "now":
        recent, before, age_shift = "last_30d", "prev_30d", 0
    else:
        raise ValueError(as_of)
    o = pd.DataFrame(index=f.index)
    o["log_impressions_recent"] = np.log1p(f[f"impressions_{recent}"])
    o["log_impressions_before"] = np.log1p(f[f"impressions_{before}"])
    o["impression_momentum"] = o["log_impressions_recent"] - o["log_impressions_before"]
    o["log_clicks_recent"] = np.log1p(f[f"clicks_{recent}"])
    o["log_clicks_before"] = np.log1p(f[f"clicks_{before}"])
    imp = f[f"impressions_{recent}"] + f[f"impressions_{before}"]
    clk = f[f"clicks_{recent}"] + f[f"clicks_{before}"]
    o["ctr_60d"] = np.where(imp > 0, clk / imp.clip(lower=1) * 100, 0.0)
    o["log_sessions_recent"] = np.log1p(f[f"sessions_{recent}"])
    o["log_sessions_before"] = np.log1p(f[f"sessions_{before}"])
    o["content_age_days"] = f["content_age_days"] - age_shift
    # missingness follows content_type -> explicit has_ flags instead of silent zeros
    o["word_count"] = f["word_count"].fillna(0)
    o["has_word_count"] = f["word_count"].notna().astype(int)
    for k in ["search_volume", "competition", "cpc"]:
        o[k] = f[k].fillna(0)
    o["has_keyword_data"] = f["search_volume"].notna().astype(int)
    cats = pd.get_dummies(f[["content_type", "main_intent"]].fillna("unknown").astype(str))
    return pd.concat([o, cats], axis=1).astype(float)


def w04_rule_score(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """W04 rule: stale (>=91d) AND ctr < half its tier benchmark, scored by impressions.
    Benchmark is fit on TRAIN clients only."""
    tr = train[train["position_tier"].isin(POSITION_ORDER)]
    bench = tr.groupby("position_tier")[["clicks_90d", "impressions_90d"]].sum()
    bench = bench["clicks_90d"] / bench["impressions_90d"] * 100
    b = test["position_tier"].map(bench)
    stale = (test["days_since_last_update"] >= 91).astype(int)
    gap = ((test["ctr"] < 0.5 * b) & test["position_tier"].isin(POSITION_ORDER)).astype(int)
    return (stale * gap * test["impressions_90d"]).fillna(0).to_numpy(dtype=float)


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=25,
                                  class_weight="balanced_subsample",
                                  random_state=RANDOM_STATE, n_jobs=-1)


def make_lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def splits(frame: pd.DataFrame, kind: str = "grouped"):
    """Yield (seed, fold, train_idx, test_idx). grouped = no client on both sides."""
    for seed in range(N_SEEDS):
        if kind == "grouped":
            sp = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            it = sp.split(frame, groups=frame["client_id"])
        elif kind == "random":
            it = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(frame)
        else:
            raise ValueError(kind)
        for fold, (tr, te) in enumerate(it):
            yield seed, fold, tr, te


def per_client_precision(test: pd.DataFrame, scores: np.ndarray, k: int = 10) -> float:
    """Mean precision@k inside each test client's own queue (clients with >= k rows).
    Mirrors real use: an editor works one client's site at a time."""
    vals = []
    s = pd.Series(scores, index=test.index)
    for _, g in test.groupby("client_id"):
        if len(g) >= k:
            vals.append(precision_at_k(g["y"], s.loc[g.index], k))
    return float(np.mean(vals)) if vals else np.nan
