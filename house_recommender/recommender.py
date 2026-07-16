"""
Content-based ML recommender.

Numeric features are scaled with scikit-learn's MinMaxScaler (learned from the
data), each requested criterion contributes a 0-1 similarity, and criteria are
combined by weight. `similar_to()` uses a k-Nearest-Neighbours model over the
scaled feature space, using *weighted Euclidean* distance so neighbours are the
houses closest in absolute price/size/rooms (importance-weighted), rather than
cosine, which only compared vector direction and made nearly everything ~98%.

Returned house dicts add: match_score (0-100), score_breakdown (criterion -> %),
and Price_per_sqft.
"""

import os
import sqlite3
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAS_FUZZ = True
except ImportError:  # graceful fallback if not installed
    _HAS_FUZZ = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
# Default to the system temp dir (always writable). Must match load_data.py so
# both modules read/write the same database file.
DB_PATH = os.environ.get("HOUSES_DB", os.path.join(tempfile.gettempdir(), "houses.db"))

DEFAULT_WEIGHTS = {
    "budget": 35,
    "area": 20,
    "bedrooms": 15,
    "bathrooms": 10,
    "house_size": 10,
    "location": 10,
}
# Kept as alias for any older imports.
WEIGHTS = DEFAULT_WEIGHTS

BUDGET_TOLERANCE = 0.10
POPULARITY_BOOST_MAX = 5.0   # max additive % boost from popularity signal

# The numeric columns fed to the ML feature scaler.
NUMERIC_FEATURES = ["Budget_BDT", "Area_sqft", "Bedrooms", "Bathrooms"]
# Ordinal encoding for the size category (used both as a feature and for scoring).
SIZE_ORDINAL = {"small": 0.0, "medium": 0.5, "large": 1.0}

# Per-feature importance applied to the kNN feature space, in the column order
# [budget, area, bedrooms, bathrooms, house_size]. These mirror DEFAULT_WEIGHTS
# so "similar" reflects what users actually care about (budget/area more than a
# bathroom). Each feature is multiplied by sqrt(normalised weight) so that plain
# Euclidean distance becomes a *weighted* Euclidean distance.
KNN_FEATURE_WEIGHTS = {
    "budget": 35, "area": 20, "bedrooms": 15, "bathrooms": 10, "house_size": 10,
}


def _knn_weight_vector():
    """sqrt of the normalised feature weights, in feature-column order."""
    keys = ["budget", "area", "bedrooms", "bathrooms", "house_size"]
    w = np.array([KNN_FEATURE_WEIGHTS[k] for k in keys], dtype=float)
    return np.sqrt(w / w.sum())

# Maps a scoring criterion -> (weight key, dataframe column, preference key).
NUMERIC_CRITERIA = [
    ("budget", "Budget_BDT", "budget_bdt"),
    ("area", "Area_sqft", "area_sqft"),
    ("bedrooms", "Bedrooms", "bedrooms"),
    ("bathrooms", "Bathrooms", "bathrooms"),
]


def load_houses():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    if {"Area_sqft", "Budget_BDT"}.issubset(df.columns):
        df["Price_per_sqft"] = (df["Budget_BDT"] / df["Area_sqft"]).round(0)
    return df


def _numeric_frame(df):
    """Coerce the numeric feature columns and fill gaps with the column median."""
    num = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    return num.fillna(num.median())


@lru_cache(maxsize=1)
def _fit_model():
    """Fit (and cache) the scaler + kNN model on the full dataset.

    Returns (df, scaler, feature_matrix, nn_model). Cached for the life of the
    process; ensure_database() runs before this on startup, so the data is ready.
    """
    df = load_houses()
    num = _numeric_frame(df)
    scaler = MinMaxScaler().fit(num)
    scaled = scaler.transform(num)

    size = (df["House_Size"].astype(str).str.lower().map(SIZE_ORDINAL)
            .fillna(0.5).to_numpy().reshape(-1, 1))
    # Apply per-feature weights, then use Euclidean distance. Weighted Euclidean
    # ranks by *absolute* closeness in size/price (unlike cosine, which only
    # compared the profile's direction and made almost everything look ~98%).
    feats = np.hstack([scaled, size]) * _knn_weight_vector()

    nn = NearestNeighbors(metric="euclidean").fit(feats)
    return df, scaler, feats, nn


def _scale_value(scaler, column, value):
    """Scale one raw value into the model's 0..1 space for a single column."""
    j = NUMERIC_FEATURES.index(column)
    lo = scaler.data_min_[j]
    rng = scaler.data_max_[j] - lo
    if rng == 0:
        return 0.0
    return float(np.clip((float(value) - lo) / rng, 0.0, 1.0))


def _location_sim(locations, query):
    s = locations.fillna("").astype(str).str.lower()
    q = str(query).lower().strip()
    if not q:
        return pd.Series(0.0, index=s.index)
    if _HAS_FUZZ:
        return s.apply(lambda x: fuzz.partial_ratio(x, q) / 100.0 if x else 0.0)
    return s.str.contains(q, na=False).astype(float)


def recommend(preferences, top_n=10, weights=None, popularity=None):
    """
    preferences: dict with any of district, location, house_size, bedrooms,
                 bathrooms, area_sqft, budget_bdt, max_price_per_sqft.
    weights:     optional per-user override of DEFAULT_WEIGHTS.
    popularity:  optional dict {(district_lower, location_lower): count}.
                 Adds a small bonus to houses popular among other users.

    Ranking is content-based: each requested criterion contributes a
    similarity in [0, 1] (numeric features are compared in the sklearn-scaled
    space), and criteria are combined by their weights.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df, scaler, _feats, _nn = _fit_model()
    p = {k: v for k, v in preferences.items() if v not in (None, "", "Any")}

    # ---- Hard filters
    mask = pd.Series(True, index=df.index)
    if p.get("district"):
        mask &= df["District"].str.lower() == str(p["district"]).lower()
    budget = p.get("budget_bdt")
    if budget:
        mask &= df["Budget_BDT"] <= budget * (1 + BUDGET_TOLERANCE)
    if p.get("max_price_per_sqft") and "Price_per_sqft" in df.columns:
        mask &= df["Price_per_sqft"] <= p["max_price_per_sqft"]

    sub = df[mask].copy()
    if sub.empty:
        return []

    # Scaled numeric features for the surviving candidates.
    scaled = pd.DataFrame(
        scaler.transform(_numeric_frame(df).loc[sub.index]),
        index=sub.index, columns=NUMERIC_FEATURES,
    )

    components = {}   # name -> per-row similarity (Series, 0..1)
    used_w = []

    # Numeric criteria: similarity = 1 - |scaled_query - scaled_row|.
    for name, col, pref_key in NUMERIC_CRITERIA:
        if p.get(pref_key) is None:
            continue
        q = _scale_value(scaler, col, p[pref_key])
        components[name] = (1.0 - (scaled[col] - q).abs()).clip(0, 1)
        used_w.append(weights[name])

    # Size category: ordinal closeness.
    if p.get("house_size"):
        q_ord = SIZE_ORDINAL.get(str(p["house_size"]).lower(), 0.5)
        row_ord = sub["House_Size"].astype(str).str.lower().map(SIZE_ORDINAL).fillna(0.5)
        components["house_size"] = (1.0 - (row_ord - q_ord).abs()).clip(0, 1)
        used_w.append(weights["house_size"])

    # Location: fuzzy text similarity.
    if p.get("location"):
        components["location"] = _location_sim(sub["Location"], p["location"])
        used_w.append(weights["location"])

    total_w = sum(used_w) if used_w else 1
    if components:
        raw = sum(weights[name] * comp for name, comp in components.items())
        score = raw / total_w * 100
    else:
        score = pd.Series(0.0, index=sub.index)

    # Collaborative-ish popularity boost: small, additive, capped.
    if popularity:
        max_count = max(popularity.values())
        keys = list(zip(sub["District"].str.lower(), sub["Location"].str.lower()))
        boost = pd.Series(
            [POPULARITY_BOOST_MAX * (popularity.get(k, 0) / max_count) for k in keys],
            index=sub.index,
        )
        score = (score + boost).clip(upper=100)

    sub["match_score"] = score.round(1)
    sub = sub.sort_values("match_score", ascending=False).head(top_n).copy()

    # Per-row breakdown (% for each criterion used) — only for top_n rows.
    breakdowns = []
    for idx in sub.index:
        b = {name: round(float(comp.loc[idx]) * 100, 1)
             for name, comp in components.items()}
        breakdowns.append(b)
    sub["score_breakdown"] = breakdowns

    return sub.to_dict(orient="records")


def similar_to(house, top_n=5, weights=None):
    """Recommend houses similar to a given one (used from the Favorites tab).

    Uses a k-Nearest-Neighbours model (weighted Euclidean distance) over the
    scaled feature space — a content-based ML retrieval rather than re-running
    the weighted scorer.
    """
    df, scaler, _feats, nn = _fit_model()

    # Build this house's feature vector in the model's space.
    raw = pd.DataFrame([[house.get(c) for c in NUMERIC_FEATURES]], columns=NUMERIC_FEATURES)
    raw = raw.apply(pd.to_numeric, errors="coerce").fillna(_numeric_frame(df).median())
    scaled = scaler.transform(raw)
    size = SIZE_ORDINAL.get(str(house.get("House_Size", "")).lower(), 0.5)
    # Same per-feature weighting the model was fit with.
    vec = np.hstack([scaled, [[size]]]) * _knn_weight_vector()

    k = min(top_n + 5, len(df))
    distances, indices = nn.kneighbors(vec, n_neighbors=k)

    out = []
    for dist, i in zip(distances[0], indices[0]):
        row = df.iloc[i].to_dict()
        same = (
            row.get("District") == house.get("District")
            and row.get("Location") == house.get("Location")
            and abs(float(row.get("Budget_BDT") or 0) - float(house.get("Budget_BDT") or 0)) < 1
            and abs(float(row.get("Area_sqft") or 0) - float(house.get("Area_sqft") or 0)) < 1
        )
        if same:
            continue
        # weighted-Euclidean distance -> similarity %. 1/(1+d) maps distance 0
        # to 100% and decays smoothly as houses get further apart.
        row["match_score"] = round(100.0 / (1.0 + float(dist)), 1)
        out.append(row)
        if len(out) >= top_n:
            break
    return out


if __name__ == "__main__":
    demo = {
        "district": "Dhaka",
        "house_size": "Medium",
        "bedrooms": 3,
        "bathrooms": 3,
        "area_sqft": 1400,
        "budget_bdt": 9000000,
    }
    for h in recommend(demo, top_n=5):
        print(f"{h['match_score']:5.1f}  {h['District']:11} {h['Location'][:20]:20} "
              f"{h['Bedrooms']}bd/{h['Bathrooms']}ba  {h['Area_sqft']}sqft  "
              f"{h['Budget_BDT']:,.0f} BDT  ppsqft={h.get('Price_per_sqft')}  "
              f"why={h['score_breakdown']}")
