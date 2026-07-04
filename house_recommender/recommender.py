"""
recommender.py
--------------
Content-based weighted scoring engine with score breakdowns,
fuzzy location matching, similar-house lookup, and optional
personalized weights / popularity boost.

Returned house dicts include:
    match_score      overall fit %, 0-100
    score_breakdown  dict criterion -> %, so the UI can explain WHY
    Price_per_sqft   derived column
"""

import os
import sqlite3
import tempfile
import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAS_FUZZ = True
except ImportError:  # graceful fallback if not installed
    _HAS_FUZZ = False

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


def load_houses():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    if {"Area_sqft", "Budget_BDT"}.issubset(df.columns):
        df["Price_per_sqft"] = (df["Budget_BDT"] / df["Area_sqft"]).round(0)
    return df


def _location_score(locations, query):
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
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    df = load_houses()
    p = {k: v for k, v in preferences.items() if v not in (None, "", "Any")}

    # ---- Hard filters
    if p.get("district"):
        df = df[df["District"].str.lower() == str(p["district"]).lower()]

    budget = p.get("budget_bdt")
    if budget:
        df = df[df["Budget_BDT"] <= budget * (1 + BUDGET_TOLERANCE)]

    if p.get("max_price_per_sqft") and "Price_per_sqft" in df.columns:
        df = df[df["Price_per_sqft"] <= p["max_price_per_sqft"]]

    if df.empty:
        return []

    components = {}   # name -> per-row contribution (Series, 0..weight)
    used = []

    if budget:
        w = weights["budget"]
        over = (df["Budget_BDT"] - budget).clip(lower=0)
        s = (1 - over / (budget * BUDGET_TOLERANCE)).clip(0, 1)
        components["budget"] = w * s
        used.append(w)

    if p.get("area_sqft"):
        w = weights["area"]
        target = float(p["area_sqft"])
        diff = (df["Area_sqft"] - target).abs() / max(target, 1)
        components["area"] = w * (1 - diff).clip(0, 1)
        used.append(w)

    if p.get("bedrooms"):
        w = weights["bedrooms"]
        diff = (df["Bedrooms"] - p["bedrooms"]).abs() / 3.0
        components["bedrooms"] = w * (1 - diff).clip(0, 1).fillna(0)
        used.append(w)

    if p.get("bathrooms"):
        w = weights["bathrooms"]
        diff = (df["Bathrooms"] - p["bathrooms"]).abs() / 3.0
        components["bathrooms"] = w * (1 - diff).clip(0, 1).fillna(0)
        used.append(w)

    if p.get("house_size"):
        w = weights["house_size"]
        s = (df["House_Size"].str.lower() == str(p["house_size"]).lower()).astype(float)
        components["house_size"] = w * s
        used.append(w)

    if p.get("location"):
        w = weights["location"]
        components["location"] = w * _location_score(df["Location"], p["location"])
        used.append(w)

    total_w = sum(used) if used else 1
    raw = sum(components.values()) if components else pd.Series(0.0, index=df.index)
    score = raw / total_w * 100

    # Collaborative-ish popularity boost: small, additive, capped.
    if popularity:
        max_count = max(popularity.values())
        keys = list(zip(df["District"].str.lower(), df["Location"].str.lower()))
        boost = pd.Series(
            [POPULARITY_BOOST_MAX * (popularity.get(k, 0) / max_count) for k in keys],
            index=df.index,
        )
        score = (score + boost).clip(upper=100)

    df = df.copy()
    df["match_score"] = score.round(1)
    df = df.sort_values("match_score", ascending=False).head(top_n).copy()

    # Per-row breakdown (% of max for each criterion used) — only for top_n.
    breakdowns = []
    for idx in df.index:
        b = {}
        for name, contrib in components.items():
            w = weights.get(name, 0)
            if w > 0:
                b[name] = round(float(contrib.loc[idx]) / w * 100, 1)
        breakdowns.append(b)
    df["score_breakdown"] = breakdowns

    return df.to_dict(orient="records")


def similar_to(house, top_n=5, weights=None):
    """Recommend houses similar to a given one (used from the Favorites tab)."""
    prefs = {
        "district": house.get("District"),
        "location": house.get("Location"),
        "house_size": house.get("House_Size"),
        "bedrooms": house.get("Bedrooms"),
        "bathrooms": house.get("Bathrooms"),
        "area_sqft": house.get("Area_sqft"),
        "budget_bdt": house.get("Budget_BDT"),
    }
    results = recommend(prefs, top_n=top_n + 1, weights=weights)
    out = []
    for r in results:
        same = (
            r.get("District") == house.get("District")
            and r.get("Location") == house.get("Location")
            and abs(float(r.get("Budget_BDT") or 0) - float(house.get("Budget_BDT") or 0)) < 1
            and abs(float(r.get("Area_sqft") or 0) - float(house.get("Area_sqft") or 0)) < 1
        )
        if not same:
            out.append(r)
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
