"""
auth.py
-------
Favorites and search history for the House Recommendation System.

All user data lives in a SEPARATE database (`users.db`) so it is never touched
when `houses.db` is rebuilt from the CSV. There are no accounts — everything
is stored under a single local profile.
"""

import os
import json
import sqlite3
import tempfile
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Default to the system temp dir (always writable). The repo mount on cloud
# hosts (e.g. Streamlit Cloud) can be read-only, which breaks writing this file.
USERS_DB = os.environ.get("USERS_DB", os.path.join(tempfile.gettempdir(), "users.db"))


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _connect():
    return sqlite3.connect(USERS_DB)


def init_user_db():
    """Create the user tables if they don't exist yet."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            district   TEXT, location TEXT, house_size TEXT,
            bedrooms   REAL, bathrooms REAL, area_sqft REAL,
            budget_bdt REAL, match_score REAL,
            saved_at   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT NOT NULL,
            prefs_json  TEXT NOT NULL,
            searched_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------

def add_favorite(username, house):
    conn = _connect()
    # Avoid duplicates (same user + same house essentials)
    exists = conn.execute(
        """SELECT 1 FROM favorites WHERE username=? AND district=? AND location=?
           AND bedrooms=? AND area_sqft=? AND budget_bdt=?""",
        (username, house.get("District"), house.get("Location"),
         house.get("Bedrooms"), house.get("Area_sqft"), house.get("Budget_BDT")),
    ).fetchone()
    if exists:
        conn.close()
        return False  # already saved
    conn.execute(
        """INSERT INTO favorites
           (username, district, location, house_size, bedrooms, bathrooms,
            area_sqft, budget_bdt, match_score, saved_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (username, house.get("District"), house.get("Location"), house.get("House_Size"),
         house.get("Bedrooms"), house.get("Bathrooms"), house.get("Area_sqft"),
         house.get("Budget_BDT"), house.get("match_score"), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True


def get_favorites(username):
    conn = _connect()
    rows = conn.execute(
        """SELECT id, district, location, house_size, bedrooms, bathrooms,
                  area_sqft, budget_bdt, match_score
           FROM favorites WHERE username=? ORDER BY saved_at DESC""",
        (username,),
    ).fetchall()
    conn.close()
    cols = ["id", "District", "Location", "House_Size", "Bedrooms",
            "Bathrooms", "Area_sqft", "Budget_BDT", "match_score"]
    return [dict(zip(cols, r)) for r in rows]


def remove_favorite(username, fav_id):
    conn = _connect()
    conn.execute("DELETE FROM favorites WHERE id=? AND username=?", (fav_id, username))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------

def add_search(username, prefs):
    conn = _connect()
    conn.execute(
        "INSERT INTO search_history (username, prefs_json, searched_at) VALUES (?,?,?)",
        (username, json.dumps(prefs), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_searches(username, limit=10):
    conn = _connect()
    rows = conn.execute(
        "SELECT id, prefs_json, searched_at FROM search_history "
        "WHERE username=? ORDER BY searched_at DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    conn.close()
    return [{"id": r[0], "prefs": json.loads(r[1]), "searched_at": r[2]} for r in rows]


def clear_searches(username):
    conn = _connect()
    conn.execute("DELETE FROM search_history WHERE username=?", (username,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Collaborative + personalization signals
# ---------------------------------------------------------------------------

def popularity_signatures():
    """Return {(district_lower, location_lower): save_count} across all users.
    Used by the recommender to give a small popularity nudge."""
    conn = _connect()
    rows = conn.execute(
        "SELECT LOWER(district), LOWER(location), COUNT(*) "
        "FROM favorites GROUP BY LOWER(district), LOWER(location)"
    ).fetchall()
    conn.close()
    return {(d or "", l or ""): c for d, l, c in rows if d}


def learned_weights(username, base_weights):
    """Personalize weights from this user's favorites.

    Idea: criteria with LOW variance across the user's saved houses are
    things they consistently care about — bump those weights up by up to 50%.
    Needs at least 3 favorites; otherwise returns base_weights unchanged.
    """
    favs = get_favorites(username)
    if len(favs) < 3:
        return dict(base_weights)

    df = pd.DataFrame(favs)
    w = dict(base_weights)

    numeric_map = {
        "budget": "Budget_BDT",
        "area": "Area_sqft",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
    }
    for key, col in numeric_map.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 2 or vals.mean() == 0:
            continue
        cv = float(vals.std() / vals.mean())  # coefficient of variation
        # cv near 0 => consistent => stronger preference => boost.
        consistency = max(0.0, 1.0 - min(cv, 1.0))
        w[key] = base_weights.get(key, 0) * (1 + 0.5 * consistency)

    if "House_Size" in df.columns and not df["House_Size"].isna().all():
        top_share = float(df["House_Size"].value_counts(normalize=True).iloc[0])
        boost = max(0.0, (top_share - 0.5) / 0.5)  # 0 if no clear preference
        w["house_size"] = base_weights.get("house_size", 0) * (1 + 0.5 * boost)

    return w


if __name__ == "__main__":
    init_user_db()
    print("User database initialised at", USERS_DB)
