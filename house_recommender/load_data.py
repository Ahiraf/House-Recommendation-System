"""
load_data.py
------------
Flexible CSV -> SQLite loader for the House Recommendation System.

The database always uses this fixed schema (one table: `houses`):
    District, Location, House_Size, Bedrooms, Bathrooms, Area_sqft, Budget_BDT, Source


To add a NEW dataset in the future no need to  touch any other file.
only:
    1. Add an entry to the SOURCES list below (or call load_csv()),
       mapping the new file's column names to our schema names.
    2. Re-run:  python load_data.py
That's it -- the API and recommender keep working unchanged.

House_Size is derived automatically from Area_sqft if you don't provide it.
"""

import os
import re
import sqlite3
import tempfile
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(HERE)          # parent "Recommendation System" folder
# DB location can be overridden with the HOUSES_DB env var. Default to the
# system temp dir, which is always writable — the repo mount on cloud hosts
# (e.g. Streamlit Cloud) can be read-only, which breaks writing the SQLite file.
DB_PATH = os.environ.get("HOUSES_DB", os.path.join(tempfile.gettempdir(), "houses.db"))

# Canonical schema columns (order matters for the table)
SCHEMA = ["District", "Location", "House_Size", "Bedrooms",
          "Bathrooms", "Area_sqft", "Budget_BDT", "Source"]

# The already-merged dataset we built. Because its columns ALREADY match the
# schema, its mapping is 1-to-1.
SOURCES = [
    {
        "path": os.path.join(DATA_DIR, "house_recommendation_dataset.csv"),
        "source_name": "merged_dataset",
        "mapping": {
            "District": "District",
            "Location": "Location",
            "House_Size": "House_Size",
            "Bedrooms": "Bedrooms",
            "Bathrooms": "Bathrooms",
            "Area_sqft": "Area_sqft",
            "Budget_BDT": "Budget_BDT",
        },
    },
    # ---- EXAMPLE: how to add a future dataset -----------------------------
    # {
    #     "path": os.path.join(DATA_DIR, "sylhet_houses_2026.csv"),
    #     "source_name": "sylhet_2026",
    #     "mapping": {
    #         "District":   "district_name",   # <- their column -> our column
    #         "Location":   "area",
    #         "Bedrooms":   "beds",
    #         "Bathrooms":  "baths",
    #         "Area_sqft":  "size_sqft",
    #         "Budget_BDT": "price_taka",
    #         # House_Size omitted -> auto-derived from Area_sqft
    #     },
    # },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_number(value):
    """Strip currency symbols / commas / Bengali text and return a float."""
    if pd.isna(value):
        return np.nan
    s = re.sub(r"[^0-9.]", "", str(value))
    try:
        return float(s) if s else np.nan
    except ValueError:
        return np.nan


def _size_category(area):
    if pd.isna(area):
        return ""
    if area < 1000:
        return "Small"
    if area <= 1800:
        return "Medium"
    return "Large"


def load_csv(path, mapping, source_name):
    """Read one CSV, remap its columns to the canonical schema, clean it."""
    if not os.path.exists(path):
        print(f"  ! skipped (not found): {path}")
        return pd.DataFrame(columns=SCHEMA)

    raw = pd.read_csv(path)
    out = pd.DataFrame()

    for schema_col, src_col in mapping.items():
        if src_col in raw.columns:
            out[schema_col] = raw[src_col]
        else:
            out[schema_col] = np.nan

    # Ensure every schema column exists
    for col in SCHEMA:
        if col not in out.columns:
            out[col] = np.nan

    # Clean numeric fields
    out["Area_sqft"] = out["Area_sqft"].apply(_clean_number)
    out["Budget_BDT"] = out["Budget_BDT"].apply(_clean_number)
    out["Bedrooms"] = pd.to_numeric(out["Bedrooms"], errors="coerce")
    out["Bathrooms"] = pd.to_numeric(out["Bathrooms"], errors="coerce")

    # Derive House_Size when missing/blank
    needs_size = out["House_Size"].isna() | (out["House_Size"].astype(str).str.strip() == "")
    out.loc[needs_size, "House_Size"] = out.loc[needs_size, "Area_sqft"].apply(_size_category)

    out["Source"] = source_name
    out["District"] = out["District"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.title()
    out["Location"] = out["Location"].astype(str).replace({"nan": "", "None": ""}).str.strip()

    # Drop rows that can't be recommended (no price or no area)
    out = out[out["Budget_BDT"].notna() & (out["Budget_BDT"] > 0)]
    out = out[out["Area_sqft"].notna() & (out["Area_sqft"] > 0)]

    # Outlier filter on price-per-sqft: keep the central 98% so a Tk-1
    # listing or a typo'd 30-crore micro-flat can't poison scoring.
    if len(out) > 20:
        pps = out["Budget_BDT"] / out["Area_sqft"]
        lo, hi = pps.quantile(0.01), pps.quantile(0.99)
        before = len(out)
        out = out[(pps >= lo) & (pps <= hi)]
        if before - len(out):
            print(f"  ~ {source_name}: dropped {before - len(out)} price-per-sqft outliers")

    print(f"  + {source_name}: {len(out)} rows from {os.path.basename(path)}")
    return out[SCHEMA]


def build_database():
    print("Building database...")
    frames = [load_csv(s["path"], s["mapping"], s["source_name"]) for s in SOURCES]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    df = df.drop_duplicates().reset_index(drop=True)

    # Tidy integer columns for storage
    for col in ["Bedrooms", "Bathrooms", "Area_sqft", "Budget_BDT"]:
        df[col] = df[col].round(0)

    # Remove any leftover/corrupt db file so we always start clean.
    for stale in (DB_PATH, DB_PATH + "-journal"):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("houses", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_district ON houses(District)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_budget ON houses(Budget_BDT)")
    conn.commit()
    conn.close()

    print(f"\nDone. {len(df)} houses written to {DB_PATH}")
    print("Districts:", ", ".join(f"{k} ({v})" for k, v in df['District'].value_counts().items()))
    return len(df)


def database_ready():
    """True only if the DB file exists AND contains a populated `houses` table.
    Guards against empty/corrupt .db files left behind by an interrupted run."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='houses'"
        )
        has_table = cur.fetchone() is not None
        count = conn.execute("SELECT COUNT(*) FROM houses").fetchone()[0] if has_table else 0
        conn.close()
        return has_table and count > 0
    except Exception:
        return False


def ensure_database():
    """Build the database only if it isn't already ready."""
    if not database_ready():
        build_database()


if __name__ == "__main__":
    build_database()
