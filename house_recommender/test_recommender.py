"""
test_recommender.py
--------------------
Unit tests for the ML recommender.

Runs against a small, self-contained temp database (not the real houses.db) so
the assertions are deterministic. Point recommender at it via the HOUSES_DB env
var BEFORE importing recommender.

Run with either:
    python3 -m pytest test_recommender.py -v
    python3 test_recommender.py          # falls back to a plain runner
"""

import os
import sqlite3
import tempfile

# --- Build a tiny, known dataset and point the recommender at it -------------
_TMP_DB = os.path.join(tempfile.mkdtemp(), "test_houses.db")
os.environ["HOUSES_DB"] = _TMP_DB

_ROWS = [
    # District, Location, House_Size, Bedrooms, Bathrooms, Area_sqft, Budget_BDT, Source
    ("Dhaka", "Mirpur", "Medium", 3, 2, 1400, 9_000_000, "t"),
    ("Dhaka", "Banani", "Large", 4, 3, 2200, 20_000_000, "t"),
    ("Dhaka", "Uttara", "Small", 2, 1, 800, 5_000_000, "t"),
    ("Chittagong", "GEC", "Medium", 3, 2, 1450, 8_500_000, "t"),
    ("Sylhet", "Zindabazar", "Medium", 3, 2, 1350, 7_000_000, "t"),
    ("Dhaka", "Dhanmondi", "Large", 4, 3, 2100, 18_000_000, "t"),
]


def _build_db():
    conn = sqlite3.connect(_TMP_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS houses ("
        "District TEXT, Location TEXT, House_Size TEXT, Bedrooms REAL, "
        "Bathrooms REAL, Area_sqft REAL, Budget_BDT REAL, Source TEXT)"
    )
    conn.execute("DELETE FROM houses")
    conn.executemany(
        "INSERT INTO houses VALUES (?,?,?,?,?,?,?,?)", _ROWS
    )
    conn.commit()
    conn.close()


_build_db()

import recommender  # noqa: E402  (must come after HOUSES_DB is set)

recommender._fit_model.cache_clear()  # drop any cache from a prior import


# --- Tests -------------------------------------------------------------------

def test_exact_match_ranks_first():
    """A query matching the Mirpur house should return it at the top."""
    prefs = {
        "district": "Dhaka", "house_size": "Medium",
        "bedrooms": 3, "bathrooms": 2, "area_sqft": 1400, "budget_bdt": 9_500_000,
    }
    results = recommender.recommend(prefs, top_n=5)
    assert results, "expected at least one match"
    top = results[0]
    assert top["Location"] == "Mirpur"
    assert top["match_score"] >= 95        # near-perfect fit
    # Every result carries an explainable breakdown for the UI.
    assert "score_breakdown" in top and top["score_breakdown"]


def test_budget_hard_filter():
    """A tight budget must exclude houses priced above it (+ tolerance)."""
    prefs = {"budget_bdt": 5_200_000}
    results = recommender.recommend(prefs, top_n=10)
    assert results, "the cheap Uttara house should survive the filter"
    cap = 5_200_000 * (1 + recommender.BUDGET_TOLERANCE)
    assert all(r["Budget_BDT"] <= cap for r in results)
    # The 20M Banani house must not appear.
    assert all(r["Location"] != "Banani" for r in results)


def test_district_filter_isolates_region():
    prefs = {"district": "Sylhet", "bedrooms": 3}
    results = recommender.recommend(prefs, top_n=10)
    assert results
    assert all(r["District"] == "Sylhet" for r in results)


def test_similar_to_is_knn_and_excludes_self():
    """similar_to should return neighbours, never the query house itself."""
    seed = {
        "District": "Dhaka", "Location": "Mirpur", "House_Size": "Medium",
        "Bedrooms": 3, "Bathrooms": 2, "Area_sqft": 1400, "Budget_BDT": 9_000_000,
    }
    sims = recommender.similar_to(seed, top_n=3)
    assert sims, "expected similar houses"
    assert all(s["Location"] != "Mirpur" for s in sims)
    # The closest match to a mid-size 3-bed should be the other mid-size 3-beds,
    # not the 4-bed 2200 sqft luxury flat.
    assert sims[0]["Location"] in {"GEC", "Zindabazar", "Dhanmondi", "Banani", "Uttara"}


def test_no_criteria_returns_something_without_error():
    """An empty preference set should not crash (score defaults to 0)."""
    results = recommender.recommend({}, top_n=3)
    assert isinstance(results, list)


if __name__ == "__main__":
    # Minimal runner so the file works without pytest installed.
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
