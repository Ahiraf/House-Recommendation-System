"""
main.py
-------
FastAPI backend for the House Recommendation System.

Run locally:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API explorer.

Endpoints:
    GET  /                 -> health check
    GET  /districts        -> list of available districts (for dropdowns)
    GET  /house-sizes      -> list of size categories
    POST /recommend        -> ranked house recommendations
"""

import os
import sqlite3
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommender import recommend, DB_PATH
from load_data import ensure_database

app = FastAPI(
    title="House Recommendation System",
    description="Recommends houses in Bangladesh based on buyer preferences.",
    version="1.0.0",
)

# Allow a separate frontend (e.g. Streamlit) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    """Build the SQLite DB on first run (or rebuild if empty/corrupt).
    This makes cloud deployment painless -- no manual setup step."""
    ensure_database()


class Preferences(BaseModel):
    district: Optional[str] = Field(None, example="Dhaka")
    location: Optional[str] = Field(None, example="Mirpur")
    house_size: Optional[str] = Field(None, example="Medium")
    bedrooms: Optional[int] = Field(None, example=3)
    bathrooms: Optional[int] = Field(None, example=2)
    area_sqft: Optional[float] = Field(None, example=1400)
    budget_bdt: Optional[float] = Field(None, example=9000000)
    top_n: int = Field(10, ge=1, le=50)


@app.get("/")
def root():
    return {"status": "ok", "message": "House Recommendation API is running."}


@app.get("/districts", response_model=List[str])
def districts():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT District FROM houses ORDER BY District"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


@app.get("/house-sizes", response_model=List[str])
def house_sizes():
    return ["Small", "Medium", "Large"]


@app.post("/recommend")
def recommend_houses(prefs: Preferences):
    data = prefs.dict()
    top_n = data.pop("top_n")
    results = recommend(data, top_n=top_n)
    return {"count": len(results), "results": results}
