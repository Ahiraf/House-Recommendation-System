# 🏠 House Recommendation System

A content-based recommendation system that suggests houses in Bangladesh based
on a buyer's preferences: **District, Location, House Size, Bedrooms, Bathrooms,
Area (sq ft), and Budget**.

Built with **Python + FastAPI + SQLite**, plus an optional **Streamlit** web UI.

---

## Features

* 🔎 **Recommendations** — ranked houses based on District, Location, Size, Rooms, Bathrooms, Area, Budget.
* ⭐ **Favorites / wishlist** — save houses and view them on a dedicated tab.
* 🕘 **Search history** — your past searches are remembered.
* 🔄 **Reset filters** — clear all inputs in one click.
* ⬇️ **CSV download** — export your recommended houses.
* The Streamlit **Deploy button is hidden** (see `.streamlit/config.toml`).

## How it works

1. `load_data.py` reads the merged CSV (`house_recommendation_dataset.csv`),
   maps it to a fixed schema, cleans it, and loads it into a SQLite database
   (`houses.db`).
2. `recommender.py` scores every candidate house against the user's preferences
   using transparent weighted **content-based filtering** (no training needed).
3. `main.py` exposes the recommender as a **FastAPI** REST API.
4. `app.py` is a **Streamlit** UI with dropdowns and sliders for non-technical users.

### Schema (one table: `houses`)

| Column      | Meaning                          |
|-------------|----------------------------------|
| District    | City/district                    |
| Location    | Area / neighborhood              |
| House_Size  | Small / Medium / Large           |
| Bedrooms    | Number of rooms                  |
| Bathrooms   | Number of bathrooms              |
| Area_sqft   | Floor area in square feet        |
| Budget_BDT  | Price in Bangladeshi Taka        |
| Source      | Which dataset the row came from  |

---

## Setup & run (local)

```bash
# 1. (optional) create a virtual environment
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. build the database from the CSV
python load_data.py

# 4a. run the Streamlit UI (easiest — single command, builds DB automatically)
streamlit run app.py

# 4b. OR run the API
uvicorn main:app --reload
#     then open http://127.0.0.1:8000/docs  (interactive API explorer)
```

### Example API request

`POST /recommend`
```json
{
  "district": "Dhaka",
  "house_size": "Medium",
  "bedrooms": 3,
  "bathrooms": 2,
  "area_sqft": 1300,
  "budget_bdt": 8000000,
  "top_n": 10
}
```
Returns houses ranked by a `match_score` from 0–100.

---

## ➕ Adding more datasets later

This is built to grow. To add a new CSV in the future you **only edit
`load_data.py`** — nothing else changes.

1. Drop your new CSV in the parent folder.
2. Add a block to the `SOURCES` list, mapping its columns to our schema:

```python
{
    "path": os.path.join(DATA_DIR, "sylhet_houses_2026.csv"),
    "source_name": "sylhet_2026",
    "mapping": {
        "District":   "district_name",   # their column -> our column
        "Location":   "area",
        "Bedrooms":   "beds",
        "Bathrooms":  "baths",
        "Area_sqft":  "size_sqft",
        "Budget_BDT": "price_taka",
        # House_Size omitted -> auto-derived from Area_sqft
    },
}
```

3. Re-run `python load_data.py`. Done — the API and UI pick up the new data
   automatically. Missing columns (e.g. bathrooms) are left blank; missing
   `House_Size` is derived from the area.

---

## ☁️ Deployment

The stack is intentionally easy to deploy because it's just Python + one DB file.

### Option A — Streamlit Community Cloud (easiest, free, for the UI)
1. Push this folder to a **GitHub** repo.
2. Go to <https://share.streamlit.io>, connect the repo, set the main file to
   `app.py`. It installs `requirements.txt` and runs automatically.

### Option B — Render / Railway / Fly.io (for the FastAPI API)
1. Push to GitHub.
2. Create a new **Web Service**, point it at the repo.
3. Build command: `pip install -r requirements.txt`
   Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   (a `Procfile` is already included for hosts that use it).

The API rebuilds `houses.db` on startup if it's missing, so no manual DB step
is needed on the server.

> **Note on storage:** free hosts often have *ephemeral* disks, so `houses.db`
> may reset on redeploy. That's fine here — the data is read-only and rebuilt
> from the CSV automatically. If you later need to *persist user-submitted*
> data, switch SQLite for a hosted **PostgreSQL** (Render/Railway offer it),
> changing only the connection code in `load_data.py` and `recommender.py`.

---

## Files

| File              | Purpose                                  |
|-------------------|------------------------------------------|
| `load_data.py`    | Flexible CSV → SQLite loader             |
| `recommender.py`  | Weighted scoring engine                  |
| `auth.py`         | Favorites and search history             |
| `main.py`         | FastAPI backend                          |
| `app.py`          | Streamlit web UI                         |
| `.streamlit/config.toml` | Hides Deploy button, sets theme   |
| `requirements.txt`| Dependencies                             |
| `Procfile`        | Start command for cloud hosts            |
