# 🏠 House Recommendation System

A Streamlit web app that recommends houses in Bangladesh based on buyer
preferences (district, location, size, rooms, area, budget), with favorites,
search history, a map view, and market charts.

The app builds its SQLite database automatically from
`house_recommendation_dataset.csv` on first run — no manual setup needed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run house_recommender/app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and click **New app**.
3. Pick this repo and branch, and set **Main file path** to:
   `house_recommender/app.py`
4. Click **Deploy**. The app installs `requirements.txt` and starts.

> **Note:** free hosts use ephemeral disks, so favorites/search history
> (stored in `users.db`) may reset on redeploy. The house data itself is
> read-only and rebuilt from the CSV automatically, so search always works.

## Project layout

| Path | Purpose |
|------|---------|
| `house_recommender/app.py` | Streamlit web UI (the deployed app) |
| `house_recommender/recommender.py` | Weighted scoring engine |
| `house_recommender/load_data.py` | CSV → SQLite loader |
| `house_recommender/auth.py` | Favorites and search history |
| `house_recommender/main.py` | Optional FastAPI backend (not used by Streamlit deploy) |
| `house_recommendation_dataset.csv` | Source dataset |
