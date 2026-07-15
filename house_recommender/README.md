# House Recommendation System — app package

This folder holds the application code. For the full project overview, features,
ML/KNN explanation, screenshots, and deployment guide, see the
[main README](../README.md).

## Quick run

```bash
pip install -r requirements.txt          # from the repo root
streamlit run app.py                     # from this folder
```

The SQLite database is built automatically from the CSV on first run.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI (the deployed app) |
| `recommender.py` | ML engine: feature scaling + KNN + scoring |
| `nlp_search.py` | OpenAI natural-language → filters |
| `load_data.py` | CSV → SQLite loader |
| `auth.py` | Favorites & search history |
| `main.py` | Optional FastAPI backend |
| `test_recommender.py` | Unit tests for the ML engine |

## AI Search key

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your
`OPENAI_API_KEY`. The file is gitignored, so the key is never committed.
