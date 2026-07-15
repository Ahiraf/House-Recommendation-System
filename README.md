<div align="center">

# 🏠 House Recommendation System

**Find your ideal home in Bangladesh — powered by Machine Learning and AI.**

Set your preferences (or just describe your dream home in plain English) and get
houses ranked by best fit, each with a *"Why this match?"* breakdown, an
interactive map, market charts, favorites, and AI-powered similar-house
suggestions.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

</div>

---

## 🎥 Demo

<!-- Add your demo video link or embedded video here. -->
<!-- Example: [![Watch the demo](thumbnail.png)](https://your-video-link) -->

_A demo video will be added here soon._

<!-- Add a screenshot of the app here, e.g. -->
<!-- ![App preview](docs/preview.png) -->

---

## ✨ Features

![Smart Recommendations](https://img.shields.io/badge/Smart_Recommendations-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) &nbsp; Content-based ML ranking across district, location, size, rooms, area, and budget — with a per-criterion *"Why this match?"* breakdown.

![AI Search](https://img.shields.io/badge/AI_Search-412991?style=flat-square&logo=openai&logoColor=white) &nbsp; Describe your ideal home in plain English (*"3-bed flat in Dhaka under 90 lakh"*) and let OpenAI turn it into filters.

![Similar Houses](https://img.shields.io/badge/Similar_Houses_(KNN)-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) &nbsp; A k-Nearest-Neighbours model finds houses most similar to any you save.

![Interactive Map](https://img.shields.io/badge/Interactive_Map-199900?style=flat-square&logo=leaflet&logoColor=white) &nbsp; Draggable, zoomable Leaflet map showing recommended houses as markers.

![Market Charts](https://img.shields.io/badge/Market_Charts-4C78A8?style=flat-square) &nbsp; Price distribution, median price by district, and area-vs-price views.

![Compare](https://img.shields.io/badge/Compare-546E7A?style=flat-square) &nbsp; Side-by-side comparison of 2–3 picked houses.

![Favorites & History](https://img.shields.io/badge/Favorites_%26_History-003B57?style=flat-square&logo=sqlite&logoColor=white) &nbsp; Save listings and revisit past searches.

![CSV Export](https://img.shields.io/badge/CSV_Export-217346?style=flat-square&logo=microsoftexcel&logoColor=white) &nbsp; Download your recommendations with one click.

---

## 🔄 How It Works

```
Your preferences / plain-English query
                 │
                 ▼
   Filters (district, budget, size, rooms, area…)
                 │
                 ▼
   ML engine  →  every house scaled to a 0–1 feature vector
                 │
                 ├─ recommend():   weighted similarity  →  ranked results + "why"
                 └─ similar_to():  k-Nearest-Neighbours →  similar houses
                 │
                 ▼
   Results · Map · Charts · Compare · Favorites
```

---

## 🧠 How the Recommendations Work (Machine Learning)

Every house is turned into a **5-number feature vector** — `budget`, `area`,
`bedrooms`, `bathrooms`, and `size` — all normalised to a `0–1` range with
scikit-learn's **`MinMaxScaler`** (the min/max are *learned* from the dataset, so
a price gap and a bedroom gap become comparable).

- **Ranking (`recommend`)** — for each preference you set, a per-feature
  similarity `1 − |your_value − house_value|` is computed and combined by weight
  (budget counts more than bathrooms). This produces the match % **and** the
  *"Why this match?"* bars.
- **Similar houses (`similar_to`)** — a **k-Nearest-Neighbours** model
  (`NearestNeighbors`, cosine distance) finds the houses whose feature vectors
  are *closest* to one you saved. Closer vector → higher match %.

> No training labels are required — it's content-based ML, so it works with the
> house data alone.

---

## 🧰 Tech Stack

| Layer            | Technology                                  |
| ---------------- | ------------------------------------------- |
| Web UI           | Streamlit                                   |
| Language         | Python 3.11                                 |
| Machine Learning | scikit-learn (MinMaxScaler + NearestNeighbors / KNN) |
| AI search        | OpenAI (`gpt-4o-mini`, structured outputs)  |
| Data handling    | pandas · NumPy                              |
| Database         | SQLite                                      |
| Maps             | Folium (Leaflet)                            |
| Charts           | Altair                                      |
| Fuzzy matching   | RapidFuzz                                   |
| Optional API     | FastAPI + Uvicorn                           |

---

## 🏗 Architecture

```
        house_recommendation_dataset.csv
                     │
                     ▼
             load_data.py  ──►  houses.db (SQLite)
                                     │
   ┌──────────────── app.py (Streamlit UI) ────────────────┐
   │              │                  │                      │
   ▼              ▼                  ▼                      ▼
recommender.py   nlp_search.py     auth.py           Folium · Altair
(ML: scaling +   (OpenAI: text     (favorites &       (map + charts)
 KNN + scoring)   → filters)        history · users.db)
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- (Optional) an OpenAI API key — only needed for the **AI Search** box

### Install & run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the app (it builds the SQLite database automatically on first run)
streamlit run house_recommender/app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

> The ML recommender, map, charts, favorites, and compare all work **without** an
> API key. Only the 🔮 AI Search box needs one.

---

## 🔑 Configuration (AI Search)

The AI Search box reads your OpenAI key from Streamlit secrets.

1. Copy the template:
   ```bash
   cp house_recommender/.streamlit/secrets.toml.example house_recommender/.streamlit/secrets.toml
   ```
2. Paste your real key into `secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-your-real-key"
   ```

`secrets.toml` is gitignored, so your key is never committed. On **Streamlit
Community Cloud**, add the same key under **App → Settings → Secrets** instead.

---

## 📁 Project Structure

```
House-Recommendation-System
├── house_recommender
│   ├── app.py                 # Streamlit UI (the deployed app)
│   ├── recommender.py         # ML engine: feature scaling + KNN + scoring
│   ├── nlp_search.py          # OpenAI natural-language → filters
│   ├── load_data.py           # CSV → SQLite loader
│   ├── auth.py                # favorites & search history
│   ├── main.py                # optional FastAPI backend
│   ├── test_recommender.py    # unit tests for the ML engine
│   ├── requirements.txt
│   ├── Procfile
│   └── .streamlit
│       ├── config.toml
│       └── secrets.toml.example
├── house_recommendation_dataset.csv   # source dataset
├── requirements.txt
└── README.md
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> → **New app**.
3. Pick this repo/branch and set **Main file path** to `house_recommender/app.py`.
4. Add your `OPENAI_API_KEY` under **Settings → Secrets**, then **Deploy**.

---

## 🧪 Tests

```bash
cd house_recommender
python3 test_recommender.py         # or: python3 -m pytest test_recommender.py -v
```

---

## 🗺 Roadmap

- Real user accounts (per-user favorites & history)
- Collaborative filtering ("users like you also liked…")
- Real geocoding for accurate map pins
- AI-written explanations per recommended house
- Persistent hosted database (PostgreSQL)

---

## 📄 License

Released under the MIT License.

---

<div align="center">
Built to make finding a home simpler, smarter, and more interactive.
</div>
