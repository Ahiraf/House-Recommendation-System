"""
app.py
------
Streamlit web UI for the House Recommendation System.

Features:
  * Recommendations with per-criterion "Why this match?" breakdowns
  * Map view of recommendations (district centroids + jitter)
  * Charts tab (price distribution, area vs price)
  * Side-by-side compare for 2-3 picked houses
  * Favorites with "Similar houses" suggestions
  * Personalized weights learned from a user's favorites
  * Collaborative popularity boost (signal from all users' favorites)
  * Search history, CSV download
"""

import os
import sqlite3

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Folium gives a fully draggable (pan + zoom) Leaflet map. Fall back to the
# built-in st.map if it isn't installed so the app still runs everywhere.
try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

from recommender import recommend, similar_to, DB_PATH, DEFAULT_WEIGHTS
from load_data import ensure_database
import auth

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ensure_database()
auth.init_user_db()

st.set_page_config(
    page_title="House Recommendation System",
    page_icon="🏠",
    layout="wide",
    # Open the "Your Preferences" sidebar by default (incl. on mobile, where
    # Streamlit otherwise hides it behind a small arrow users often miss).
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
      /* Hide only the deploy button + footer. We deliberately do NOT hide the
         whole toolbar/header, because the sidebar's reopen ("»") arrow lives
         there — hiding the toolbar makes it impossible to reopen "Your
         Preferences" once collapsed. */
      .stDeployButton {display: none;}
      [data-testid="stAppDeployButton"] {display: none;}
      footer {visibility: hidden;}

      /* Belt-and-suspenders: force the sidebar expand/collapse controls to stay
         visible across Streamlit versions (testid names have changed over time). */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stExpandSidebarButton"],
      [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
      }

      /* ---- Mobile / small screens ---- */
      @media (max-width: 640px) {
        /* Trim big desktop margins so content uses the full width. */
        .block-container {padding: 1rem 0.75rem 3rem 0.75rem !important;}

        /* Let side-by-side columns wrap and stack instead of getting squished
           (house cards, Find/Reset buttons, compare tables, etc.). */
        [data-testid="stHorizontalBlock"] {flex-wrap: wrap !important;}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
          flex: 1 1 100% !important;
          min-width: 100% !important;
        }

        /* Full-width sidebar so the preferences form is easy to use. */
        [data-testid="stSidebar"] {min-width: 85vw !important; width: 85vw !important;}

        /* Tabs (Search / My Favorites / Search History / Charts): on a narrow
           screen the row overflows and some tabs get cut off. Let them wrap to
           multiple lines and shrink spacing so all four stay visible/tappable. */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          flex-wrap: wrap !important;
          gap: 0.15rem !important;
          overflow-x: visible !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          padding: 0.25rem 0.5rem !important;
          font-size: 0.8rem !important;
          white-space: nowrap !important;
        }

        /* Slightly smaller headings so titles don't wrap awkwardly. */
        h1 {font-size: 1.5rem !important;}
        h2 {font-size: 1.2rem !important;}
      }
    </style>
""", unsafe_allow_html=True)


# Bangladesh district centroids (lat, lon). Houses without a known district
# are silently skipped on the map.
DISTRICT_LATLON = {
    "dhaka": (23.8103, 90.4125),
    "chattogram": (22.3569, 91.7832),
    "chittagong": (22.3569, 91.7832),
    "sylhet": (24.8949, 91.8687),
    "rajshahi": (24.3745, 88.6042),
    "khulna": (22.8456, 89.5403),
    "barishal": (22.7010, 90.3535),
    "barisal": (22.7010, 90.3535),
    "rangpur": (25.7439, 89.2752),
    "mymensingh": (24.7471, 90.4203),
    "cumilla": (23.4607, 91.1809),
    "comilla": (23.4607, 91.1809),
    "narayanganj": (23.6238, 90.5000),
    "gazipur": (24.0023, 90.4264),
    "jessore": (23.1685, 89.2072),
    "bogura": (24.8465, 89.3776),
    "cox's bazar": (21.4272, 92.0058),
    "tangail": (24.2513, 89.9167),
    "pabna": (24.0064, 89.2372),
    "dinajpur": (25.6217, 88.6354),
    "faridpur": (23.6070, 89.8429),
    "noakhali": (22.8696, 91.0995),
}


@st.cache_data
def get_districts():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT DISTINCT District FROM houses ORDER BY District").fetchall()
    conn.close()
    return [r[0] for r in rows]


@st.cache_data
def load_all_houses():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    if {"Area_sqft", "Budget_BDT"}.issubset(df.columns):
        df["Price_per_sqft"] = (df["Budget_BDT"] / df["Area_sqft"]).round(0)
    return df


def add_latlon(records):
    """Attach jittered lat/lon to a list of house dicts."""
    rng = np.random.default_rng(42)
    out = []
    for r in records:
        d = str(r.get("District") or "").lower().strip()
        latlon = DISTRICT_LATLON.get(d)
        nr = dict(r)
        if latlon:
            nr["lat"] = latlon[0] + rng.uniform(-0.03, 0.03)
            nr["lon"] = latlon[1] + rng.uniform(-0.03, 0.03)
        else:
            nr["lat"] = None
            nr["lon"] = None
        out.append(nr)
    return out


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# Single local profile used for favorites / history (no accounts).
USER = "guest"

DEFAULTS = {
    "results": [],
    "searched": False,
    "compare_ids": set(),
    "f_district": "Any",
    "f_location": "",
    "f_size": "Any",
    "f_bedrooms": 3,
    "f_bathrooms": 2,
    "f_area": 1400,
    "f_budget": 9000000,
    "f_max_pps": 0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def reset_filters():
    st.session_state.f_district = "Any"
    st.session_state.f_location = ""
    st.session_state.f_size = "Any"
    st.session_state.f_bedrooms = 3
    st.session_state.f_bathrooms = 2
    st.session_state.f_area = 1400
    st.session_state.f_budget = 9000000
    st.session_state.f_max_pps = 0
    st.session_state.results = []
    st.session_state.searched = False
    st.session_state.compare_ids = set()


# ===========================================================================
# Renderers
# ===========================================================================
def house_label(h):
    bd = int(h["Bedrooms"]) if pd.notna(h.get("Bedrooms")) else "?"
    ba = int(h["Bathrooms"]) if pd.notna(h.get("Bathrooms")) else "?"
    return (f"**{h.get('Location') or '—'}**, {h.get('District')} — "
            f"{bd} bd / {ba} ba · {int(h['Area_sqft'])} sqft · "
            f"{h['Budget_BDT']:,.0f} BDT")


def render_breakdown(b):
    """Render the 'why this match' breakdown as labelled progress bars."""
    if not b:
        st.caption("No criteria were specified — score is 0.")
        return
    labels = {
        "budget": "💰 Budget fit",
        "area": "📐 Area fit",
        "bedrooms": "🛏️ Bedrooms",
        "bathrooms": "🛁 Bathrooms",
        "house_size": "🏠 Size category",
        "location": "📍 Location",
    }
    for key, pct in b.items():
        st.write(f"{labels.get(key, key)} — **{pct:.0f}%**")
        st.progress(min(max(pct / 100.0, 0.0), 1.0))


def render_map(records):
    df = pd.DataFrame(add_latlon(records))
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        st.info("No map coordinates available for these districts.")
        return

    if not _HAS_FOLIUM:
        # Fallback: still zoomable/pannable, but folium is smoother.
        st.map(df[["lat", "lon"]], zoom=6)
        return

    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(
        location=center,
        zoom_start=6,
        tiles="OpenStreetMap",
        dragging=True,          # click-and-drag to pan
        scrollWheelZoom=True,   # wheel to zoom
        control_scale=True,
    )
    for _, row in df.iterrows():
        price = row.get("Budget_BDT")
        district = row.get("District", "")
        popup = f"{district}"
        if pd.notna(price):
            popup += f" — {float(price):,.0f} BDT"
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=popup,
            tooltip=district or None,
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(fmap)
    st_folium(fmap, use_container_width=True, height=480, returned_objects=[])


def render_compare(records, compare_ids):
    picked = [r for r in records if r.get("_uid") in compare_ids]
    if len(picked) < 2:
        return
    st.subheader(f"⚖️ Compare ({len(picked)})")
    rows = {
        "District": [p["District"] for p in picked],
        "Location": [p["Location"] for p in picked],
        "Size": [p.get("House_Size", "") for p in picked],
        "Bedrooms": [p["Bedrooms"] for p in picked],
        "Bathrooms": [p["Bathrooms"] for p in picked],
        "Area (sqft)": [int(p["Area_sqft"]) for p in picked],
        "Price (BDT)": [f"{p['Budget_BDT']:,.0f}" for p in picked],
        "Price/sqft": [f"{p.get('Price_per_sqft', 0):,.0f}" for p in picked],
        "Match %": [f"{p['match_score']:.1f}" for p in picked],
    }
    cmp_df = pd.DataFrame(rows, index=[f"#{i+1}" for i in range(len(picked))]).T
    st.dataframe(cmp_df, use_container_width=True)


# ===========================================================================
# MAIN APP
# ===========================================================================
def main_app():
    user = USER

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Your Preferences")
        st.selectbox("District", ["Any"] + get_districts(), key="f_district")
        st.text_input("Location / Area (optional)", key="f_location",
                      placeholder="e.g. Mirpur, Banani")
        st.selectbox("House Size", ["Any", "Small", "Medium", "Large"], key="f_size")
        st.slider("Number of Rooms (bedrooms)", 0, 10, key="f_bedrooms")
        st.slider("Number of Bathrooms", 0, 10, key="f_bathrooms")
        st.number_input("Desired Area (sq ft)", min_value=0, step=50, key="f_area")
        st.number_input("Budget (BDT)", min_value=0, step=500000, key="f_budget")
        st.number_input("Max Price / sqft (BDT, 0 = no limit)",
                        min_value=0, step=500, key="f_max_pps")
        top_n = st.slider("How many results?", 5, 30, 10)

        use_smart = st.checkbox(
            "Use personalized weights (learned from my favorites)",
            value=True,
            help="Needs at least 3 saved favorites; otherwise defaults are used.",
        )
        use_popularity = st.checkbox(
            "Boost houses popular with other users", value=True,
        )

        c1, c2 = st.columns(2)
        search = c1.button("Find Houses", type="primary", use_container_width=True)
        c2.button("Reset", use_container_width=True, on_click=reset_filters)

    def current_prefs():
        return {
            "district": None if st.session_state.f_district == "Any" else st.session_state.f_district,
            "location": st.session_state.f_location or None,
            "house_size": None if st.session_state.f_size == "Any" else st.session_state.f_size,
            "bedrooms": st.session_state.f_bedrooms or None,
            "bathrooms": st.session_state.f_bathrooms or None,
            "area_sqft": st.session_state.f_area or None,
            "budget_bdt": st.session_state.f_budget or None,
            "max_price_per_sqft": st.session_state.f_max_pps or None,
        }

    tab_search, tab_favs, tab_history, tab_charts = st.tabs(
        ["🔎 Search", "⭐ My Favorites", "🕘 Search History", "📊 Charts"]
    )

    # -------- SEARCH TAB --------
    with tab_search:
        st.title("🔎 Find Your Home")
        st.caption("Set your preferences in the sidebar, then click **Find Houses**.")

        if search:
            prefs = current_prefs()
            weights = auth.learned_weights(user, DEFAULT_WEIGHTS) if use_smart else None
            popularity = auth.popularity_signatures() if use_popularity else None
            results = recommend(prefs, top_n=top_n, weights=weights, popularity=popularity)
            # tag every result with a stable unique id for compare-checkbox state
            for i, r in enumerate(results):
                r["_uid"] = i
            st.session_state.results = results
            st.session_state.searched = True
            st.session_state.compare_ids = set()
            auth.add_search(user, prefs)

        results = st.session_state.results
        if not results:
            if st.session_state.searched:
                st.warning(
                    "😕 **No houses matched your filters.** Try relaxing them — "
                    "for example set **Max Price / sqft** to `0` (no limit), "
                    "raise your **Budget**, change **House Size** to *Any*, or "
                    "clear the **Location** field."
                )
            else:
                st.info("No results yet. Use the sidebar to search.")
        else:
            st.success(f"Found {len(results)} matching houses (ranked by best fit).")

            # Summary table
            df = pd.DataFrame(results)
            show = df.rename(columns={
                "match_score": "Match %", "Budget_BDT": "Price (BDT)",
                "Area_sqft": "Area (sqft)", "Price_per_sqft": "Price/sqft",
            })[["Match %", "District", "Location", "House_Size",
                "Bedrooms", "Bathrooms", "Area (sqft)", "Price (BDT)", "Price/sqft"]].copy()
            show["Price (BDT)"] = show["Price (BDT)"].map(lambda x: f"{x:,.0f}")
            show["Price/sqft"] = show["Price/sqft"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download results (CSV)",
                df.to_csv(index=False).encode(),
                file_name="my_recommendations.csv",
                mime="text/csv",
            )

            # Map
            with st.expander("🗺️ Map view", expanded=False):
                render_map(results)

            # Per-result actions: save, compare, why-this-match
            st.subheader("Results")
            for i, h in enumerate(results):
                with st.container(border=True):
                    col_info, col_match, col_save, col_cmp = st.columns([6, 1.2, 1, 1.3])
                    col_info.write(house_label(h))
                    col_info.caption(
                        f"{h.get('House_Size') or '—'} · "
                        f"price/sqft {h.get('Price_per_sqft', 0):,.0f} BDT"
                    )
                    col_match.metric("Match", f"{h['match_score']:.0f}%")

                    if col_save.button("⭐ Save", key=f"save_{i}"):
                        added = auth.add_favorite(user, h)
                        st.toast("Saved to favorites!" if added else "Already in favorites.")

                    checked = h["_uid"] in st.session_state.compare_ids
                    new_checked = col_cmp.checkbox(
                        "Compare", value=checked, key=f"cmp_{i}",
                        disabled=(not checked and len(st.session_state.compare_ids) >= 3),
                    )
                    if new_checked and not checked:
                        st.session_state.compare_ids.add(h["_uid"])
                    elif not new_checked and checked:
                        st.session_state.compare_ids.discard(h["_uid"])

                    with st.expander("Why this match?"):
                        render_breakdown(h.get("score_breakdown") or {})

            render_compare(results, st.session_state.compare_ids)

    # -------- FAVORITES TAB --------
    with tab_favs:
        st.title("⭐ My Favorites")
        favs = auth.get_favorites(user)
        if not favs:
            st.info("You haven't saved any houses yet. Save some from the Search tab.")
        else:
            for h in favs:
                with st.container(border=True):
                    col1, col2 = st.columns([5, 1])
                    col1.write(house_label(h))
                    if col2.button("🗑 Remove", key=f"rm_{h['id']}"):
                        auth.remove_favorite(user, h["id"])
                        st.rerun()

                    with st.expander("🔁 Similar houses"):
                        sims = similar_to(h, top_n=5)
                        if not sims:
                            st.caption("No similar houses found.")
                        else:
                            for s in sims:
                                st.write(f"• {house_label(s)} — **{s['match_score']:.0f}%** match")

    # -------- HISTORY TAB --------
    with tab_history:
        st.title("🕘 Search History")
        hist = auth.get_searches(user, limit=15)
        if not hist:
            st.info("No searches yet.")
        else:
            if st.button("Clear history"):
                auth.clear_searches(user)
                st.rerun()
            for s in hist:
                p = {k: v for k, v in s["prefs"].items() if v}
                st.write(f"🕒 {s['searched_at'][:19].replace('T', ' ')} — " +
                         (", ".join(f"{k}: {v}" for k, v in p.items()) or "no filters"))

    # -------- CHARTS TAB --------
    with tab_charts:
        st.title("📊 Market Snapshot")
        st.caption("How prices, areas and listings are distributed in the dataset.")
        all_df = load_all_houses()
        if all_df.empty:
            st.info("No data loaded.")
            return

        # 1) Median price by district
        by_d = (all_df.groupby("District")["Budget_BDT"]
                .median().sort_values(ascending=False).head(15).reset_index())
        st.subheader("Median price by district (top 15)")
        st.altair_chart(
            alt.Chart(by_d).mark_bar().encode(
                x=alt.X("Budget_BDT:Q", title="Median price (BDT)"),
                y=alt.Y("District:N", sort="-x"),
                tooltip=["District", alt.Tooltip("Budget_BDT:Q", format=",")],
            ).properties(height=400),
            use_container_width=True,
        )

        # 2) Area vs price scatter — colour = district, marker for user's budget
        st.subheader("Area vs price")
        sample = all_df.sample(min(len(all_df), 1500), random_state=0)
        scatter = alt.Chart(sample).mark_circle(opacity=0.4).encode(
            x=alt.X("Area_sqft:Q", title="Area (sqft)"),
            y=alt.Y("Budget_BDT:Q", title="Price (BDT)"),
            color=alt.Color("District:N", legend=None),
            tooltip=["District", "Location", "Area_sqft", "Budget_BDT"],
        ).properties(height=380)

        user_budget = st.session_state.f_budget
        layers = [scatter]
        if user_budget:
            ref = alt.Chart(pd.DataFrame({"y": [user_budget]})).mark_rule(
                color="red", strokeDash=[6, 4]
            ).encode(y="y:Q")
            layers.append(ref)
        st.altair_chart(alt.layer(*layers), use_container_width=True)
        if user_budget:
            st.caption(f"🔴 dashed line = your budget ({user_budget:,.0f} BDT)")

        # 3) Price-per-sqft histogram
        st.subheader("Price-per-sqft distribution")
        st.altair_chart(
            alt.Chart(all_df).mark_bar().encode(
                x=alt.X("Price_per_sqft:Q", bin=alt.Bin(maxbins=40), title="Price / sqft"),
                y="count()",
            ).properties(height=280),
            use_container_width=True,
        )


# ===========================================================================
# Route
# ===========================================================================
main_app()
