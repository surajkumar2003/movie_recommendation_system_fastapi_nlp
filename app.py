import requests
import streamlit as st

# =============================
# CONFIG  or"http://127.0.0.1:8000"
# =============================
API_BASE ="https://movie-recommendation-system-fastapi-nlp.onrender.com" or "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# =============================
# STYLES (Dark Netflix-like)
# =============================
st.markdown(
    """
<style>
body { background-color: #0f1117; color: #e5e7eb; }
.block-container { padding-top: 1rem; max-width: 1400px; }
.movie-title {
    font-size: 0.9rem;
    line-height: 1.2rem;
    height: 2.4rem;
    overflow: hidden;
    text-align: center;
}
.card {
    background: #111827;
    border-radius: 14px;
    padding: 12px;
}
.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}
button[kind="secondary"] {
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# API HELPER
# =============================
@st.cache_data(ttl=60)
def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# =============================
# POSTER GRID
# =============================
def poster_grid(movies, cols=6, key_prefix="grid"):
    if not movies:
        st.info("No movies found.")
        return

    rows = (len(movies) + cols - 1) // cols
    idx = 0

    for r in range(rows):
        columns = st.columns(cols)
        for c in range(cols):
            if idx >= len(movies):
                break

            m = movies[idx]
            idx += 1

            with columns[c]:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                if m.get("poster_url"):
                    st.image(m["poster_url"], use_container_width=True)
                else:
                    st.write("🖼️ No poster")

                if st.button(
                    "Open",
                    key=f"{key_prefix}_{r}_{c}_{m['tmdb_id']}",
                ):
                    st.session_state.view = "details"
                    st.session_state.tmdb_id = m["tmdb_id"]
                    st.rerun()

                st.markdown(
                    f"<div class='movie-title'>{m['title']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

# =============================
# SESSION STATE
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"

if "tmdb_id" not in st.session_state:
    st.session_state.tmdb_id = None

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")

    if st.button("🏠 Home"):
        st.session_state.view = "home"
        st.session_state.tmdb_id = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 🏠 Home Feed")

    category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
    )

    grid_cols = st.slider("Grid columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.markdown("## 🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type keyword → suggestions → open → details + recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# HOME VIEW
# ==========================================================
if st.session_state.view == "home":

    query = st.text_input(
        "Search by movie title (keyword)",
        placeholder="Type: avenger, batman, love...",
    )

    st.divider()

    # SEARCH MODE
    if query.strip():
        data = api_get("/tmdb/search", {"query": query})

        if not data:
            st.error("Search failed.")
        else:
            results = data.get("results", [])
            cards = [
                {
                    "tmdb_id": m["id"],
                    "title": m.get("title", ""),
                    "poster_url": f"{TMDB_IMG}{m['poster_path']}"
                    if m.get("poster_path")
                    else None,
                }
                for m in results
            ]
            st.markdown("### Results")
            poster_grid(cards, cols=grid_cols, key_prefix="search")

    # HOME FEED
    else:
        st.markdown(f"### 🏠 Home — {category.replace('_', ' ').title()}")

        home_movies = api_get(
            "/home",
            {"category": category, "limit": 24},
        )

        if not home_movies:
            st.error("Home feed failed.")
        else:
            poster_grid(home_movies, cols=grid_cols, key_prefix="home")

# ==========================================================
# DETAILS VIEW
# ==========================================================
else:
    tmdb_id = st.session_state.tmdb_id

    if not tmdb_id:
        st.warning("No movie selected.")
        st.stop()

    if st.button("← Back to Home"):
        st.session_state.view = "home"
        st.session_state.tmdb_id = None
        st.rerun()

    details = api_get(f"/movie/id/{tmdb_id}")

    if not details:
        st.error("Failed to load movie details.")
        st.stop()

    left, right = st.columns([1, 2.5])

    with left:
        if details.get("poster_url"):
            st.image(details["poster_url"], use_container_width=True)

    with right:
        st.markdown(f"## {details['title']}")
        st.markdown(
            f"<div class='small-muted'>Release: {details.get('release_date','-')}</div>",
            unsafe_allow_html=True,
        )

        genres = ", ".join([g["name"] for g in details.get("genres", [])])
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Overview")
        st.write(details.get("overview", "No overview available."))

    st.divider()

    st.markdown("### ✅ Recommendations")

    bundle = api_get(
        "/movie/search",
        {"query": details["title"], "tfidf_top_n": 12, "genre_limit": 12},
    )

    if bundle:
        st.markdown("#### 🔎 Similar Movies")
        poster_grid(
            [
                {
                    "tmdb_id": x["tmdb"]["tmdb_id"],
                    "title": x["tmdb"]["title"],
                    "poster_url": x["tmdb"]["poster_url"],
                }
                for x in bundle["tfidf_recommendations"]
                if x.get("tmdb")
            ],
            cols=grid_cols,
            key_prefix="tfidf",
        )

        st.markdown("#### 🎭 More Like This")
        poster_grid(
            bundle["genre_recommendations"],
            cols=grid_cols,
            key_prefix="genre",
        )
