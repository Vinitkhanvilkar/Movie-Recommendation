import requests
import streamlit as st
import pickle
import time
import os
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MovieMatch",
    page_icon="./assets/Sic.ico",                      
    layout="wide"
)

# Inject Google Fonts and Enhanced Custom CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;400&display=swap" rel="stylesheet">
<style>
body, .main, .stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    font-family: 'Montserrat', sans-serif;
}

/* Button Styling */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 1.1em;
    box-shadow: 0 4px 14px 0 rgba(0,114,255,0.25);
    transition: background 0.3s, transform 0.2s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
    color: #fff;
    transform: translateY(-2px) scale(1.05);
}


</style>

""", unsafe_allow_html=True)

# Load ML model data
Movie = pickle.load(open('Artifacts/Movie_list.pkl', 'rb'))

# Load split similarity matrices
@st.cache_data
def load_similarity_matrices():
    part1 = pickle.load(open('Artifacts/Similarity_part1.pkl', 'rb'))
    part2 = pickle.load(open('Artifacts/Similarity_part2.pkl', 'rb'))
    part3 = pickle.load(open('Artifacts/Similarity_part3.pkl', 'rb'))
    similarity = np.vstack([part1, part2, part3])
    return similarity

Similarity = load_similarity_matrices()
Movie_List = Movie['title'].values

# Retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

@st.cache_data(ttl=3600)
def get_cached_poster(movie_title):
    return fetch_poster_with_fallbacks(movie_title)

def fetch_poster_with_fallbacks(movie_title):
    public_api_key = "410e4aae26f611565b373d6c0701a6e2"
    try:
        response = session.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": movie_title, "api_key": public_api_key, "language": "en-US"},
            timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if tmdb_api_key:
        try:
            response = session.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"query": movie_title, "api_key": tmdb_api_key, "language": "en-US"},
                timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    poster_path = data['results'][0].get('poster_path')
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception:
            pass
    clean_title = movie_title.replace(":", "").replace("'", "").replace('"', "")
    encoded_title = requests.utils.quote(clean_title[:30])
    return f"https://via.placeholder.com/500x750/1E1E1E/FF4B4B?text={encoded_title}"

def recommend(movie):
    index = Movie[Movie['title'] == movie].index[0]
    distances = sorted(list(enumerate(Similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_name = []
    recommended_movie_Poster = []
    for i in distances[1:6]:
        movie_title = Movie.iloc[i[0]].title
        recommended_movie_name.append(movie_title)
        recommended_movie_Poster.append(get_cached_poster(movie_title))
    return recommended_movie_name, recommended_movie_Poster

# UI - Header
st.markdown("""
    <div class='header-animate' style='text-align: center; padding: 24px 0 10px 0;'>
        <h1 style='color: #FF4B4B; font-size: 3.2em; letter-spacing: 2px; font-family: Montserrat, sans-serif; margin-bottom: 0.2em;'>
            <span style="background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                MOVIEMATCH
            </span>
        </h1>
        <h2 style='color: #FFFFFF; font-size: 2em; font-weight: 400; margin-bottom: 0.3em;'>
            Movie Recommendation System
        </h2>
        <p style='color: #FFFFFF; font-size: 1.2em; font-weight: 400;'>
            Discover your next favorite movie!
        </p>
    </div>
""", unsafe_allow_html=True)

# Select movie
st.markdown("### 🎯 <span style='color:#FF4B4B;'>Select Your Movie</span>", unsafe_allow_html=True)
movie = st.selectbox('Choose your favorite movie:', Movie_List, help="Select a movie you like to get similar recommendations")

if st.button("🎯 Get Recommendations", key="recommend"):
    with st.spinner('Finding the perfect movies for you...'):
        names, posters = recommend(movie)
        st.session_state.recommendations = (names, posters)

if 'recommendations' in st.session_state and isinstance(st.session_state.recommendations, tuple):
    names, posters = st.session_state.recommendations
    st.markdown("### 🎯 <span style='color:#FF4B4B;'>Recommended Movies</span>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"""
                <div class='movie-card'>
                    <img src='{posters[i]}' style='width: 100%; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);' alt='{names[i]}'>
                    <div class='movie-title'>{names[i]}</div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 50px;'>
        <p style='color: #888; font-size: 1.05em;'>Made With 🤍 By Vinit</p>
    </div>
""", unsafe_allow_html=True)
