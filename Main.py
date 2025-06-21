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
    page_title="CineMatch",
    page_icon="./assets/Sic.ico",                      
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .movie-title {
        font-size: 1.2em;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 10px;
    }
    .movie-card {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Load ML model data
Movie = pickle.load(open('Artifacts/Movie_list.pkl', 'rb'))

# Load split similarity matrices
@st.cache_data
def load_similarity_matrices():
    """Load and combine the split similarity matrices"""
    part1 = pickle.load(open('Artifacts/Similarity_part1.pkl', 'rb'))
    part2 = pickle.load(open('Artifacts/Similarity_part2.pkl', 'rb'))
    part3 = pickle.load(open('Artifacts/Similarity_part3.pkl', 'rb'))
    
    # Combine the parts back into the full similarity matrix
    similarity = np.vstack([part1, part2, part3])
    return similarity

Similarity = load_similarity_matrices()
Movie_List = Movie['title'].values

# Configure retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)


# ----------------------------
# TMDB: Fetch Poster for Recs
# ----------------------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
        headers = {
            "accept": "application/json",
            "Authorization": os.getenv("JWT_SECRET")
        }
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except requests.exceptions.RequestException as e:
        # Log error but don't print to avoid cluttering the UI
        # print(f"Error fetching poster for movie_id {movie_id}: {e}")
        time.sleep(2)  # Longer delay before next request
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception as e:
        # Catch any other unexpected errors
        time.sleep(2)
        return "https://via.placeholder.com/500x750?text=Error"


# ----------------------------
# ML Model Recommender
# ----------------------------
def recommend(movie):
    index = Movie[Movie['title'] == movie].index[0]
    distances = sorted(list(enumerate(Similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_name = []
    recommended_movie_Poster = []
    for i in distances[1:6]:
        movie_id = Movie.iloc[i[0]].movie_id
        recommended_movie_name.append(Movie.iloc[i[0]].title)
        recommended_movie_Poster.append(fetch_poster(movie_id))
    return recommended_movie_name, recommended_movie_Poster


# ----------------------------
# Streamlit UI
# ----------------------------
# Header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #FF4B4B; font-size: 3em;'> CINEMATCH</h1>
        <h1 style='color: #FFFFFF; font-size: 2em;'> Movie Recommendation System</h1>
        <p style='color: #FFFFFF; font-size: 1.2em;'>Discover your next favorite movie!</p>
    </div>
""", unsafe_allow_html=True)

# Main content
st.markdown("### 🎯 Select Your Movie")
movie = st.selectbox(
    'Choose your favorite movie:',
    Movie_List,
    help="Select a movie you like to get similar recommendations"
)

if st.button("🎯 Get Recommendations", key="recommend"):
    with st.spinner('Finding the perfect movies for you...'):
        names, posters = recommend(movie)
        st.session_state.recommendations = (names, posters)

# Display recommendations if available
if 'recommendations' in st.session_state and isinstance(st.session_state.recommendations, tuple):
    names, posters = st.session_state.recommendations
    st.markdown("### 🎯 Recommended Movies")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"""
                <div class='movie-card'>
                    <img src='{posters[i]}' style='width: 100%; border-radius: 5px;'>
                    <div class='movie-title'>{names[i]}</div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 50px;'>
        <p style='color: #666666;'>Powered by TMDB API and Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
