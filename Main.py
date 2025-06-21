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
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Cache for poster URLs to avoid repeated API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_poster(movie_title):
    """Get poster URL with on-demand caching to improve performance"""
    return fetch_poster_with_fallbacks(movie_title)

# ----------------------------
# Enhanced Poster Fetching with Multiple Fallbacks
# ----------------------------
def fetch_poster_with_fallbacks(movie_title):
    """
    Fetch movie poster with multiple fallback options
    """
    # Use a public TMDB API key for search
    public_api_key = "1b94fd19aee581580f557b35be07720"  # Public demo key
    
    # Method 1: Search by movie title
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {
            "query": movie_title,
            "api_key": public_api_key,
            "language": "en-US"
        }
        response = session.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    
    # Method 2: Try with user's personal TMDB API key if available
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if tmdb_api_key:
        try:
            search_url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                "query": movie_title,
                "api_key": tmdb_api_key,
                "language": "en-US"
            }
            response = session.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('results') and len(data['results']) > 0:
                    poster_path = data['results'][0].get('poster_path')
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception:
            pass
    
    # Method 3: Create a styled placeholder with movie title
    # Clean the title for URL encoding
    clean_title = movie_title.replace(":", "").replace("'", "").replace('"', "")
    encoded_title = requests.utils.quote(clean_title[:30])  # Limit length
    return f"https://via.placeholder.com/500x750/1E1E1E/FF4B4B?text={encoded_title}"

# ----------------------------
# ML Model Recommender
# ----------------------------
def recommend(movie):
    index = Movie[Movie['title'] == movie].index[0]
    distances = sorted(list(enumerate(Similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_name = []
    recommended_movie_Poster = []
    for i in distances[1:6]:
        movie_title = Movie.iloc[i[0]].title
        recommended_movie_name.append(movie_title)
        # Use on-demand cached poster fetching
        recommended_movie_Poster.append(get_cached_poster(movie_title))
    return recommended_movie_name, recommended_movie_Poster


# ----------------------------
# Streamlit UI
# ----------------------------
# Header with custom styling
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #FF4B4B; font-size: 3em;'> MOVIEMATCH</h1>
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
                    <img src='{posters[i]}' style='width: 100%; border-radius: 5px;' alt='{names[i]}'>
                    <div class='movie-title'>{names[i]}</div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <hr style="border: 0.5px solid #ccc;" />
    <div style="text-align: center; font-size: 14px; color: gray;">
        Made with ❤️ using <b>Python</b> & <b>Machine Learning</b> | Built by <a href="https://www.linkedin.com/in/siddhantrathod" target="_blank">Siddhant Rathod</a> <br>
        Dataset: <a href="https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata" target="_blank">TMDB 5000 Movie Dataset</a> | Source Code Coming Soon on GitHub 🚀 <br>
        &copy; 2025 MovieMatch | All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)
