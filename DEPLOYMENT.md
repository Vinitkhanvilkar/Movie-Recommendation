# MovieMatch Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
1. Your code is pushed to a GitHub repository
2. You have a Streamlit Cloud account

### Deployment Steps
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set the main file path to `Main.py`
5. Deploy!

### Environment Variables (Optional)
If you want to use your own TMDB API key for better poster quality:

1. Go to [TMDB](https://www.themoviedb.org/settings/api) to get a v3 API key.
2. In Streamlit Cloud, go to your app settings.
3. Add a secret with the following:
   - Key: `TMDB_API_KEY`
   - Value: Your TMDB v3 API key (it's a long string of characters)

### Troubleshooting

#### Images Not Showing
The app now has multiple fallback mechanisms:
1. **Public API Key**: Uses a demo TMDB API key
2. **User API Key**: Uses your personal TMDB API key if configured
3. **Placeholder Images**: Shows styled placeholders with movie titles

#### Performance Issues
- The app uses caching to improve performance
- Poster URLs are cached for 1 hour
- Similarity matrices are cached for the session

### File Structure
```
├── Main.py              # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .streamlit/         # Streamlit configuration
│   └── config.toml
├── Artifacts/          # ML model files
│   ├── Movie_list.pkl
│   ├── Similarity_part1.pkl
│   ├── Similarity_part2.pkl
│   ├── Similarity_part3.pkl
│   └── Similarity_part4.pkl
└── assets/             # Static assets
    ├── Sic.ico
    └── Sicon.ico
```

### Features
- ✅ Movie recommendation system
- ✅ Multiple poster fetching methods
- ✅ Responsive design
- ✅ Caching for performance
- ✅ Error handling and fallbacks
- ✅ Works without API keys 