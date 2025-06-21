# 🎬 MovieMatch – Movie Recommendation System

MovieMatch is a machine learning-based movie recommendation system built using Python. It leverages content-based filtering on movie metadata to recommend similar films based on user preferences.

---

## 📌 Features

- 🔍 **Content-Based Filtering** – Recommends movies similar to the one you select.
- 📂 Preprocessed datasets from TMDB (The Movie Database).
- 🧠 Machine Learning with cosine similarity for movie matching.
- 💾 Uses pre-saved `.pkl` files for performance (faster load).
- 💡 Simple and clean codebase, perfect for learning and extension.
- 🖼️ Ready for deployment using Streamlit or any Python-based web framework.

---


## 🛠️ How It Works

1. Reads movie and credits data from CSV.
2. Cleans and merges them using pandas.
3. Extracts useful metadata (genres, keywords, cast, crew, etc.).
4. Transforms it into a vector using `CountVectorizer`.
5. Calculates cosine similarity between movies.
6. Returns top N most similar movies.

---

## 🧪 Example

Input:
Recommend movies similar to: Inception


Output:
- Interstellar
- The Prestige
- The Matrix
- Shutter Island
- Memento


# 🧠 ML Libraries Used
- scikit-learn
- nltk
- pandas
- numpy
- streamlit

📦 Dataset Source
TMDB 5000 Movie Dataset: Kaggle

🧑‍💻 Author
Siddhant Rathod
