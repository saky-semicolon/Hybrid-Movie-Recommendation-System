import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model_path = "movie_recommendation_model_fixed.h5"
model = load_model(model_path)

# Mock data (Replace with your actual dataset or database)
movies = [
    {"id": 0, "title": "The Shawshank Redemption"},
    {"id": 1, "title": "The Godfather"},
    {"id": 2, "title": "The Dark Knight"},
    {"id": 3, "title": "Pulp Fiction"},
    {"id": 4, "title": "Forrest Gump"},
]

movie_titles = [movie["title"] for movie in movies]
movie_ids = {movie["title"]: movie["id"] for movie in movies}

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Search for a movie and get recommendations!")

# User input
selected_movie = st.selectbox("Select a movie you like:", movie_titles)

if st.button("Recommend"):
    # Get the movie ID of the selected movie
    selected_movie_id = movie_ids[selected_movie]

    # Prepare input for the model (e.g., map ID to input format)
    input_movie_id = np.array([selected_movie_id]).reshape(-1, 1)

    # Get predictions (Example: mock similar movie recommendations)
    predictions = model.predict(input_movie_id)
    similar_movies_ids = np.argsort(-predictions.flatten())[:5]  # Top 5 recommendations

    # Map back to movie titles
    recommended_movies = [movies[movie_id]["title"] for movie_id in similar_movies_ids]

    st.write("Recommended Movies for You:")
    for idx, movie in enumerate(recommended_movies, 1):
        st.write(f"{idx}. {movie}")
