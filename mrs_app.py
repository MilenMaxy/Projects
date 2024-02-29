import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load the saved components from the pickle file
with open('movie_recommendation_model.pkl', 'rb') as file:
    saved_components = pickle.load(file)

df = saved_components['df']
vect = saved_components['vect']
similar = saved_components['similar']

def get_movie_recommendations(movie_name):
    name_match = difflib.get_close_matches(movie_name, df['title'].tolist())

    if name_match:
        closest_match = name_match[0]
        index_finder = df[df.title == closest_match].index.values[0]
        similarity_score = list(enumerate(similar[index_finder]))
        sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommendations = []
        for movie in sorted_movies:
            index = movie[0]
            title = df.loc[index, 'title']
            recommendations.append(title)

        return recommendations[:15]  # Return top 15 recommendations
    else:
        return []

# Streamlit app code
st.title("Movie Recommendation App")

# User input
user_input = st.text_input("Enter Your Favorite Movie Name:")

if user_input:
    recommendations = get_movie_recommendations(user_input)

    if recommendations:
        st.subheader("Movies Suggested For You To Watch:")
        for i, movie_title in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie_title}")
    else:
        st.warning("Sorry! No close matches found for the entered movie name.")

