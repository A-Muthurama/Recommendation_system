# Recommendation_system
The Movie Recommendation System is a Flask-based web application that provides personalized movie recommendations. It uses a content-based recommendation approach, leveraging the genre of movies to find and suggest similar ones. Users can input a movieId to get recommendations of similar movies based on their preferences.

How It Works
Dataset: The system uses a dataset containing movieId, title, and genre as features.
Content-Based Filtering:
Movie genres are processed using TF-IDF Vectorization to quantify the genre text data.
Cosine Similarity is calculated between movies to find the most similar ones.
User Input:
The user provides a movieId as input, which acts as the basis for generating recommendations.
Output:
The system returns a list of recommended movies, including their titles and genres.

