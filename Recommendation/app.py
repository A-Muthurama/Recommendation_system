from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load data
movies = pd.read_csv(r'C:\Users\iammu\OneDrive\Desktop\Project2\movies.csv')  # Replace with the correct path to your dataset
movies['genre'] = movies['genre'].fillna('')  # Handle missing genres

# Create TF-IDF matrix based on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genre'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on movieId
def recommend_movies(movie_id, cosine_sim=cosine_sim, movies=movies, top_n=10):
    if movie_id not in movies['movieId'].values:
        return {"error": f"Movie ID {movie_id} not found in the dataset"}

    # Get the index of the movie that matches the ID
    idx = movies[movies['movieId'] == movie_id].index[0]

    # Get similarity scores for all movies with the input movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top_n similar movies
    top_movies = sim_scores[1:top_n + 1]  # Exclude the first movie (itself)

    # Return movie titles and genres
    recommendations = [{"title": movies.iloc[i[0]]['title'], "genre": movies.iloc[i[0]]['genre']} for i in top_movies]
    return recommendations

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_id = request.form.get('movie_id', type=int)
    top_n = request.form.get('top_n', type=int, default=5)

    if not movie_id:
        return jsonify({"error": "Please provide a movie ID"}), 400

    recommendations = recommend_movies(movie_id, top_n=top_n)
    if "error" in recommendations:
        return jsonify(recommendations), 400

    return render_template('recommendations.html', movie_id=movie_id, recommendations=recommendations)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
