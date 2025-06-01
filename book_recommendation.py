
# Book Recommendation System using Content-Based Filtering

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches

# =========================
# 📥 Data Understanding
# =========================

# Load dataset
books = pd.read_csv('books.csv')

# Basic info
print("Dataset contains", len(books), "books.")
print("Columns:", books.columns)

# Handle missing values
books['title'] = books['title'].fillna('')
books['authors'] = books['authors'].fillna('')

# =========================
# 📊 Exploratory Data Analysis (EDA)
# =========================

print("Top 5 books by average rating:")
print(books[['title', 'average_rating']].sort_values(by='average_rating', ascending=False).head())

# =========================
# 🧹 Data Preparation
# =========================

# Combine title and author as features
books['combined_features'] = books['title'] + ' ' + books['authors']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# =========================
# 🤖 Modeling
# =========================

# Recommendation function
def recommend(title_partial, df=books, similarity=cosine_sim, top_n=5):
    all_titles = df['title'].tolist()
    close_matches = get_close_matches(title_partial, all_titles, n=1, cutoff=0.5)
    if not close_matches:
        return f"No close match found for '{title_partial}'."

    title = close_matches[0]
    print(f"Showing recommendations for: {title}")

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][['title', 'authors', 'average_rating']]

# =========================
# 🧪 Evaluation
# =========================

# Define relevant books as those with high rating
relevant_books = set(books[books['average_rating'] >= 4.0]['title'])

# Evaluate with Precision@K
def evaluate_precision_at_k(title_input, k=5):
    recommendations = recommend(title_input)
    if isinstance(recommendations, str):
        return recommendations

    recommended_titles = recommendations['title'].values
    relevant_recommended = [title for title in recommended_titles if title in relevant_books]

    precision = len(relevant_recommended) / k
    return f"Precision@{k} for '{title_input}': {precision:.2f}'"
