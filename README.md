
# Book Recommendation System using Content-Based Filtering

---

## 📌 Project Overview

In this project, we develop a **Content-Based Recommendation System** for books using the Goodreads dataset. Content-based filtering is used to provide personalized book recommendations based on features such as the book title and authors. 

This system aims to improve book discovery by recommending similar books to users based on their interests.

---

## 🎯 Business Understanding

### Problem Statements
1. Users often struggle to discover new books that match their interests.
2. Book platforms need personalized systems to improve user experience and engagement.

### Goals
1. To build a recommendation system that suggests books similar to a given title.
2. To evaluate the model using appropriate metrics to ensure relevance.

### Solution Approach
We use Content-Based Filtering with TF-IDF vectorization on book titles and authors. Cosine similarity is calculated to recommend the top-N similar books.

---

## 📂 Data Understanding

The dataset used is [GoodBooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k), which contains book metadata and user ratings.

The dataset used includes the following:
- `books.csv`: Contains metadata of books such as `title`, `authors`, `average_rating`.

Example of features:
- `title`: The title of the book.
- `authors`: Name(s) of the author(s).
- `average_rating`: The average rating score given by users.

---

## 📊 Exploratory Data Analysis (EDA)

- The dataset includes more than 10,000 books.
- Books like *Harry Potter*, *The Hunger Games*, and *The Fault in Our Stars* have very high ratings.
- `average_rating` is used to determine relevance for evaluation.

---

## 🧹 Data Preparation

- Missing values in `title` and `authors` were filled with empty strings.
- Features were combined (`title` + `authors`) to form a unified text input.
- TF-IDF Vectorizer was applied to convert the text into numerical vectors.

---

## 🤖 Modeling

- We used **TF-IDF Vectorization** for text features and **Cosine Similarity** for measuring similarity.
- A book title is used as input, and the system recommends the top-N most similar books.

---

## 🧪 Evaluation

- We use **Precision@K** as the evaluation metric, where K = 5.
- A book is considered relevant if it has an average rating >= 4.0.
- Example:
  ```
  Showing recommendations for: Hunger Games
  Precision@5 for 'Hunger Games': 0.20
  ```

---

## 📌 Conclusion

This project shows how a simple content-based approach can deliver meaningful recommendations. Future work can incorporate genres, descriptions, and user preferences for hybrid systems.


