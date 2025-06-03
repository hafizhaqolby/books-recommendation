
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

#### Dataset Dimensions

| File Name      | Rows     | Columns | Description                              |
|----------------|----------|---------|------------------------------------------|
| `books.csv`    | 10,000   | 23      | Contains metadata for 10,000 books       |
| `book_tags.csv`| 999,912  | 3       | User-assigned tags (represented by IDs)  |
| `tags.csv`     | 34,252   | 2       | Mapping of tag IDs to tag names          |

### Dataset Feature Description (`books.csv`)

| No. | Column Name                  | Non-Null Count | Data Type | Description                                                                 |
|-----|------------------------------|----------------|-----------|-----------------------------------------------------------------------------|
| 1   | id                           | 10,000         | int64     | Unique identifier for each book entry                                      |
| 2   | book_id                      | 10,000         | int64     | Identifier used across other related datasets                              |
| 3   | best_book_id                 | 10,000         | int64     | ID of the best edition of the book                                         |
| 4   | work_id                      | 10,000         | int64     | Work ID representing all editions                                          |
| 5   | books_count                  | 10,000         | int64     | Number of editions available                                               |
| 6   | isbn                         | 9,300          | object    | ISBN number (10 digits)                                                    |
| 7   | isbn13                       | 9,415          | float64   | ISBN number (13 digits)                                                    |
| 8   | authors                      | 10,000         | object    | Author(s) of the book                                                      |
| 9   | original_publication_year    | 9,979          | float64   | Year when the book was first published                                     |
| 10  | original_title               | 9,415          | object    | Original title of the book                                                 |
| 11  | title                        | 10,000         | object    | Current book title                                                         |
| 12  | language_code                | 8,916          | object    | Language of the book                                                       |
| 13  | average_rating               | 10,000         | float64   | Average user rating                                                        |
| 14  | ratings_count                | 10,000         | int64     | Total number of ratings received                                           |
| 15  | work_ratings_count           | 10,000         | int64     | Number of ratings across all editions                                      |
| 16  | work_text_reviews_count      | 10,000         | int64     | Number of text reviews across all editions                                 |
| 17  | ratings_1                    | 10,000         | int64     | Count of 1-star ratings                                                    |
| 18  | ratings_2                    | 10,000         | int64     | Count of 2-star ratings                                                    |
| 19  | ratings_3                    | 10,000         | int64     | Count of 3-star ratings                                                    |
| 20  | ratings_4                    | 10,000         | int64     | Count of 4-star ratings                                                    |
| 21  | ratings_5                    | 10,000         | int64     | Count of 5-star ratings                                                    |
| 22  | image_url                    | 10,000         | object    | URL to the book’s image                                                    |
| 23  | small_image_url              | 10,000         | object    | URL to a smaller version of the book’s image                               |

---

## 📊 Exploratory Data Analysis (EDA)

- The dataset includes more than 10,000 books.
### 1. Distribution of Book Ratings
The histogram shows that most books have an average rating between 3.5 and 4.3, peaking around 4.0. The distribution is approximately normal with a slight right skew, indicating that high ratings are common, while very low or very high ratings are rare. This supports the use of 4.0 as a relevance threshold in evaluating recommendation performance (e.g., for Precision@K).

### 2. Top 10 Authors with Most Books
The dataset reveals **Stephen King** as the most prolific author with the highest number of books (60), followed by:
- Nora Roberts (~50 books)
- Dean Koontz (~40 books)  
- Other notable authors include Agatha Christie and James Patterson

---

## 🧹 Data Preparation

This section describes the steps taken to prepare the dataset for building a content-based recommendation system. The preparation includes merging datasets, cleaning data, and engineering features to be used in modeling.

### 1. Merging Tag Names into Book Tags
We started by merging the `book_tags` dataset with the `tags` dataset to obtain human-readable tag names for each book.

```python
# Merge tag names into book_tags
book_tags = book_tags.merge(tags, on='tag_id')
```
### 2. Selecting Top Tags per Book
To reduce noise, we only kept the top 10 most frequently assigned tags for each book.

```python
# Remove duplicates and keep most relevant tags (e.g., top 10 per book)
top_tags = book_tags.groupby('goodreads_book_id').apply(
    lambda x: x.sort_values('count', ascending=False).head(10)
).reset_index(drop=True)
```
### 3. Aggregating Tags into a Single Feature
We grouped and aggregated the top tags into a single string per book, which can later be used for content similarity.

```python
# Create combined tag string per book
book_tags_agg = top_tags.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
book_tags_agg.columns = ['book_id', 'tags']
```
### 4. Merging Tags with Book Metadata
The generated tag strings were merged back into the main `books` dataset.

```python
# Merge back to books
books = books.merge(book_tags_agg, left_on='book_id', right_on='book_id', how='left')
books['tags'] = books['tags'].fillna('')
```
### 5. Feature Engineering: Creating the Content Column
We constructed a new feature called `content` by concatenating the book's title, authors, and tags. This combined feature is used as the textual input for the recommendation model.

```python
# Create content feature using title, authors, and tags
books['content'] = books['title'] + ' ' + books['authors'] + ' ' + books['tags']
```
This `content` column forms the basis of our similarity-based recommendations, which will be vectorized and compared using cosine similarity in the next phase.

### 6. Vectorizing the Content Feature with TF-IDF
To numerically represent the content column, we applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This transformation helps identify important words while reducing the weight of common ones.

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['content'])
```
The resulting `tfidf_matrix` is a sparse matrix representation of each book’s content, ready to be used in similarity calculations during the modeling phase.

---

## 🤖 Modeling

In this stage, we build a content-based book recommendation system using cosine similarity. After transforming the text data into numerical form using TF-IDF (as explained in the Data Preparation section), we calculate the similarity between books and generate recommendations.

### 1. Similarity Calculation
We use **cosine similarity** to measure the similarity between TF-IDF vectors of books. The cosine similarity score ranges from 0 (completely different) to 1 (exactly similar).

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```
### 2. Recommendation Function
We define a function to return the top-N most similar books based on a given input title. It handles partial title input by matching it with the closest book title in the dataset.

```python
def recommend(title_partial, df=books, similarity=cosine_sim, top_n=5):
    all_titles = df['title'].tolist()
    close_matches = get_close_matches(title_partial, all_titles, n=1, cutoff=0.5)
    if not close_matches:
        return f"No close match found for '{title_partial}'."

    title = close_matches[0]
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][['title', 'authors', 'average_rating']]
```

---

## 🧪 Evaluation

- We use **Precision@K** as the evaluation metric, where K = 5.
- A book is considered relevant if it has an average rating >= 4.0.
- Example:
  ```
  Showing recommendations for: Hunger Games
  Precision@5 for 'Hunger Games': 0.20
  ```
- Here is an example of the top-5 book recommendations when the input is:
  ```
  recommend("Hunger Games")
  ```
  **Output**
```
| Title                                                | Authors               | Average Rating |
|------------------------------------------------------|-----------------------|----------------|
| Keep the Aspidistra Flying                           | George Orwell         | 3.87           |
| The World of the Hunger Games (Hunger Games Trilogy) | Kate Egan             | 4.48           |
| Silas Marner                                         | George Eliot          | 3.60           |
| Nostromo                                             | Joseph Conrad         | 3.81           |
| The Mill on the Floss                                | George Eliot, A.S.    | 3.77           |
```

---

## 📌 Conclusion

This project shows how a simple content-based approach can deliver meaningful recommendations. Future work can incorporate genres, descriptions, and user preferences for hybrid systems.


