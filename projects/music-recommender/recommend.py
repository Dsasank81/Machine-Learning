"""Content-based music recommender using cosine similarity over track metadata."""
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED = {"track", "artist", "genre", "tags"}

def recommend(query, catalog, limit=5):
    text = (catalog["artist"].fillna("") + " " + catalog["genre"].fillna("") + " " + catalog["tags"].fillna(""))
    matrix = TfidfVectorizer(stop_words="english").fit_transform(text)
    q = TfidfVectorizer(stop_words="english", vocabulary=None)
    # Fit query in the same feature space via the catalog vectorizer.
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).ravel()
    result = catalog.copy()
    result["similarity"] = scores
    return result.sort_values("similarity", ascending=False).head(limit)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="songs.csv")
    parser.add_argument("--query", required=True, help="Describe a mood, genre, artist or sound")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    data = pd.read_csv(args.catalog)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    print(recommend(args.query, data, args.limit)[["track", "artist", "genre", "similarity"]].to_string(index=False))

if __name__ == "__main__":
    main()
