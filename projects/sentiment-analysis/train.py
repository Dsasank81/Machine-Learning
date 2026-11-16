"""IMDb-style sentiment classifier. Expects CSV columns: review, sentiment (positive/negative)."""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/reviews.csv")
    p.add_argument("--text-column", default="review")
    p.add_argument("--label-column", default="sentiment")
    args = p.parse_args()
    df = pd.read_csv(args.data).dropna(subset=[args.text_column, args.label_column])
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_column].astype(str), df[args.label_column].astype(str),
        test_size=0.2, random_state=42, stratify=df[args.label_column])
    model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=50000)),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(classification_report(y_test, pred))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    import joblib
    joblib.dump(model, "sentiment_model.joblib")
    print("Saved sentiment_model.joblib and confusion_matrix.png")

if __name__ == "__main__":
    main()
