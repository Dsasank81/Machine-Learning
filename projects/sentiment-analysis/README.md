# Sentiment Analysis

A reproducible TF-IDF + Logistic Regression baseline for binary review sentiment classification.

## Setup
`python -m pip install pandas scikit-learn matplotlib joblib`

Place a labelled CSV at `data/reviews.csv` with columns `review,sentiment`, then run:

```bash
python train.py --data data/reviews.csv
```

The script uses a stratified 80/20 split, reports precision/recall/F1, saves a confusion matrix and serializes the fitted pipeline. Use a properly licensed dataset and keep the raw dataset out of Git if its terms prohibit redistribution.

This is a reconstructed baseline implementation, not a claim that the original experimental results or dataset are included.
