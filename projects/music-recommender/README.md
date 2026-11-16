# Music Recommendation System

A small content-based recommender. It ranks tracks by TF-IDF similarity between a natural-language query and each track's artist, genre and tags.

## Run
`python -m pip install pandas scikit-learn`
`python recommend.py --query "calm ambient instrumental" --limit 3`

Replace `songs.csv` with your own licensed catalogue using the columns `track,artist,genre,tags`. The bundled catalogue is illustrative sample data, not a commercial music dataset.
