# Real Estate Price Prediction

Train a scikit-learn regression pipeline for a tabular housing dataset. The CSV must contain a numeric target column (default: `price`) and any numeric/categorical predictor columns.

## Run
`python -m pip install pandas scikit-learn joblib`
`python train.py --data housing.csv --target price`

Reports MAE, RMSE and R² and saves the preprocessing + model pipeline. Data quality, geographic leakage, temporal validation and local market drift must be assessed before using predictions for decisions. No property valuation is guaranteed.
