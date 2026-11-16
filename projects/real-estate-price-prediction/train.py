"""Train and evaluate a tabular house-price regression baseline."""
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import make_column_selector
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="housing.csv")
    ap.add_argument("--target", default="price")
    args = ap.parse_args()
    df = pd.read_csv(args.data).dropna(subset=[args.target])
    X, y = df.drop(columns=[args.target]), df[args.target]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("prep", prep), ("regressor", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"MAE: {mean_absolute_error(yte,pred):.2f}")
    print(f"RMSE: {mean_squared_error(yte,pred)**0.5:.2f}")
    print(f"R2: {r2_score(yte,pred):.3f}")
    joblib.dump(model, "house_price_pipeline.joblib")

if __name__ == "__main__":
    main()
