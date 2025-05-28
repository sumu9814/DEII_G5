import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin

from custom_transformer import FeatureEngineeringTransformer

def load_data():
    return pd.read_csv("raw_github_data.csv")


def prepare_data_for_modeling(df):
    y = df['stargazers_count']
    X = df.drop(['full_name', 'stargazers_count', 'watchers_count', 'forks_count'], axis=1, errors='ignore')
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_full_pipeline(X_train):
    feature_eng = FeatureEngineeringTransformer()
    X_train_eng = feature_eng.fit_transform(X_train)
    categorical_cols = [col for col in X_train_eng.columns if X_train_eng[col].dtype == 'object']
    numeric_cols = [col for col in X_train_eng.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

    return feature_eng, preprocessor, X_train_eng


def local_hyperparameter_tuning(X_train, y_train, feature_eng, preprocessor):
    models = {
        "random_forest": [RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
                          for n in [50, 100, 200] for d in [5, 10, 15]],
        "gradient_boosting": [GradientBoostingRegressor(n_estimators=n, max_depth=d, random_state=42)
                              for n in [50, 100, 200] for d in [5, 10, 15]],
        "ridge": [Ridge(alpha=a) for a in [0.001, 0.01, 0.1, 1.0, 10.0]],
        "lasso": [Lasso(alpha=a) for a in [0.001, 0.01, 0.1, 1.0, 10.0]],
        "linear_regression": [LinearRegression()]
    }

    best_score = -np.inf
    best_model = None

    for model_list in models.values():
        for model in model_list:
            pipeline = Pipeline([
                ('feature_engineering', feature_eng),
                ('preprocessor', preprocessor),
                ('selector', SelectKBest(f_regression, k=10)),
                ('model', model)
            ])
            score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2').mean()
            if score > best_score:
                best_score = score
                best_model = model

    return best_model, best_score


def save_outputs(model, score):
    joblib.dump(model, "model_local.pkl")
    with open("best_accuracy_local.txt", "w") as f:
        f.write(str(score))
    print("Model and score saved.")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
    feature_eng, preprocessor, _ = create_full_pipeline(X_train)

    best_model, _ = local_hyperparameter_tuning(X_train, y_train, feature_eng, preprocessor)

    pipeline = Pipeline([
        ('feature_engineering', feature_eng),
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(f_regression, k=10)),
        ('model', best_model)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Final model R-squared: {score:.4f}")
    save_outputs(pipeline, score)


if __name__ == "__main__":
    main()

