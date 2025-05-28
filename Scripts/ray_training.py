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

import ray
from ray import tune
from ray.air import session


# data loaded from github to give access to the workers
def load_data():
    url = "https://raw.githubusercontent.com/pitabogdan2002/DEII/main/raw_github_data.csv"
    return pd.read_csv(url)

# feature engineering class
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.language_counts_map = {}
        self.avg_desc_len = 0
        self.medians = {}

    def fit(self, X, y=None):
        for col in ['days_since_created', 'days_since_updated', 'days_since_pushed']:
            if col in X.columns:
                self.medians[col] = X[col].median()
        self.avg_desc_len = X['description_len'].mean()
        self.language_counts_map = X['language'].value_counts().to_dict()
        return self

    def transform(self, X):
        df_eng = X.copy()
        for col in ['days_since_created', 'days_since_updated', 'days_since_pushed']:
            if col in df_eng.columns:
                df_eng[col] = df_eng[col].fillna(self.medians.get(col, 0))

        df_eng['commit_per_day'] = df_eng['commit_count'] / (df_eng['days_since_created'] + 1)
        df_eng['update_frequency'] = df_eng['days_since_created'] / (df_eng['days_since_updated'] + 1)
        df_eng['recent_activity'] = 1 / (df_eng['days_since_pushed'] + 1)
        df_eng['activity_score'] = df_eng['commit_count'] / (df_eng['days_since_created'] + 1)
        df_eng['description_len_ratio'] = df_eng['description_len'] / (self.avg_desc_len + 1)
        df_eng['language_popularity'] = df_eng['language'].map(self.language_counts_map).fillna(0)

        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        for col in df_eng.columns:
            if df_eng[col].dtype in ['float64', 'int64']:
                df_eng[col] = df_eng[col].fillna(df_eng[col].median())

        return df_eng


# prepare data for prediction
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


def tune_models_with_ray():
    def train_model(config):
        df = load_data()
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
        feature_eng, preprocessor, _ = create_full_pipeline(X_train)

        model_type = config["model_type"]

        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                random_state=42
            )
        elif model_type == "ridge":
            model = Ridge(alpha=config["alpha"])
        elif model_type == "lasso":
            model = Lasso(alpha=config["alpha"])
        elif model_type == "linear_regression":
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        pipeline = Pipeline([
            ('feature_engineering', feature_eng),
            ('preprocessor', preprocessor),
            ('selector', SelectKBest(f_regression, k=10)),
            ('model', model)
        ])

        score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2').mean()
        session.report({"score": score})


    combinations = {
        "model_type": tune.grid_search(["random_forest", "ridge", "lasso", "linear_regression", "gradient_boosting"]),
        "alpha": tune.sample_from(lambda spec: np.random.choice([0.001, 0.01, 0.1, 1.0, 10.0]) if spec.config["model_type"] in ["ridge", "lasso"] else None),
        "n_estimators": tune.sample_from(lambda spec: np.random.choice([50, 100, 200]) if spec.config["model_type"] in ["random_forest", "gradient_boosting"] else None),
        "max_depth": tune.sample_from(lambda spec: np.random.choice([3, 5, 10, 15]) if spec.config["model_type"] in ["random_forest", "gradient_boosting"] else None),
    }

    ray.init(address="auto")
    result = tune.run(
        train_model,
        config=combinations,
        metric="score",
        mode="max",
        storage_path="file:///home/appuser/ray_results",
        name="github_model_tuning"
    )

    result.results_df.to_csv("ray_tuning_results.csv", index=False)
    print("Saved Ray tuning results to ray_tuning_results.csv")
    return result.get_best_config(metric="score", mode="max")


def save_outputs(model, score):
    joblib.dump(model, "model_ray.pkl")
    with open("best_accuracy_ray.txt", "w") as f:
        f.write(str(score))
    print("Model and score saved.")


def main():
    best_config = tune_models_with_ray()
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
    feature_eng, preprocessor, _ = create_full_pipeline(X_train)

    if best_config["model_type"] == "random_forest":
        model = RandomForestRegressor(n_estimators=best_config["n_estimators"], max_depth=best_config["max_depth"], random_state=42)
    elif best_config["model_type"] == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=best_config["n_estimators"], max_depth=best_config["max_depth"], random_state=42)
    elif best_config["model_type"] == "ridge":
        model = Ridge(alpha=best_config["alpha"])
    elif best_config["model_type"] == "lasso":
        model = Lasso(alpha=best_config["alpha"])
    elif best_config["model_type"] == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError("Invalid model_type")

    pipeline = Pipeline([
        ('feature_engineering', feature_eng),
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(f_regression, k=10)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Final model R-squared: {score:.4f}")
    save_outputs(pipeline, score)


if __name__ == "__main__":
    main()
