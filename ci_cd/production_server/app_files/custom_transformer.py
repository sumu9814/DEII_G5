from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
