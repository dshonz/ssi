
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        #трансформация детерминированная.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ##
        #Возвращает копию DataFrame с новыми/исправленными колонками.
        ##
        X = X.copy()

        # Year_of_Release — привести к числу,  медианой если есть
        if 'Year_of_Release' in X.columns:
            X['Year_of_Release'] = pd.to_numeric(X['Year_of_Release'], errors='coerce')
            med = int(X['Year_of_Release'].median(skipna=True) if not X['Year_of_Release'].dropna().empty else 2000)
            X['Year_of_Release'] = X['Year_of_Release'].fillna(med).astype(int)

        # заполнить Unknown
        text_cols = ['Name', 'Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
        for c in text_cols:
            if c in X.columns:
                X[c] = X[c].fillna('Unknown').astype(str)

        # User_Score — заменить 'tbd' на NaN, затем numeric
        if 'User_Score' in X.columns:
            X['User_Score'] = X['User_Score'].replace('tbd', np.nan)
            X['User_Score'] = pd.to_numeric(X['User_Score'], errors='coerce').fillna(0.0)

        # Critic_Score - numeric
        if 'Critic_Score' in X.columns:
            X['Critic_Score'] = pd.to_numeric(X['Critic_Score'], errors='coerce').fillna(0.0)

        # Count columns - int
        for c in ['Critic_Count', 'User_Count']:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(int)

        # Sales в известных регионах - numeric и заполнение нулями
        for c in ['NA_Sales', 'EU_Sales', 'Other_Sales']:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)
            else:
                X[c] = 0.0

        # Новая агрегация
        X['Total_Other_Regions'] = X['NA_Sales'] + X['EU_Sales'] + X['Other_Sales']

        # Отбрасываем колонки  уникальны для игры (имя, издателюю) что бы они приводили к переобучению в маленьких датасетах.
        drop_cols = [c for c in ['Name', 'Developer', 'Publisher'] if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)

        return X
