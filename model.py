from typing import List, Any, Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class JPRegressor:
    ##
    #numeric_features: список числовых колонок 
    #categorical_features: список категориальных колонок
    ##

    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.transformer: Optional[ColumnTransformer] = None
        self.model: Optional[Any] = None
        self._build()

    def _build(self):
        transformers = []
        if self.numeric_features:
            transformers.append(('num_impute', SimpleImputer(strategy='median'), self.numeric_features))
            transformers.append(('num_scale', StandardScaler(), self.numeric_features))
        if self.categorical_features:
            # OneHotEncoder с handle_unknown='ignore' —  правим для работы с тестом, где могут быть новые категории.
            transformers.append(('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_features))
        self.transformer = ColumnTransformer(transformers=transformers)

        # Для скорости использую HistGradientBoostingRegressor
        try:
            self.model = HistGradientBoostingRegressor(random_state=42)
        except Exception:
            # На некоторых окружениях может не быть HGB; использую RandomForest как fallback.
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        ##
        #Обучает трансформер и модель.
        #Возвращает self для удобства 
        ##
        Xt = self.transformer.fit_transform(X)
        self.model.fit(Xt, y)
        return self

    def predict(self, X):
        ##
        #Выполняет предсказание и обрезает отрицательные значения
        ##
        Xt = self.transformer.transform(X)
        preds = self.model.predict(Xt)
        # Выставил что продажи не могут быть отрицательными
        return np.maximum(preds, 0.0)

    def evaluate(self, X, y):
        #Средняя абсолютная ошибка MAE 
        preds = self.predict(X)
        return mean_absolute_error(y, preds)
