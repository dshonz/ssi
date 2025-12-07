import os
from data_load import DataLoader
from preproc import FeatureEngineer
from model import JPRegressor
from utils import save_predictions
from sklearn.model_selection import train_test_split
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_TRAIN = os.path.join(DEFAULT_DATA_DIR, 'Video_Games.csv')
DEFAULT_TEST = os.path.join(DEFAULT_DATA_DIR, 'Video_Games_Test.csv')
DEFAULT_SAMPLE = os.path.join(DEFAULT_DATA_DIR, 'Video_Games_Sample.csv')
DEFAULT_OUT = os.path.join(DEFAULT_DATA_DIR, 'predictions.csv')


def run_pipeline(train_csv: str = None, test_csv: str = None, sample_csv: str = None, out_path: str = None):
    train_csv = train_csv or DEFAULT_TRAIN
    test_csv = test_csv or DEFAULT_TEST
    sample_csv = sample_csv or DEFAULT_SAMPLE
    out_path = out_path or DEFAULT_OUT

    loader = DataLoader(train_csv, test_csv, sample_csv)
    train_df = loader.load_train()
    test_df = loader.load_test()
    print (train_df)
    print (test_df)
    if 'JP_Sales' not in train_df.columns:
        raise KeyError("Target column 'JP_Sales' is missing in training data")

    X = train_df.drop(columns=['JP_Sales'])
    y = train_df['JP_Sales']

    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X)
    test_fe = fe.transform(test_df)

    # определить числовые и категориальные колонки автоматически
    numeric_features = [c for c in X_fe.columns if X_fe[c].dtype.kind in 'fi']
    categorical_features = [c for c in X_fe.columns if c not in numeric_features]

    # Быстрая валидация 80 yf 20 
    X_train, X_val, y_train, y_val = train_test_split(X_fe, y, test_size=0.2, random_state=42)

    model = JPRegressor(numeric_features=numeric_features, categorical_features=categorical_features)
    model.fit(X_train, y_train)
    val_mae = model.evaluate(X_val, y_val)
    print(f"Validation MAE: {val_mae:.6f}")

    preds = model.predict(test_fe)

    # Используем колонку ID если CSV содержит её
    sample = loader.load_sample()
    ids = None
    if sample is not None and 'ID' in sample.columns:
        ids = sample['ID']

    out_df = save_predictions(ids, preds, out_path)
    print(f"Predictions saved to {out_path}")

    return {'mae': val_mae, 'predictions_df': out_df}


if __name__ == '__main__':
    run_pipeline()
