
from typing import Optional
import pandas as pd
import os

class DataLoader:
    def __init__(self, train_path: str, test_path: str, sample_path: Optional[str] = None):
        self.train_path = train_path
        self.test_path = test_path
        self.sample_path = sample_path

    def load_train(self) -> pd.DataFrame:
        #Загрузить обучающую выборку. FileNotFoundError при отсутствии файла.
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train file not found at {self.train_path}")
        return pd.read_csv(self.train_path)

    def load_test(self) -> pd.DataFrame:
        #Загрузить тестовую выборку. FileNotFoundError при отсутствии файла.
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test file not found at {self.test_path}")
        return pd.read_csv(self.test_path)

    def load_sample(self) -> Optional[pd.DataFrame]:
        #Загрузить пример выходного файла (если указан)
        if self.sample_path and os.path.exists(self.sample_path):
            return pd.read_csv(self.sample_path)
        return None
