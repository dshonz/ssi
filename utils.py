import pandas as pd
from typing import Optional, Iterable


def save_predictions(ids: Optional[Iterable], preds, out_path: str) -> pd.DataFrame:
    ##
    #Сохраню предсказания в CSV в формате:
    #ID,JP_Sales
    #Если ids является None — используется индекс 0..n-1
    ## (тут использовал на первой иттерации разработки. 
    # todo - Проверить на финализации и удали при необходимости)
    df = pd.DataFrame({'ID': list(ids) if ids is not None else list(range(len(preds))),
                       'JP_Sales': preds})
    df.to_csv(out_path, index=False)
    return df
