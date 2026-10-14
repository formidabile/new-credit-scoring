"""Загрузка собранных признаков и разбиение на dev / hold-out."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import ID_COL, TARGET_COL

RANDOM_STATE = 42


def load_features(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_dev_holdout(
    df: pd.DataFrame,
    ids_path: str | Path,
    holdout_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение. Список ID hold-out сохраняется в файл и при повторных
    вызовах читается оттуда, поэтому разбиение не поплывёт, даже если изменится порядок строк
    или набор признаков.

    dev      - на нём всё: CV, подбор гиперпараметров, выбор порога, калибровка;
    hold-out - трогаем один раз, для финальной оценки.
    """
    ids_path = Path(ids_path)

    if ids_path.exists():
        holdout_ids = pd.read_csv(ids_path)[ID_COL]
    else:
        _, hold = train_test_split(
            df[[ID_COL, TARGET_COL]],
            test_size=holdout_size,
            stratify=df[TARGET_COL],
            random_state=seed,
        )
        holdout_ids = hold[ID_COL].sort_values()
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        holdout_ids.to_csv(ids_path, index=False)

    mask = df[ID_COL].isin(holdout_ids)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
