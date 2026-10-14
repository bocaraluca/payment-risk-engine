from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IEEE_CIS_FEATURE_COLUMNS = (
    "TransactionAmt",
    "TransactionDT",
    "ProductCD",
    "card1",
    "card2",
    "card4",
    "card6",
    "addr1",
    "P_emaildomain",
    "C1",
    "C2",
    "C13",
    "C14",
    "D1",
    "D2",
    "V1",
    "V2",
    "V3",
)
TARGET_COLUMN = "isFraud"


@dataclass(frozen=True, slots=True)
class IEEECISDataset:
    features: pd.DataFrame
    target: pd.Series

    def temporal_split(self, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if not 0 < test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1")

        split_index = int(len(self.features) * (1 - test_fraction))
        return self.features.iloc[:split_index], self.features.iloc[split_index:], self.target.iloc[:split_index], self.target.iloc[split_index:]


def load_ieee_cis_dataset(file_path: Path, nrows: int | None = None) -> IEEECISDataset:
    columns = [TARGET_COLUMN, *IEEE_CIS_FEATURE_COLUMNS]
    dataframe = pd.read_csv(file_path, usecols=columns, nrows=nrows)
    dataframe = dataframe.sort_values("TransactionDT").reset_index(drop=True)
    target = dataframe.pop(TARGET_COLUMN).astype("int8")
    return IEEECISDataset(features=dataframe, target=target)
