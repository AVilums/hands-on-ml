from pathlib import Path
from zlib import crc32
from sklearn.model_selection import train_test_split
import logging
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

TEST_RATIO = 0.2

def main() -> int:
    try:
        # load the data
        housing_data = load_housing_data()

        # explore
        explore_df(housing_data)

        # split
        train_set, test_set = train_test_split(housing_data, test_size=TEST_RATIO, random_state=42)

        # categorise income
        housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                            labels=[0, 1, 2, 3, 4])

        cat_counts = housing_data["income_cat"].value_counts().sort_index()
        cat_counts.plot.bar(rot=0, grid=True)
        plt.xlabel("Income category")
        plt.ylabel("Number of districts")
        plt.show()

    except Exception:
        logging.exception("Failed during runtime")
        return 1
    finally:
        return 0


def load_housing_data() -> pd.DataFrame:
    tar_path = Path("dataset/housing.tgz")

    if not tar_path.is_file():
        dataset_path = Path("dataset")
        dataset_path.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path) as tar:
            tar.extractall(path=dataset_path, filter="data")

    return pd.read_csv("dataset/housing/housing.csv")


def explore_df(df: pd.DataFrame) -> None:
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df["ocean_proximity"].value_counts(), '\n')
    print(df.describe(), '\n')
    df.hist(bins=50, figsize=(20, 10))
    plt.show()


def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda x: is_id_in_test_set(x, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

if __name__ == "__main__":
    raise SystemExit(main())