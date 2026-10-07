from pathlib import Path
import logging
import tarfile
import urllib.request
import pandas as pd

logging.basicConfig(level=logging.INFO)

def main() -> int:
    try:
        # load the data
        housing_data = load_housing_data()

        # explore
        explore_df(housing_data)


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


if __name__ == "__main__":
    raise SystemExit(main())