from pathlib import Path

import joblib
import pandas as pd


def load_pre_split_dataset(logger, dataset_train_path, maps_path):
    """
    Load the already materialized training split and its corresponding maps.

    Parameters
    ----------
    logger : logging.Logger
        Logger used for status messages.
    dataset_train_path : str or Path
        Filesystem location of the preprocessed training CSV created by the data_split pipeline.
    maps_path : str or Path
        Filesystem location of the serialized maps saved by the data_split pipeline.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame containing the training split and the decoded maps.
    """

    dataset_train_path = Path(dataset_train_path)
    maps_path = Path(maps_path)

    if not dataset_train_path.is_file():
        raise FileNotFoundError(f"Training split not found at: {dataset_train_path}")

    if not maps_path.is_file():
        raise FileNotFoundError(f"Maps artifact not found at: {maps_path}")

    logger.info(f"Loading pre-split training dataset from: {dataset_train_path}")
    df_train = pd.read_csv(dataset_train_path)

    logger.info(f"Loading preprocessing maps from: {maps_path}")
    maps = joblib.load(maps_path)

    logger.info(
        "Loaded pre-split dataset (%d rows, %d columns) and maps object with keys: %s",
        len(df_train),
        len(df_train.columns),
        list(maps.keys()) if isinstance(maps, dict) else type(maps),
    )

    return df_train, maps
