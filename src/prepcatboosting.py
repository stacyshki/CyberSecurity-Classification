import src.transformLargeDF as tldf
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import Pool
from typing import List


def ready2catboost(data_path: Path, csv_train: str, csv_test: str,
                    cols_to_leave: List[str], date_col, target: str,
                    fillna_target: dict | bool = False,
                    fillnatarget_col: bool = False,
                    sample: bool = False) -> List[str]:
    
    """
    Prepare train and test datasets for CatBoost training.
    This function:
    - Loads train and test CSV files
    - Optionally fills missing target values using a mapping column
    - Converts a date column to datetime and generates additional date features
    - Keeps only selected columns
    - Downcasts integer columns to the smallest possible integer type
        (int8 / int16 / int32 / int64) based on value ranges to reduce
        memory usage
    - Identifies categorical features (object / category dtypes)
    - Optionally samples the data
    - Fills missing categorical values with a placeholder string
    - Saves processed datasets as Feather files for fast I/O
    
    Parameters
    ----------
    data_path : Path
        Base directory containing the input CSV files and where output files
        will be saved.
    csv_train : str
        Filename of the training CSV file.
    csv_test : str
        Filename of the test CSV file.
    cols_to_leave : List[str]
        List of column names to keep in the final datasets.
    date_col : str
        Name of the column containing date information.
    target : str
        Name of the target column.
    fillna_target : bool or dict, optional
        Mapping used to fill missing target values.
        If False, no target imputation is performed.
    fillnatarget_col : str, optional
        Column whose values are used to map missing target values.
        Used only if `fillna_target` is provided.
    sample : bool or int, optional
        If an integer is provided, randomly samples this number of rows
        from both train and test datasets.
    
    Returns
    -------
    List[str]
        List of categorical feature column names (excluding the target),
        suitable for passing to CatBoost as `cat_features`.
    
    Output Files
    ------------
    - catb_train.feather : processed training dataset
    - catb_test.feather : processed test dataset
    
    Examples
    --------
    >>> from pathlib import Path
    >>> data_path = Path("data/")
    >>> cols = [
    ...     "user_id", "product_id", "city",
    ...     "created_at", "price", "target"
    ... ]
    >>>
    >>> cat_features = ready2catboost(
    ...     data_path=data_path,
    ...     csv_train="train.csv",
    ...     csv_test="test.csv",
    ...     cols_to_leave=cols,
    ...     date_col="created_at",
    ...     target="target",
    ...     fillna_target={0: 0, 1: 1},
    ...     fillnatarget_col="product_id",
    ...     sample=100_000
    ... )
    >>>
    >>> cat_features
    ['user_id', 'product_id', 'city']
    """
    
    catb_train = pd.read_csv(data_path / csv_train, low_memory=False)
    catb_test = pd.read_csv(data_path / csv_test, low_memory=False)
    
    if fillna_target:
        catb_train[target] = catb_train[target].fillna(
            catb_train[fillnatarget_col].map(fillna_target)
        )
        catb_train.dropna(subset=[target], inplace=True)
        
        catb_test[target] = catb_test[target].fillna(
            catb_test[fillnatarget_col].map(fillna_target)
        )
        catb_test.dropna(subset=[target], inplace=True)
    
    catb_train[date_col] = pd.to_datetime(catb_train[date_col])
    catb_test[date_col] = pd.to_datetime(catb_test[date_col])
    
    catb_train = catb_train[cols_to_leave]
    catb_test = catb_test[cols_to_leave]
    
    tldf.add_date_columns(catb_train, date_col)
    tldf.add_date_columns(catb_test, date_col)
    
    cols_int64 = catb_train.dtypes[catb_train.dtypes == 'int64'].index
    possible_integers = pd.Series([8, 16, 32, 64])
    
    for i in cols_int64:
        value_max_log = np.log2(catb_train[i].max())
        value_min_log = np.log2(abs(catb_train[i].min()) + 1e-6)
        biggest_log = max(value_max_log, value_min_log)
        transform_to = possible_integers[possible_integers > biggest_log]
        catb_train[i] = catb_train[i].astype(f'int{transform_to.min()}')
    
    cols_int64 = catb_test.dtypes[catb_test.dtypes == 'int64'].index
    possible_integers = pd.Series([8, 16, 32, 64])
    
    for i in cols_int64:
        value_max_log = np.log2(catb_test[i].max())
        value_min_log = np.log2(abs(catb_test[i].min()) + 1e-6)
        biggest_log = max(value_max_log, value_min_log)
        transform_to = possible_integers[possible_integers > biggest_log]
        catb_test[i] = catb_test[i].astype(f'int{transform_to.min()}')
    
    cols_int32 = catb_train.dtypes[catb_train.dtypes == 'int32'].index
    possible_integers = pd.Series([8, 16, 32])
    
    for i in cols_int32:
        value_max_log = np.log2(catb_train[i].max())
        value_min_log = np.log2(abs(catb_train[i].min()) + 1e-6)
        biggest_log = max(value_max_log, value_min_log)
        transform_to = possible_integers[possible_integers > biggest_log]
        catb_train[i] = catb_train[i].astype(f'int{transform_to.min()}')
    
    cols_int32 = catb_test.dtypes[catb_test.dtypes == 'int32'].index
    possible_integers = pd.Series([8, 16, 32])
    
    for i in cols_int32:
        value_max_log = np.log2(catb_test[i].max())
        value_min_log = np.log2(abs(catb_test[i].min()) + 1e-6)
        biggest_log = max(value_max_log, value_min_log)
        transform_to = possible_integers[possible_integers > biggest_log]
        catb_test[i] = catb_test[i].astype(f'int{transform_to.min()}')
    
    cat_features = catb_train.select_dtypes(
                include=['object', 'category']).columns
    
    cat_features = cat_features[~cat_features.isin([target])].tolist()
    
    if sample:
        catb_train = catb_train.sample(n=sample, random_state=42)
        catb_test = catb_test.sample(n=sample, random_state=42)
    
    for i in cat_features:
        catb_train[i] = catb_train[i].fillna('__NA__')
        catb_test[i] = catb_test[i].fillna('__NA__')
    
    catb_train.to_feather(data_path / 'catb_train.feather')
    catb_test.to_feather(data_path / 'catb_test.feather')
    
    return cat_features


def toPool(catb_train: pd.DataFrame, catb_test: pd.DataFrame, target: str,
            cat_features: List[str]) -> tuple[Pool, Pool, pd.Series]:
    
    """
    Convert prepared pandas DataFrames into CatBoost Pool objects.
    This function:
    - Splits features and target from train and test DataFrames
    - Creates CatBoost `Pool` objects for training and evaluation
    - Preserves categorical feature indices via `cat_features`
    - Returns test target separately for downstream evaluation
    
    Parameters
    ----------
    catb_train : pd.DataFrame
        Prepared training dataset containing features and target column.
    catb_test : pd.DataFrame
        Prepared test dataset containing features and target column.
    target : str
        Name of the target column.
    cat_features : List[str]
        List of categorical feature column names.
    
    Returns
    -------
    train_data : Pool
        CatBoost Pool object for training.
    eval_data : Pool
        CatBoost Pool object for validation / evaluation.
    y_test : pd.Series
        Target values from the test dataset.
    
    Examples
    --------
    >>> from catboost import Pool
    >>> import pandas as pd
    >>> catb_train = pd.DataFrame({
    ...     "user_id": ["u1", "u2", "u3"],
    ...     "price": [100, 200, 150],
    ...     "target": [1, 0, 1]
    ... })
    >>> catb_test = pd.DataFrame({
    ...     "user_id": ["u4", "u5"],
    ...     "price": [120, 180],
    ...     "target": [0, 1]
    ... })
    >>> cat_features = ["user_id"]
    >>> train_pool, eval_pool, y_test = toPool(
    ...     catb_train=catb_train,
    ...     catb_test=catb_test,
    ...     target="target",
    ...     cat_features=cat_features
    ... )
    """
    
    X_train = catb_train.drop(columns=[target])
    y_train = catb_train[target]
    X_test = catb_test.drop(columns=[target])
    y_test = catb_test[target]
    
    train_data = Pool(X_train, y_train, cat_features=cat_features)
    eval_data = Pool(X_test, y_test, cat_features=cat_features)
    
    return train_data, eval_data, y_test