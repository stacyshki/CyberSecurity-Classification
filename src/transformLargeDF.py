from typing import Callable
from pandas.api.types import is_string_dtype, is_numeric_dtype
import re
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from pandas.api.types import is_string_dtype


def get_part_of_day(hour: int) -> str:
    
    """
    Determine the part of the day for a given hour.
    
    Parameters:
    -----------
    hour : int
        An integer representing the hour of the day (0-23).
    
    Returns:
    -----------
    str
        A string indicating the part of the day: 'Night', 'Morning',
        'Afternoon', or 'Evening'.
    """
    
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


def add_date_columns(df: pd.DataFrame, fields: list[str],
                    drop: bool = True) -> pd.DataFrame:
    
    """
    Expand datetime columns into multiple informative date-related columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing datetime columns to expand.
    fields : list[str]
        List of column names to process (must be datetime or convertible to
        datetime).
    drop : bool, default True
        Whether to drop the original datetime columns after processing.
    
    Returns:
    --------
    pd.DataFrame
        The dataframe with new date-related columns such as Year, Month, Day,
        Dayofweek, Dayofyear, Is_month_start, Is_month_end, Is_quarter_start,
        Is_quarter_end, Is_year_start, Is_year_end, Hour, IsWeekend, and
        PartOfDay.
    """
    
    if isinstance(fields, str): 
        fields = [fields]
    
    for field in tqdm(fields):
        fld = df[field]
        fld_dtype = fld.dtype
        
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
    
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[field] = fld = pd.to_datetime(fld, infer_datetime_format = True)
            
        targ_pre = re.sub('[Dd]ate$', '', field)
        attr = [
            'Year', 'Month', 'Day', 
            'Dayofweek', 'Dayofyear',
            'Is_month_start', 'Is_month_end',
            'Is_quarter_start', 'Is_quarter_end',
            'Is_year_start', 'Is_year_end', 'Hour'
        ]
        for n in tqdm(attr): 
            df[targ_pre + n] = getattr(fld.dt, n.lower())
    
    df[field + 'IsWeekend'] = df['Timestamp'].dt.weekday >= 5
    df[field + 'PartOfDay'] = df[field + 'Hour'].apply(
        get_part_of_day
    )
    
    if drop: 
        df.drop(columns = fields, axis = 1, inplace = True)
    return


def changeToCats(df: pd.DataFrame) -> None:
    
    """
    Convert all string/object columns in a dataframe to ordered categorical
    columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to process. Columns of type string/object will 
        be converted inplace.
    
    Returns:
    --------
    None
        Changes are applied inplace.
    """
    
    for n, c in tqdm(df.items()):
        if is_string_dtype(c) or c.dtypes==object: 
            df[n] = c.astype('category').cat.as_ordered()
    return


def fix_missing(df: pd.DataFrame, col: pd.Series, name: str, 
                na_dict: dict) -> dict:
    
    """
    Fill missing numeric values with median and create a missing 
    indicator column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the column.
    col : pd.Series
        The column to process.
    name : str
        Name of the column.
    na_dict : dict
        Dictionary storing fill values for missing data.
    
    Returns:
    --------
    dict
        Updated na_dict including the filler used for this column.
    """
    
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name + '_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


def numericalize(df: pd.DataFrame, col: str, name: str,
                max_n_cat: int | None) -> pd.DataFrame:
    
    """
    Convert a categorical column to integer codes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the column.
    col : pd.Series
        The categorical column to convert.
    name : str
        Name of the column.
    max_n_cat : int or None
        Maximum number of categories to convert. If None, all are converted.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with the categorical column replaced by integer codes.
    """
    
    if (not is_numeric_dtype(col) 
        and (max_n_cat is None or len(col.cat.categories) > max_n_cat)):
        df[name] = pd.Categorical(col).codes + 1
    return df


def process_df(
    df: pd.DataFrame, 
    y_field: str | None = None, 
    skip_flds: list = [],
    ignore_flds: list = [], 
    na_dict: dict = {},
    preproc_fn: Callable = None, 
    max_n_cat = None, 
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    
    """
    Prepare a dataframe for modeling by converting it to numeric, handling
    missing values, and encoding categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
    y_field : str or None, default None
        Name of the target variable column. If provided, it is separated from
        the features.
    skip_flds : list, default []
        List of columns to skip from processing.
    ignore_flds : list, default []
        Columns to retain but not process.
    na_dict : dict, default {}
        Dictionary of pre-defined fill values for missing data.
    preproc_fn : Callable, optional
        Function to apply preprocessing to the dataframe before processing.
    max_n_cat : int or None
        Maximum number of categories to convert to integer codes. None means
        convert all.
    
    Returns:
    --------
    tuple
        df : pd.DataFrame
            The processed dataframe with numeric and encoded features.
        y : np.ndarray or None
            Target values if y_field was provided; otherwise None.
        na_dict : dict
            Dictionary of fill values used for missing data.
    """
    
    df_ignored = df.loc[:, ignore_flds]
    df = df.drop(columns = ignore_flds)
    
    if preproc_fn: 
        preproc_fn(df)
        
    if y_field is None: 
        y = None
    else:
        if not is_numeric_dtype(df[y_field]): 
            df[y_field] = pd.Categorical(df[y_field]).codes
        y = df[y_field].values
        skip_flds += [y_field]
    
    df = df.drop(columns = skip_flds)
    
    na_dict_initial = na_dict.copy()
    for n, c in df.items(): 
        na_dict = fix_missing(df, c, n, na_dict)
    
    if len(na_dict_initial) > 0:
        df = df.drop(
            [a + '_na' for a in 
                list(set(na_dict.keys()) - set(na_dict_initial.keys()))], 
            axis = 1,
        )
    for n,c in df.items(): 
        df = numericalize(df, c, n, max_n_cat)
        
    df = pd.get_dummies(df, dummy_na = True)
    df = pd.concat([df_ignored, df], axis = 1)
    return (df, y, na_dict)
