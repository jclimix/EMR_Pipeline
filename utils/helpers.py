import pandas as pd
import os
from datetime import datetime
from io import StringIO
from loguru import logger
import numpy as np

directory = f"data/raw/"

def is_invalid_value(val):
    """
    Helper function to check if a value is considered invalid.
    Returns True if the value is None, NaN, empty string, or contains 'invalid'.
    """
    if pd.isna(val):
        return True
    val_str = str(val).lower().strip()
    return val_str in ['', 'nan', 'none', 'null', 'invalid']

def clean_csv(directory, file):
    """
    Cleans and loads a CSV file with potentially malformed quotes.
    Strips outer double quotes and replaces double double-quotes ("") with single quotes.
    Returns the cleaned data as a pandas DataFrame.
    """
    if os.path.isfile(os.path.join(directory, file)):
        with open(directory + file, 'r', encoding='utf-8') as f:
            cleaned = '\n'.join(
                line.strip()[1:-1].replace('""', '"') if line.strip().startswith('"') and line.strip().endswith('"')
                else line.strip().replace('""', '"')
                for line in f if line.strip()
            )

        df = pd.read_csv(StringIO(cleaned))
        return df

def load_csv(file):
    """
    Loads a cleaned CSV file from the raw data directory.
    Uses clean_csv to preprocess the file and logs a preview of the DataFrame.
    """
    directory = "data/raw/"
    df = clean_csv(directory, file)
    logger.debug(f"Data Loaded (Preview):\n{df}")
    return df

def validate_date(df, column):
    """
    Validate and standardize a date column to 'YYYY-MM-DD' format.
    Attempts multiple formats in-place. Logs and sets unrecognized dates to NaN.
    """
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%m/%d/%Y", "%m.%d.%Y", "%m-%d-%Y",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"
    ]

    cleaned = []
    for idx, val in df[column].items():
        val_str = str(val).strip()
        parsed = None

        if val_str.lower() in {"", "nan", "none"}:
            cleaned.append(np.nan)
        else:
            for fmt in formats:
                try:
                    parsed = datetime.strptime(val_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            cleaned.append(parsed if parsed else np.nan)
            if not parsed:
                logger.warning(f"Invalid date in column '{column}' at row {idx}: '{val}' (unrecognized format)")

    df[column] = cleaned
    logger.info(f"{column.replace('_', ' ').capitalize()} validation complete.")
