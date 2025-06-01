import sys, os, re
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import *
from utils.logger_setup import configure_logger

logger = configure_logger("logs/icd_reference_validation.log", "DEBUG")

def is_null_like(val):
    """
    Check if a value is considered null-like.
    Returns True for NaN, empty strings, or common null indicators (e.g., 'nan', 'none', 'null').
    """
    return (
        pd.isna(val)
        or str(val).strip().lower() in {'', 'nan', 'none', 'null'}
    )

def validate_icd_code(df, column="icd_code"):
    """
    Validate values in the specified ICD code column using a regex pattern.
    Valid ICD codes must start with a letter, followed by two digits, and an optional decimal portion.
    Invalid entries are logged and set to NaN.
    """
    pattern = r'^[A-Z]\d{2}(\.\d+)?$'
    for idx, val in df[column].astype(str).items():
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid ICD code at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def validate_description(df, column="description"):
    """
    Validate the description column to ensure non-null, non-empty values.
    Null-like entries are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        if is_null_like(val):
            logger.warning(f"Missing or invalid description at row {idx}")
            df.at[idx, column] = np.nan

def validate_status(df, column="status"):
    """
    Validate the status column to ensure values are either 'active' or 'inactive' (case-insensitive).
    Standardizes valid values to capitalized format. Invalid entries are logged and set to NaN.
    """
    valid_statuses = {"active", "inactive"}
    for idx, val in df[column].astype(str).items():
        status = val.strip().lower()
        if status not in valid_statuses:
            logger.warning(f"Invalid status at row {idx}: '{val}'")
            df.at[idx, column] = np.nan
        else:
            df.at[idx, column] = status.capitalize()

def transform_icd_data(df):
    """
    Run all validation functions on the ICD reference DataFrame.
    Logs results, formats data, and saves the cleaned output to the staging directory.
    """
    try:
        validate_icd_code(df)
        validate_description(df)
        validate_date(df, "effective_date")
        validate_status(df)

        logger.info("Data validation complete.")
        logger.debug("Cleaned DataFrame (preview):")
        logger.debug(df)
        staged_dir = "data/staged/"
        staged_filename = "icd_reference_cln.csv"
        staged_path = os.path.join(staged_dir, staged_filename)
        df.to_csv(staged_path, index=False)
        logger.info(f"Data staged to {staged_path}")

    except Exception as e:
        logger.critical(f"ICD reference validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    df = load_csv("icd_reference.csv")
    transform_icd_data(df)