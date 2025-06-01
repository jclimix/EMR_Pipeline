import sys, os, re
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import *
from utils.logger_setup import configure_logger

# Swap INFO with DEBUG to preview loaded data
# Leave as INFO to prevent patient data from being logged
logger = configure_logger(f"logs/lab_results_validation.log", "DEBUG")

def validate_lab_id(df, column="lab_id"):
    """
    Validate the 'lab_id' column to ensure it matches the format 'L####'.
    Invalid entries are logged and replaced with NaN.
    """
    pattern = r"^L\d{4}$"
    for idx, val in df[column].astype(str).items():
        if not re.fullmatch(pattern, val):
            logger.warning(f"Invalid lab ID at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def validate_visit_id(df, column='visit_id'):
    """
    Validate the 'visit_id' column to ensure it matches the format 'V####'.
    Logs missing or invalid entries and sets them to NaN.
    """
    pattern = r'^V\d+$'

    for idx, val in df[column].astype(str).items():
        raw = val.strip()

        if pd.isna(val) or raw.lower() in ['nan', '', 'none']:
            df.at[idx, column] = np.nan
            logger.warning(f"Missing visit ID at row {idx}: '{val}' (set to NaN)")
        elif not re.fullmatch(pattern, raw):
            df.at[idx, column] = np.nan
            logger.warning(f"Invalid visit ID at row {idx}: '{val}' (must start with 'V' followed by digits)")

    logger.info("Visit ID column validation complete.")

def validate_test_name(df, column="test_name"):
    """
    Validate the 'test_name' column to ensure it's not empty or invalid.
    Missing values are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ["", "nan", "none"]:
            logger.warning(f"Missing test name at row {idx}")
            df.at[idx, column] = np.nan

    logger.info("Visit name column validation complete.")

def validate_test_value(df, column='test_value'):
    """
    Validate and clean the 'test_value' column.
    Accepts numeric values or certain text terms (e.g., Positive, Negative).
    Formats numeric values to two decimals and standardizes text. Logs and replaces invalid entries.
    """
    allowed_text = {"positive", "negative", "pending"}

    for idx, val in df[column].astype(str).items():
        val_str = val.strip().lower()

        if pd.isna(val) or val_str in ["", "nan", "none"]:
            df.at[idx, column] = np.nan
            logger.warning(f"Missing test value at row {idx}: '{val}' (set to NaN)")
        elif val_str in allowed_text:
            df.at[idx, column] = val.strip().capitalize()
        else:
            try:
                numeric = float(val)
                df.at[idx, column] = round(numeric, 2)
            except ValueError:
                df.at[idx, column] = np.nan
                logger.warning(f"Invalid test value at row {idx}: '{val}' (not numeric or allowed text — set to NaN)")

    logger.info("Test value column validation complete.")

def validate_test_units(df, column='test_units'):
    """
    Validate the 'test_units' column to ensure units are present for numeric test values.
    Logs and sets units to NaN if missing.
    """
    for idx in df.index:
        val = str(df.at[idx, 'test_value']).strip()
        unit = str(df.at[idx, column]).strip().lower()

        try:
            float(val)
        except ValueError:
            continue  # if not numeric, skip

        if unit in {'', 'nan', 'none'} or pd.isna(df.at[idx, column]):
            logger.warning(f"Missing test unit at row {idx} for numeric value '{val}'")
            df.at[idx, column] = np.nan

    logger.info(f"Test unit validation complete.")

def validate_reference_range(df, column='reference_range'):
    """
    Validate the 'reference_range' column to ensure values are either:
    - A numeric range
    - A known non-numeric term like 'Negative'
    Logs and sets invalid entries to NaN.
    """
    pattern = r'^\d+(\.\d+)?\s*-\s*\d+(\.\d+)?$'
    allowed = {'negative', 'positive', 'pending', 'none', 'nan'}

    for idx, val in df[column].astype(str).items():
        val_clean = val.strip().lower()

        if val_clean in allowed or val_clean == '':
            continue
        if not re.fullmatch(pattern, val_clean):
            logger.warning(f"Invalid reference range at row {idx}: '{val}' (expected format like '11.0-14.0' or 'Negative')")
            df.at[idx, column] = np.nan

    logger.info(f"{column.replace('_', ' ').capitalize()} validation complete.")

def validate_date(df, column):
    """
    Validate and standardize date columns to the format 'YYYY-MM-DD'.
    Accepts multiple input formats. Logs and replaces unrecognized formats with NaN.
    """
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ["", "nan", "none"]:
            df.at[idx, column] = np.nan
            continue
        for fmt in formats:
            try:
                parsed = datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
                df.at[idx, column] = parsed
                break
            except:
                continue
        else:
            logger.warning(f"Invalid date in column '{column}' at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

    logger.info(f"{column.capitalize().replace('_', ' ')} validation complete.")

def transform_lab_data(df):
    """
    Run all validation functions on the lab results DataFrame.
    Outputs the cleaned file to data/staged/ and logs the process.
    """
    try:
        validate_lab_id(df)
        validate_visit_id(df)
        validate_test_name(df)
        validate_test_value(df)
        validate_test_units(df)
        validate_reference_range(df)
        validate_date(df, "date_performed")
        validate_date(df, "date_resulted")

        logger.info("Data validation complete.")
        logger.debug("Cleaned DataFrame (preview):")
        logger.debug(df)
        staged_dir = "data/staged/"
        staged_filename = "lab_results_cln.csv"
        staged_path = os.path.join(staged_dir, staged_filename)
        df.to_csv(staged_path, index=False)
        logger.info(f"Data staged to {staged_path}")

    except Exception as e:
        logger.critical(f"Visit data validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    df = load_csv(f"lab_results.csv")
    transform_lab_data(df)
