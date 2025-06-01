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
logger = configure_logger(f"logs/visit_data_validation.log", "DEBUG")

def validate_visit_id(df, column='visit_id'):
    """
    Validate the 'visit_id' column to ensure each value starts with 'V' followed by digits.
    Invalid entries are logged and set to NaN.
    """
    pattern = r'^V\d+$'
    for idx, val in df[column].astype(str).items():
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid visit_id at row {idx}: '{val}' (must start with 'V' followed by digits)")
            df.at[idx, column] = np.nan

def validate_provider_id(df, column='provider_id'):
    """
    Validate the 'provider_id' column to ensure each value starts with 'PR' followed by digits.
    Invalid or empty entries are logged and set to NaN.
    """
    pattern = r'^PR\d+$'
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none']:
            df.at[idx, column] = np.nan
            continue
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid provider_id at row {idx}: '{val}' (must start with 'PR' followed by digits)")
            df.at[idx, column] = np.nan

def validate_date(df, column):
    """
    Validate and standardize a date column to 'YYYY-MM-DD' format.
    Accepts a variety of common date formats. Logs and sets to NaN if parsing fails.
    """
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%m-%d-%Y"]
    def try_parse(val, idx):
        if str(val).strip().lower() in ['nan', '', 'none']:
            return np.nan
        for fmt in formats:
            try:
                return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        logger.warning(f"Invalid date in column '{column}' at row {idx}: '{val}' (unrecognized format)")
        return np.nan
    df[column] = [try_parse(val, idx) for idx, val in df[column].astype(str).items()]
    logger.info(f"{column.capitalize().replace('_', ' ')} validation complete.")

def validate_currency(df, column='currency'):
    """
    Validate the 'currency' column to ensure values are valid 3-letter currency codes.
    Invalid values are logged and set to NaN.
    """
    valid_currencies = {'USD', 'MXN', 'JPY', 'CAD', 'EUR'}
    for idx, val in df[column].astype(str).str.strip().items():
        if val not in valid_currencies:
            logger.warning(f"Invalid currency at row {idx}: '{val}' (must be a valid 3-letter code)")
            df.at[idx, column] = np.nan

def validate_icd_code(df, column='icd_code'):
    """
    Validate the 'icd_code' column to ensure ICD format: a letter, 2 digits, optional dot and suffix.
    Invalid entries are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        val = val.strip()
        if val == '' or val.lower() in ['nan', 'none']:
            df.at[idx, column] = np.nan
            continue
        if not re.fullmatch(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', val):
            logger.warning(f"Invalid ICD code at row {idx}: '{val}' (not a valid format)")
            df.at[idx, column] = np.nan

def validate_visit_status(df, column='visit_status'):
    """
    Validate the 'visit_status' column to ensure each value matches known visit statuses.
    Invalid values are logged and set to NaN.
    """
    valid_statuses = {'Completed', 'Cancelled', 'In Progress', 'Scheduled', 'Open'}
    for idx, val in df[column].astype(str).str.strip().items():
        if val not in valid_statuses:
            logger.warning(f"Invalid visit_status at row {idx}: '{val}' (not in {valid_statuses})")
            df.at[idx, column] = np.nan

def validate_billable_amount(df, column='billable_amount'):
    """
    Validate and format the 'billable_amount' column to two decimal places (as a string).
    Invalid or non-numeric entries are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        raw = val.strip()
        
        if pd.isna(val) or raw.lower() in ['nan', '', 'none']:
            df.at[idx, column] = np.nan
            continue
        
        try:
            float_val = float(raw)
            df.at[idx, column] = f"{float_val:.2f}"
        except ValueError:
            logger.warning(f"Invalid billable amount at row {idx}: '{val}' (set to NaN)")
            df.at[idx, column] = np.nan

    logger.info("Billable amount column validation and formatting complete.")

def validate_location(df, column='location'):
    """
    Validate the 'location' column to ensure it's not missing, unknown, or blank.
    Invalid entries are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none', 'unknown']:
            logger.warning(f"Missing or unknown location at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def validate_reason(df, column='reason_for_visit'):
    """
    Validate the 'reason_for_visit' column to ensure it is not empty or invalid.
    Missing values are logged and set to NaN.
    """
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none']:
            logger.warning(f"Missing reason_for_visit at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def clean_reason_and_icd_code(df, reason_col='reason_for_visit', icd_col='icd_code'):
    """
    Parse and separate concatenated values from 'reason_for_visit' into valid reason and ICD code.
    Moves valid ICD code to the 'icd_code' column and logs actions or warnings.
    """
    icd_pattern = r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$'

    for idx, val in df[reason_col].astype(str).items():
        parts = [p.strip() for p in val.split(',')]
        reason = parts[0] if parts else ''
        icd_candidate = parts[1] if len(parts) > 1 else ''

        df.at[idx, reason_col] = reason if reason.lower() not in ['nan', '', 'none'] else np.nan

        if re.fullmatch(icd_pattern, icd_candidate):
            df.at[idx, icd_col] = icd_candidate
            logger.info(f"Moved ICD code '{icd_candidate}' to '{icd_col}' from row {idx}")
        elif icd_candidate:
            logger.warning(f"Invalid ICD code fragment in reason at row {idx}: '{icd_candidate}'")

def clean_billable_and_currency(df, bill_col='billable_amount', curr_col='currency'):
    """
    Detect and correct cases where a currency value was mistakenly placed in the 'billable_amount' column.
    Swaps values if a 3-letter currency code is found in the wrong column.
    """
    for idx in df.index:
        bill_val = str(df.at[idx, bill_col]).strip()
        curr_val = str(df.at[idx, curr_col]).strip().upper()

        if re.fullmatch(r'[A-Z]{3}', bill_val) and curr_val in ['NAN', '', 'NONE']:
            df.at[idx, curr_col] = bill_val
            df.at[idx, bill_col] = np.nan
            logger.warning(f"Swapped values at row {idx}: Moved '{bill_val}' to currency and cleared billable_amount.")

def transform_visit_data(df):
    """
    Run all validation and cleaning functions on the visit dataset.
    Outputs a cleaned CSV file to the data/staged/ directory and logs the process.
    """
    try:
        clean_reason_and_icd_code(df)
        clean_billable_and_currency(df)
        validate_visit_id(df)
        validate_provider_id(df)
        validate_date(df, 'visit_date')
        validate_location(df)
        validate_reason(df)
        validate_icd_code(df)
        validate_visit_status(df)
        validate_billable_amount(df)
        validate_currency(df)
        validate_date(df, 'follow_up_date')

        logger.info("Data validation complete.")
        logger.debug("Cleaned DataFrame (preview):")
        logger.debug(df)
        staged_dir = "data/staged/"
        staged_filename = "visit_data_cln.csv"
        staged_path = os.path.join(staged_dir, staged_filename)
        df.to_csv(staged_path, index=False)
        logger.info(f"Data staged to {staged_path}")

    except Exception as e:
        logger.critical(f"Visit data validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    df = load_csv(f"visit_data.csv")
    transform_visit_data(df)