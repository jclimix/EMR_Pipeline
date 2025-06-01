import sys, os, re
import pandas as pd
import numpy as np
from loguru import logger

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helpers import *
from utils.logger_setup import configure_logger

# Swap INFO with DEBUG to preview loaded data
# Leave as INFO to prevent patient data from being logged
logger = configure_logger(f"logs/patient_data_validation.log", "DEBUG")

def validate_patient_id(df):
    """
    Validate the 'patient_id' column to ensure each ID starts with a letter followed by digits.
    Invalid entries are logged and set to NaN.
    """
    pattern = r'^[A-Za-z]\d+$'
    invalid_mask = ~df['patient_id'].astype(str).str.fullmatch(pattern)
    for idx, val in df[invalid_mask]['patient_id'].items():
        logger.warning(f"Invalid value in column 'patient_id' at row {idx}: '{val}' (invalid format)")
        df.at[idx, 'patient_id'] = np.nan
    logger.info("Patient ID validation complete.")

def validate_names(df, columns=['first_name', 'last_name'], banned_words=None):
    """
    Validate name columns to ensure they follow proper capitalization and character rules.
    Flags banned words and invalid/missing values, then replaces them with NaN.
    """
    pattern = r'^[A-ZÀ-ÖØ-Ý][a-zA-Zà-öø-ÿĀ-žĀ-ſ]{1,}$'
    banned_words = set(word.lower() for word in (banned_words or ['invalid', 'dob', 'name', 'firstname', 'lastname']))

    for col in columns:
        col_series = df[col].astype(str)
        for idx, val in col_series.items():
            original_val = val
            val_lower = val.lower().strip()

            if pd.isna(df.at[idx, col]) or val_lower in ['nan', '', 'none']:
                logger.warning(f"Missing name in column '{col}' at row {idx}: '{original_val}' (set to NaN)")
                df.at[idx, col] = np.nan
            elif val_lower in banned_words:
                logger.warning(f"Banned name in column '{col}' at row {idx}: '{original_val}' (replaced with NaN)")
                df.at[idx, col] = np.nan
            elif not re.fullmatch(pattern, val.strip()):
                logger.warning(f"Invalid name format in column '{col}' at row {idx}: '{original_val}' (invalid format)")

        logger.info(f"{col} validation complete.")

def validate_gender(df, column='gender'):
    """
    Validate the 'gender' column to ensure each value is a single-character string (e.g., 'M', 'F').
    Invalid values are logged and set to NaN.
    """
    for idx, val in df[column].items():
        if not isinstance(val, str) or len(val.strip()) != 1:
            logger.warning(f"Invalid gender at row {idx}: '{val}' (not a single-character string)")
            df.at[idx, column] = np.nan
    logger.info("Gender column validation complete.")

def validate_address(df, column='address'):
    """
    Validate the 'address' column to ensure it's a string of at least 5 characters
    and starts with a letter or number. Invalid entries are logged and set to NaN.
    """
    for idx, val in df[column].items():
        if not isinstance(val, str) or len(val.strip()) < 5 or not re.match(r'^[A-Za-z0-9]', val.strip()):
            logger.warning(f"Invalid address at row {idx}: '{val}' (must be string, ≥5 chars, start with letter/number)")
            df.at[idx, column] = np.nan
    logger.info("Address column validation complete.")

def validate_city(df, column='city'):
    """
    Validate the 'city' column to ensure values start with a letter and contain only
    letters, spaces, or hyphens. Invalid entries including 'unknown' are logged and replaced with NaN.
    """
    pattern = r'^[A-Za-z][A-Za-z\s\-]{1,}$'
    for idx, val in df[column].astype(str).items():
        val_lower = val.lower().strip()
        if val_lower == 'unknown' or not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid city at row {idx}: '{val}' (must start with a letter and contain only letters, spaces, or hyphens)")
            df.at[idx, column] = np.nan
    logger.info("City column validation complete.")

def validate_state(df, column='state'):
    """
    Validate the 'state' column to ensure each value is a valid US state abbreviation.
    Non-matching values are logged and set to NaN.
    """
    valid_states = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    }

    for idx, val in df[column].astype(str).str.strip().items():
        if val.upper() not in valid_states:
            logger.warning(f"Invalid state at row {idx}: '{val}' (not a valid US state abbreviation)")
            df.at[idx, column] = np.nan
    logger.info("State column validation complete.")

def validate_zip_code(df, column='zip'):
    """
    Validate and format the 'zip' column to be 5 digits or ZIP+4 format (e.g., 12345 or 12345-6789).
    Pads 4-digit zip codes with leading zeros. Invalid formats are logged and set to NaN.
    """
    pattern = r'^\d{5}(-\d{4})?$'
    df[column] = df[column].astype(str)

    for idx, val in df[column].items():
        val_str = val.strip()

        if re.fullmatch(r'^\d+\.0$', val_str):
            val_str = val_str.split('.')[0]

        if val_str.isdigit() and len(val_str) < 5:
            val_str = val_str.zfill(5)

        if not re.fullmatch(pattern, val_str):
            logger.warning(f"Invalid ZIP code at row {idx}: '{val}' → '{val_str}' (must be 5 digits or ZIP+4 format)")
            val_str = np.nan

        df.at[idx, column] = val_str

    logger.info("ZIP code column validation complete.")

def validate_phone(df, column='phone'):
    """
    Validate and format the 'phone' column into (XXX) XXX-XXXX format.
    Accepts various formats, extracts digits, and reformats where possible.
    Invalid or malformed numbers are logged and replaced with NaN.
    """
    for idx, val in df[column].astype(str).items():
        raw = val.strip()

        if pd.isna(df.at[idx, column]) or raw.lower() in ['nan', '', 'none']:
            continue

        digits = re.sub(r'\D', '', raw)

        if len(digits) == 10:
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            df.at[idx, column] = formatted
        else:
            logger.warning(f"Invalid phone number at row {idx}: '{val}' (could not reformat or invalid length)")
            df.at[idx, column] = np.nan

    logger.info("Phone column validation and formatting complete.")

def validate_insurance_id(df, column='insurance_id'):
    """
    Validate the 'insurance_id' column to ensure each value consists of 3 letters followed by 3 digits.
    Invalid entries are logged and replaced with NaN.
    """
    pattern = r'^[A-Za-z]{3}\d{3}$'

    for idx, val in df[column].astype(str).items():
        raw = val.strip()

        if pd.isna(df.at[idx, column]) or raw.lower() in ['nan', '', 'none']:
            continue

        if not re.fullmatch(pattern, raw):
            logger.warning(f"Invalid insurance ID at row {idx}: '{val}' (must match pattern: 3 letters followed by 3 digits)")
            df.at[idx, column] = np.nan

    logger.info("Insurance ID column validation complete.")

def transform_patient_data(df):
    """
    Run all validation functions on the patient DataFrame and output a cleaned version.
    Saves the staged file to the data/staged/ directory and logs progress or failure.
    """
    try:
        validate_patient_id(df)
        validate_names(df)
        validate_date(df, "date_of_birth")
        validate_gender(df)
        validate_address(df)
        validate_city(df)
        validate_state(df)
        validate_zip_code(df)
        validate_phone(df)
        validate_insurance_id(df)
        validate_date(df, "insurance_effective_date")

        logger.info("Data validation complete.")
        logger.debug("Cleaned DataFrame (preview):")
        logger.debug(df)
        staged_dir = "data/staged/"
        staged_filename = "patient_data_cln.csv"
        staged_path = os.path.join(staged_dir, staged_filename)
        df.to_csv(staged_path, index=False)
        logger.info(f"Data staged to {staged_path}")

    except Exception as e:
        logger.critical(f"Data validation failed unexpectedly: {e}")
        sys.exit(1)

if __name__ == '__main__':
    df = load_csv("patient_data.csv")
    transform_patient_data(df)