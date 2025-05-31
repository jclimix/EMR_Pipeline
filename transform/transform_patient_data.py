import sys, os, re
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Set up Loguru logger
logger.remove()

# File logger (plain format)
logger.add("logs/patient_data_validation.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Terminal logger (bold + colored)
logger.add(sys.stdout, level="INFO", format=(
    "\033[1m<green>{time:YYYY-MM-DD HH:mm:ss}</green>\033[0m | "
    "\033[1m<level>{level}</level>\033[0m | "
    "\033[1m<cyan>{message}</cyan>\033[0m"
))

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.clean_csv import clean_csv

# Load data
file = "patient_data.csv"
directory = "data/raw/"
df = clean_csv(directory, file)

logger.info("Loaded patient data:")
print(df)

# ---- VALIDATION FUNCTIONS ---- #

def validate_patient_id(df):
    pattern = r'^[A-Za-z]\d+$'
    invalid_mask = ~df['patient_id'].astype(str).str.fullmatch(pattern)
    for idx, val in df[invalid_mask]['patient_id'].items():
        logger.warning(f"Invalid value in column 'patient_id' at row {idx}: '{val}' (invalid format)")
        df.at[idx, 'patient_id'] = np.nan
    logger.info("Patient ID validation complete.")

def validate_name_columns(df, columns=['first_name', 'last_name'], banned_words=None):
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

def validate_and_format_dob(df, date_column='date_of_birth'):
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d.%m.%Y", "%d-%m-%Y", "%m.%d.%Y"]

    def try_parse(val, idx):
        for fmt in formats:
            try:
                return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        logger.warning(f"Invalid date_of_birth at row {idx}: '{val}' (unrecognized or invalid date)")
        return np.nan

    df[date_column] = [try_parse(val, idx) for idx, val in df[date_column].astype(str).items()]
    logger.info("Date of birth validation and formatting complete.")

def validate_gender_column(df, column='gender'):
    for idx, val in df[column].items():
        if not isinstance(val, str) or len(val.strip()) != 1:
            logger.warning(f"Invalid gender at row {idx}: '{val}' (not a single-character string)")
            df.at[idx, column] = np.nan
    logger.info("Gender column validation complete.")

def validate_address_column(df, column='address'):
    for idx, val in df[column].items():
        if not isinstance(val, str) or len(val.strip()) < 5 or not re.match(r'^[A-Za-z0-9]', val.strip()):
            logger.warning(f"Invalid address at row {idx}: '{val}' (must be string, ≥5 chars, start with letter/number)")
            df.at[idx, column] = np.nan
    logger.info("Address column validation complete.")

def validate_city_column(df, column='city'):
    pattern = r'^[A-Za-z][A-Za-z\s\-]{1,}$'
    for idx, val in df[column].astype(str).items():
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid city at row {idx}: '{val}' (must start with a letter and contain only letters, spaces, or hyphens)")
            df.at[idx, column] = np.nan
    logger.info("City column validation complete.")

def validate_state_column(df, column='state'):
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

def validate_zip_code_column(df, column='zip'):
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

def validate_phone_column(df, column='phone'):
    pattern = r'^\(\d{3}\)\s\d{3}-\d{4}$'

    for idx, val in df[column].astype(str).items():
        raw = val.strip()

        # Skip if already NaN
        if pd.isna(df.at[idx, column]) or raw.lower() in ['nan', '', 'none']:
            continue

        # Extract digits only
        digits = re.sub(r'\D', '', raw)

        # Format if it's 10 digits
        if len(digits) == 10:
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            df.at[idx, column] = formatted
        else:
            logger.warning(f"Invalid phone number at row {idx}: '{val}' (could not reformat or invalid length)")
            df.at[idx, column] = np.nan

    logger.info("Phone column validation and formatting complete.")

def validate_insurance_id_column(df, column='insurance_id'):
    pattern = r'^[A-Za-z]{3}\d{3}$'

    for idx, val in df[column].astype(str).items():
        raw = val.strip()

        if pd.isna(df.at[idx, column]) or raw.lower() in ['nan', '', 'none']:
            continue

        if not re.fullmatch(pattern, raw):
            logger.warning(f"Invalid insurance ID at row {idx}: '{val}' (must match pattern: 3 letters followed by 3 digits)")
            df.at[idx, column] = np.nan

    logger.info("Insurance ID column validation complete.")

def validate_insurance_effective_date_column(df, column='insurance_effective_date'):
    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",  # ISO-like formats
        "%m/%d/%Y", "%m.%d.%Y",              # U.S. formats
        "%d-%m-%Y", "%d.%m.%Y"               # European formats
    ]

    def try_parse(val, idx):
        if pd.isna(val) or str(val).strip().lower() in ['nan', '', 'none']:
            return np.nan
        for fmt in formats:
            try:
                return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        logger.warning(f"Invalid insurance effective date at row {idx}: '{val}' (unrecognized format)")
        return np.nan

    df[column] = [try_parse(val, idx) for idx, val in df[column].astype(str).items()]
    logger.info("Insurance effective date column validation and formatting complete.")

# ---- RUN VALIDATION ---- #

if __name__ == '__main__':
    try:
        validate_patient_id(df)
        validate_name_columns(df)
        validate_and_format_dob(df)
        validate_gender_column(df)
        validate_address_column(df)
        validate_city_column(df)
        validate_state_column(df)
        validate_zip_code_column(df)
        validate_phone_column(df)
        validate_insurance_id_column(df)
        validate_insurance_effective_date_column(df)

        logger.info("Data validation complete.")
        print("Cleaned DataFrame:")
        print(df)
        df.to_csv('data/staged/patient_data_cln.csv', index=False)

    except Exception as e:
        logger.critical(f"Data validation failed unexpectedly: {e}")
        sys.exit(1)
