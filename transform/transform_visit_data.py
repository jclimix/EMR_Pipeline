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
logger.add("logs/visit_data_validation.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

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
file = "visit_data.csv"
directory = "data/raw/"
df = clean_csv(directory, file)

logger.info("Loaded patient data:")
print(df)

# ---- VALIDATION FUNCTIONS ---- #

def validate_visit_id(df, column='visit_id'):
    pattern = r'^V\d+$'
    for idx, val in df[column].astype(str).items():
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid visit_id at row {idx}: '{val}' (must start with 'V' followed by digits)")
            df.at[idx, column] = np.nan

def validate_provider_id(df, column='provider_id'):
    pattern = r'^PR\d+$'
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none']:
            df.at[idx, column] = np.nan
            continue
        if not re.fullmatch(pattern, val.strip()):
            logger.warning(f"Invalid provider_id at row {idx}: '{val}' (must start with 'PR' followed by digits)")
            df.at[idx, column] = np.nan

def validate_date_column(df, column):
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"]
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

def validate_currency(df, column='currency'):
    valid_currencies = {'USD', 'MXN', 'JPY', 'CAD', 'EUR'}
    for idx, val in df[column].astype(str).str.strip().items():
        if val not in valid_currencies:
            logger.warning(f"Invalid currency at row {idx}: '{val}' (must be a valid 3-letter code)")
            df.at[idx, column] = np.nan

def validate_icd_code(df, column='icd_code'):
    for idx, val in df[column].astype(str).items():
        val = val.strip()
        if val == '' or val.lower() in ['nan', 'none']:
            df.at[idx, column] = np.nan
            continue
        if not re.fullmatch(r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$', val):
            logger.warning(f"Invalid ICD code at row {idx}: '{val}' (not a valid format)")
            df.at[idx, column] = np.nan

def validate_visit_status(df, column='visit_status'):
    valid_statuses = {'Completed', 'Cancelled', 'In Progress', 'Scheduled', 'Open'}
    for idx, val in df[column].astype(str).str.strip().items():
        if val not in valid_statuses:
            logger.warning(f"Invalid visit_status at row {idx}: '{val}' (not in {valid_statuses})")
            df.at[idx, column] = np.nan

def validate_billable_amount(df, column='billable_amount'):
    for idx, val in df[column].astype(str).items():
        raw = val.strip()
        
        # Skip if already missing
        if pd.isna(val) or raw.lower() in ['nan', '', 'none']:
            df.at[idx, column] = np.nan
            continue
        
        try:
            float_val = float(raw)
            df.at[idx, column] = f"{float_val:.2f}"  # Format as 0.00
        except ValueError:
            logger.warning(f"Invalid billable amount at row {idx}: '{val}' (set to NaN)")
            df.at[idx, column] = np.nan

    logger.info("Billable amount column validation and formatting complete.")

def validate_location(df, column='location'):
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none', 'unknown']:
            logger.warning(f"Missing or unknown location at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def validate_reason(df, column='reason_for_visit'):
    for idx, val in df[column].astype(str).items():
        if val.strip().lower() in ['nan', '', 'none']:
            logger.warning(f"Missing reason_for_visit at row {idx}: '{val}'")
            df.at[idx, column] = np.nan

def clean_reason_and_icd_code(df, reason_col='reason_for_visit', icd_col='icd_code'):
    icd_pattern = r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$'

    for idx, val in df[reason_col].astype(str).items():
        parts = [p.strip() for p in val.split(',')]
        reason = parts[0] if parts else ''
        icd_candidate = parts[1] if len(parts) > 1 else ''

        # Update reason
        df.at[idx, reason_col] = reason if reason.lower() not in ['nan', '', 'none'] else np.nan

        # Move valid ICD code to icd_code column
        if re.fullmatch(icd_pattern, icd_candidate):
            df.at[idx, icd_col] = icd_candidate
            logger.info(f"Moved ICD code '{icd_candidate}' to '{icd_col}' from row {idx}")
        elif icd_candidate:
            logger.warning(f"Invalid ICD code fragment in reason at row {idx}: '{icd_candidate}'")

def clean_billable_and_currency(df, bill_col='billable_amount', curr_col='currency'):
    for idx in df.index:
        bill_val = str(df.at[idx, bill_col]).strip()
        curr_val = str(df.at[idx, curr_col]).strip().upper()

        # If billable_amount is a currency and currency is empty or invalid
        if re.fullmatch(r'[A-Z]{3}', bill_val) and curr_val in ['NAN', '', 'NONE']:
            df.at[idx, curr_col] = bill_val
            df.at[idx, bill_col] = np.nan
            logger.warning(f"Swapped values at row {idx}: Moved '{bill_val}' to currency and cleared billable_amount.")

# ---- RUN VALIDATION ---- #

if __name__ == '__main__':
    try:
        clean_reason_and_icd_code(df)
        clean_billable_and_currency(df)
        validate_visit_id(df)
        validate_provider_id(df)
        validate_date_column(df, 'visit_date')
        validate_location(df)
        validate_reason(df)
        validate_icd_code(df)
        validate_visit_status(df)
        validate_billable_amount(df)
        validate_currency(df)
        validate_date_column(df, 'follow_up_date')

        logger.info("Visit data validation complete.")
        print(df)

    except Exception as e:
        logger.critical(f"Visit data validation failed: {e}")
        sys.exit(1)
