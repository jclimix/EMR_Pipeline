import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(Path(__file__).parent.parent))

# Import ETL modules
from extract.xlsx_to_csv import extract_excel_data
from transform.patient_transform import transform_patient_data
from transform.visit_transform import transform_visit_data
from transform.lab_transform import transform_lab_data
from transform.icd_transform import transform_icd_data
from load.load_data_to_db import main as load_to_db
from utils.logger_setup import configure_logger
from utils.helpers import *

# logging setup
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'pipeline_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
logger = configure_logger(str(log_file))

def run_pipeline():
    """Main pipeline function to run the ETL process."""
    try:
        logger.info("Starting ETL pipeline.")
        
        # Extract: Convert XLSX to CSVs
        logger.info("Starting extraction phase.")
        extract_excel_data(f"data/source/Data Eng Data Set.xlsx")
        logger.info("Extraction completed successfully.")
        
        # Transform: Process and transform each dataset
        logger.info("Starting transformation phase.")
        
        patient_data_df = load_csv("patient_data.csv")
        transform_patient_data(patient_data_df)
        visit_data_df = load_csv("visit_data.csv")
        transform_visit_data(visit_data_df)
        lab_results_df = load_csv("lab_results.csv")
        transform_lab_data(lab_results_df)
        icd_ref_df = load_csv("icd_reference.csv")
        transform_icd_data(icd_ref_df)
        logger.info("Transformation completed successfully.")
        
        # Load: Load transformed data into SQL database
        logger.info("Starting load phase.")
        load_to_db()
        logger.info("Load completed successfully.")
        
        logger.info("ETL pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
