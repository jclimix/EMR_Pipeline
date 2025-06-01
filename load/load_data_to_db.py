import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

def create_database():

    conn = sqlite3.connect('data/final/emr_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS icd_reference (
        icd_code TEXT PRIMARY KEY,
        description TEXT,
        effective_date DATE,
        status TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        date_of_birth DATE,
        gender TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        zip TEXT,
        phone TEXT,
        insurance_id TEXT,
        insurance_effective_date DATE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS visits (
        visit_id TEXT PRIMARY KEY,
        patient_id TEXT,
        provider_id TEXT,
        visit_date DATE,
        location TEXT,
        reason_for_visit TEXT,
        icd_code TEXT,
        visit_status TEXT,
        billable_amount REAL,
        currency TEXT,
        follow_up_date DATE,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY (icd_code) REFERENCES icd_reference(icd_code)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lab_results (
        lab_id TEXT PRIMARY KEY,
        visit_id TEXT,
        test_name TEXT,
        test_value TEXT,
        test_units TEXT,
        reference_range TEXT,
        date_performed DATE,
        date_resulted DATE,
        FOREIGN KEY (visit_id) REFERENCES visits(visit_id)
    )
    ''')
    
    return conn

def load_csv_to_table(conn, csv_path, table_name):

    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)

def main():

    conn = create_database()
    data_dir = Path('data/staged')
    try:
        load_csv_to_table(conn, data_dir / 'icd_reference_cln.csv', 'icd_reference')
        load_csv_to_table(conn, data_dir / 'patient_data_cln.csv', 'patients')
        load_csv_to_table(conn, data_dir / 'visit_data_cln.csv', 'visits')
        load_csv_to_table(conn, data_dir / 'lab_results_cln.csv', 'lab_results')
        
        logger.info("Data successfully loaded into the database!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main() 