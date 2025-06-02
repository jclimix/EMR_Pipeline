import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
from transform.visit_transform import (
    validate_visit_id,
    validate_provider_id,
    validate_date,
    validate_currency,
    validate_icd_code,
    validate_visit_status,
    validate_billable_amount,
    validate_location,
    validate_reason,
    clean_reason_and_icd_code,
    clean_billable_and_currency
)

@pytest.fixture
def sample_visit_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'visit_id': ['V123', 'V456', 'invalid', 'V789'],
        'provider_id': ['PR123', 'PR456', 'invalid', ''],
        'visit_date': ['2024-01-01', '01/15/2024', 'invalid', ''],
        'location': ['Main Clinic', 'Urgent Care', 'unknown', ''],
        'reason_for_visit': ['Annual Checkup', 'Fever, A12.3', 'Appt.', ''],
        'icd_code': ['A12.3', 'B45.2', 'R345', None],
        'visit_status': ['Completed', 'In Progress', 'invalid', ''],
        'billable_amount': ['150.00', 'USD', 'EUR', None],
        'currency': ['USD', '', None, ''],
        'follow_up_date': ['2024-02-01', '02/15/2024', 'invalid', '']
    })

def test_validate_visit_id(sample_visit_df):
    validate_visit_id(sample_visit_df)
    assert pd.isna(sample_visit_df.loc[2, 'visit_id'])
    assert sample_visit_df.loc[0, 'visit_id'] == 'V123'
    assert sample_visit_df.loc[1, 'visit_id'] == 'V456'
    assert sample_visit_df.loc[3, 'visit_id'] == 'V789'

def test_validate_provider_id(sample_visit_df):
    validate_provider_id(sample_visit_df)
    assert pd.isna(sample_visit_df.loc[2, 'provider_id'])
    assert pd.isna(sample_visit_df.loc[3, 'provider_id'])
    assert sample_visit_df.loc[0, 'provider_id'] == 'PR123'
    assert sample_visit_df.loc[1, 'provider_id'] == 'PR456'

def test_validate_date(sample_visit_df):
    validate_date(sample_visit_df, 'visit_date')
    validate_date(sample_visit_df, 'follow_up_date')
    
    assert sample_visit_df.loc[0, 'visit_date'] == '2024-01-01'
    assert sample_visit_df.loc[1, 'visit_date'] == '2024-01-15'
    assert pd.isna(sample_visit_df.loc[2, 'visit_date'])
    assert pd.isna(sample_visit_df.loc[3, 'visit_date'])
    
    assert sample_visit_df.loc[0, 'follow_up_date'] == '2024-02-01'
    assert sample_visit_df.loc[1, 'follow_up_date'] == '2024-02-15'
    assert pd.isna(sample_visit_df.loc[2, 'follow_up_date'])
    assert pd.isna(sample_visit_df.loc[3, 'follow_up_date'])

def test_validate_currency(sample_visit_df):
    validate_currency(sample_visit_df)
    assert sample_visit_df.loc[0, 'currency'] == 'USD'
    assert pd.isna(sample_visit_df.loc[1, 'currency'])
    assert pd.isna(sample_visit_df.loc[2, 'currency'])
    assert pd.isna(sample_visit_df.loc[3, 'currency'])

def test_validate_icd_code(sample_visit_df):
    validate_icd_code(sample_visit_df)
    assert sample_visit_df.loc[0, 'icd_code'] == 'A12.3'
    assert sample_visit_df.loc[1, 'icd_code'] == 'B45.2'
    assert pd.isna(sample_visit_df.loc[2, 'icd_code'])
    assert pd.isna(sample_visit_df.loc[3, 'icd_code'])

def test_validate_visit_status(sample_visit_df):
    validate_visit_status(sample_visit_df)
    assert sample_visit_df.loc[0, 'visit_status'] == 'Completed'
    assert sample_visit_df.loc[1, 'visit_status'] == 'In Progress'
    assert pd.isna(sample_visit_df.loc[2, 'visit_status'])
    assert pd.isna(sample_visit_df.loc[3, 'visit_status'])

def test_validate_billable_amount(sample_visit_df):
    validate_billable_amount(sample_visit_df)
    assert sample_visit_df.loc[0, 'billable_amount'] == '150.00'
    assert pd.isna(sample_visit_df.loc[1, 'billable_amount'])  
    assert pd.isna(sample_visit_df.loc[2, 'billable_amount'])
    assert pd.isna(sample_visit_df.loc[3, 'billable_amount'])

def test_validate_location(sample_visit_df):
    validate_location(sample_visit_df)
    assert sample_visit_df.loc[0, 'location'] == 'Main Clinic'
    assert sample_visit_df.loc[1, 'location'] == 'Urgent Care'
    assert pd.isna(sample_visit_df.loc[2, 'location'])  
    assert pd.isna(sample_visit_df.loc[3, 'location'])  

def test_validate_reason(sample_visit_df):
    validate_reason(sample_visit_df)
    assert sample_visit_df.loc[0, 'reason_for_visit'] == 'Annual Checkup'
    assert sample_visit_df.loc[1, 'reason_for_visit'] == 'Fever, A12.3'
    assert sample_visit_df.loc[2, 'reason_for_visit'] == 'Appt.'
    assert pd.isna(sample_visit_df.loc[3, 'reason_for_visit'])

def test_clean_reason_and_icd_code(sample_visit_df):
    clean_reason_and_icd_code(sample_visit_df)
    assert sample_visit_df.loc[0, 'reason_for_visit'] == 'Annual Checkup'
    assert sample_visit_df.loc[1, 'reason_for_visit'] == 'Fever'
    assert sample_visit_df.loc[2, 'reason_for_visit'] == 'Appt.'
    assert pd.isna(sample_visit_df.loc[3, 'reason_for_visit'])
    
    assert sample_visit_df.loc[0, 'icd_code'] == 'A12.3'
    assert sample_visit_df.loc[1, 'icd_code'] == 'A12.3'  
    assert sample_visit_df.loc[2, 'icd_code'] == 'R345'
    assert pd.isna(sample_visit_df.loc[3, 'icd_code'])

def test_clean_billable_and_currency(sample_visit_df):
    clean_billable_and_currency(sample_visit_df)
    assert sample_visit_df.loc[0, 'billable_amount'] == '150.00'
    assert pd.isna(sample_visit_df.loc[1, 'billable_amount'])  
    assert pd.isna(sample_visit_df.loc[2, 'billable_amount'])
    assert pd.isna(sample_visit_df.loc[3, 'billable_amount'])
    
    assert sample_visit_df.loc[0, 'currency'] == 'USD'
    assert sample_visit_df.loc[1, 'currency'] == 'USD'
    assert sample_visit_df.loc[2, 'currency'] == 'EUR'
    assert pd.isna(sample_visit_df.loc[3, 'currency'])