import pytest
import pandas as pd
import numpy as np
import os
from transform.patient_transform import (
    validate_patient_id,
    validate_names,
    validate_gender,
    validate_address,
    validate_city,
    validate_state,
    validate_zip_code,
    validate_phone,
    validate_insurance_id
)

@pytest.fixture
def sample_patient_df():
    """Create a sample DataFrame for testing patient validation functions"""
    return pd.DataFrame({
        'patient_id': ['A123', 'B456', '123', 'invalid'],
        'first_name': ['John', 'Mary', 'invalid', 'test'],
        'last_name': ['Smith', 'Johnson', 'invalid', 'test'],
        'gender': ['M', 'F', 'Male', 'X'],
        'address': ['123 Main St', '456 Oak Ave', 'town', 'none'],
        'city': ['Boston', 'New York', '123 City', 'unknown'],
        'state': ['MA', 'NY', 'XX', 'Massachusetts'],
        'zip': ['02108', '10001', '123', '12345-6789'],
        'phone': ['(555) 123-4567', '555-123-4567', '1234567890', 'invalid'],
        'insurance_id': ['ABC123', 'XYZ789', '123ABC', 'invalid'],
        'date_of_birth': ['1990-01-01', '01/15/1995', 'invalid', None],
        'insurance_effective_date': ['2024-01-01', '01/15/2024', 'invalid', None]
    })

def test_validate_patient_id(sample_patient_df):
    validate_patient_id(sample_patient_df)
    assert sample_patient_df.loc[0, 'patient_id'] == 'A123'
    assert sample_patient_df.loc[1, 'patient_id'] == 'B456'
    assert pd.isna(sample_patient_df.loc[2, 'patient_id'])  
    assert pd.isna(sample_patient_df.loc[3, 'patient_id']) 

def test_validate_names(sample_patient_df):
    validate_names(sample_patient_df)
    # Test first names
    assert sample_patient_df.loc[0, 'first_name'] == 'John'
    assert sample_patient_df.loc[1, 'first_name'] == 'Mary'
    assert pd.isna(sample_patient_df.loc[2, 'first_name'])
    assert pd.isna(sample_patient_df.loc[3, 'first_name'])  
    
    # Test last names
    assert sample_patient_df.loc[0, 'last_name'] == 'Smith'
    assert sample_patient_df.loc[1, 'last_name'] == 'Johnson'
    assert pd.isna(sample_patient_df.loc[2, 'last_name'])  
    assert pd.isna(sample_patient_df.loc[3, 'last_name']) 

def test_validate_gender(sample_patient_df):
    validate_gender(sample_patient_df)
    assert sample_patient_df.loc[0, 'gender'] == 'M'
    assert sample_patient_df.loc[1, 'gender'] == 'F'
    assert sample_patient_df.loc[2, 'gender'] == 'M'
    assert pd.isna(sample_patient_df.loc[3, 'gender']) 

def test_validate_address(sample_patient_df):
    validate_address(sample_patient_df)
    assert sample_patient_df.loc[0, 'address'] == '123 Main St'
    assert sample_patient_df.loc[1, 'address'] == '456 Oak Ave'
    assert pd.isna(sample_patient_df.loc[2, 'address']) 
    assert pd.isna(sample_patient_df.loc[3, 'address'])  

def test_validate_city(sample_patient_df):
    validate_city(sample_patient_df)
    assert sample_patient_df.loc[0, 'city'] == 'Boston'
    assert sample_patient_df.loc[1, 'city'] == 'New York'
    assert pd.isna(sample_patient_df.loc[2, 'city'])  
    assert pd.isna(sample_patient_df.loc[3, 'city'])  

def test_validate_state(sample_patient_df):
    validate_state(sample_patient_df)
    assert sample_patient_df.loc[0, 'state'] == 'MA'
    assert sample_patient_df.loc[1, 'state'] == 'NY'
    assert pd.isna(sample_patient_df.loc[2, 'state']) 
    assert pd.isna(sample_patient_df.loc[3, 'state']) 

def test_validate_zip_code(sample_patient_df):
    validate_zip_code(sample_patient_df)
    assert sample_patient_df.loc[0, 'zip'] == '02108'
    assert sample_patient_df.loc[1, 'zip'] == '10001'
    assert pd.isna(sample_patient_df.loc[2, 'zip'])  
    assert sample_patient_df.loc[3, 'zip'] == '12345-6789' 

def test_validate_phone(sample_patient_df):
    validate_phone(sample_patient_df)
    assert sample_patient_df.loc[0, 'phone'] == '(555) 123-4567'
    assert sample_patient_df.loc[1, 'phone'] == '(555) 123-4567'  
    assert sample_patient_df.loc[2, 'phone'] == '(123) 456-7890'  
    assert pd.isna(sample_patient_df.loc[3, 'phone'])  

def test_validate_insurance_id(sample_patient_df):
    validate_insurance_id(sample_patient_df)
    assert sample_patient_df.loc[0, 'insurance_id'] == 'ABC123'
    assert sample_patient_df.loc[1, 'insurance_id'] == 'XYZ789'
    assert pd.isna(sample_patient_df.loc[2, 'insurance_id'])  
    assert pd.isna(sample_patient_df.loc[3, 'insurance_id']) 