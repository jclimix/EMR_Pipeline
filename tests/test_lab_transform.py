import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import os
from transform.lab_transform import (
    validate_lab_id,
    validate_visit_id,
    validate_test_name,
    validate_test_value,
    validate_test_units,
    validate_reference_range,
    validate_date
)

@pytest.fixture
def sample_lab_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'lab_id': ['L1234', 'L5678', 'invalid', 'L9012'],
        'visit_id': ['V123', 'V456', 'invalid', 'V789'],
        'test_name': ['Blood Test', 'Urine Test', '', 'None'],
        'test_value': ['12.5', 'Positive', 'invalid', ''],
        'test_units': ['mg/dL', '', 'units', ''],
        'reference_range': ['10.0-15.0', 'Negative', 'invalid', ''],
        'date_performed': ['2024-01-01', '01/15/2024', 'invalid', ''],
        'date_resulted': ['2024-01-02', '01/16/2024', 'invalid', '']
    })

def test_validate_lab_id(sample_lab_df):
    validate_lab_id(sample_lab_df)
    assert pd.isna(sample_lab_df.loc[2, 'lab_id'])
    assert sample_lab_df.loc[0, 'lab_id'] == 'L1234'
    assert sample_lab_df.loc[1, 'lab_id'] == 'L5678'
    assert sample_lab_df.loc[3, 'lab_id'] == 'L9012'

def test_validate_visit_id(sample_lab_df):
    validate_visit_id(sample_lab_df)
    assert pd.isna(sample_lab_df.loc[2, 'visit_id'])
    assert sample_lab_df.loc[0, 'visit_id'] == 'V123'
    assert sample_lab_df.loc[1, 'visit_id'] == 'V456'
    assert sample_lab_df.loc[3, 'visit_id'] == 'V789'

def test_validate_test_name(sample_lab_df):
    validate_test_name(sample_lab_df)
    assert pd.isna(sample_lab_df.loc[2, 'test_name'])
    assert pd.isna(sample_lab_df.loc[3, 'test_name'])
    assert sample_lab_df.loc[0, 'test_name'] == 'Blood Test'
    assert sample_lab_df.loc[1, 'test_name'] == 'Urine Test'

def test_validate_test_value(sample_lab_df):
    validate_test_value(sample_lab_df)
    assert sample_lab_df.loc[0, 'test_value'] == 12.5
    assert sample_lab_df.loc[1, 'test_value'] == 'Positive'
    assert pd.isna(sample_lab_df.loc[2, 'test_value'])
    assert pd.isna(sample_lab_df.loc[3, 'test_value'])

def test_validate_test_units(sample_lab_df):

    validate_test_value(sample_lab_df)
    
    # copy the DataFrame to avoid modifying the original
    df = sample_lab_df.copy()
    validate_test_units(df)
    
    assert df.loc[0, 'test_units'] == 'mg/dL'
    assert pd.isna(df.loc[1, 'test_units'])
    assert pd.isna(df.loc[3, 'test_units'])

def test_validate_reference_range(sample_lab_df):
    df = sample_lab_df.copy()
    validate_reference_range(df)
    
    assert df.loc[0, 'reference_range'] == '10.0-15.0'
    assert df.loc[1, 'reference_range'] == 'Negative'
    assert pd.isna(df.loc[2, 'reference_range'])
    assert pd.isna(df.loc[3, 'reference_range'])

def test_validate_date(sample_lab_df):
    validate_date(sample_lab_df, 'date_performed')
    validate_date(sample_lab_df, 'date_resulted')
    
    assert sample_lab_df.loc[0, 'date_performed'] == '2024-01-01'
    assert sample_lab_df.loc[1, 'date_performed'] == '2024-01-15'
    assert pd.isna(sample_lab_df.loc[2, 'date_performed'])
    assert pd.isna(sample_lab_df.loc[3, 'date_performed'])
    
    assert sample_lab_df.loc[0, 'date_resulted'] == '2024-01-02'
    assert sample_lab_df.loc[1, 'date_resulted'] == '2024-01-16'
    assert pd.isna(sample_lab_df.loc[2, 'date_resulted'])
    assert pd.isna(sample_lab_df.loc[3, 'date_resulted'])