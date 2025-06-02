import sys, os
import pytest
import pandas as pd
import numpy as np

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transform.icd_transform import (
    is_invalid_value,
    validate_icd_code,
    validate_description,
    validate_status
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'icd_code': ['A01', 'B02.1', 'invalid', 'C03', 'D04.2', 'E5', 'F.1'],
        'description': ['Valid desc', '', 'None', 'nan', 'Valid desc 2', 'null', 'Valid desc 3'],
        'status': ['active', 'inactive', 'invalid', 'ACTIVE', 'Inactive', 'pending', 'unknown']
    })

def test_is_invalid_value():
    assert is_invalid_value(np.nan)
    assert is_invalid_value('')
    assert is_invalid_value('nan')
    assert is_invalid_value('none')
    assert is_invalid_value('null')
    assert is_invalid_value('  ')
    assert not is_invalid_value('valid')
    assert not is_invalid_value('0')
    assert not is_invalid_value('false')

def test_validate_icd_code(sample_df):
    df = sample_df.copy()
    validate_icd_code(df)
    assert df.loc[0, 'icd_code'] == 'A01'
    assert df.loc[1, 'icd_code'] == 'B02.1'
    assert df.loc[3, 'icd_code'] == 'C03'
    assert df.loc[4, 'icd_code'] == 'D04.2'
    assert pd.isna(df.loc[2, 'icd_code'])  
    assert pd.isna(df.loc[5, 'icd_code'])  
    assert pd.isna(df.loc[6, 'icd_code'])  

def test_validate_description(sample_df):
    df = sample_df.copy()
    validate_description(df)
    assert df.loc[0, 'description'] == 'Valid desc'
    assert df.loc[4, 'description'] == 'Valid desc 2'
    assert df.loc[6, 'description'] == 'Valid desc 3'
    assert pd.isna(df.loc[1, 'description'])  
    assert pd.isna(df.loc[2, 'description'])  
    assert pd.isna(df.loc[3, 'description'])  
    assert pd.isna(df.loc[5, 'description'])  

def test_validate_status(sample_df):
    df = sample_df.copy()
    validate_status(df)
    assert df.loc[0, 'status'] == 'Active'
    assert df.loc[1, 'status'] == 'Inactive'
    assert df.loc[3, 'status'] == 'Active'
    assert df.loc[4, 'status'] == 'Inactive'
    assert pd.isna(df.loc[2, 'status'])  
    assert pd.isna(df.loc[5, 'status'])  
    assert pd.isna(df.loc[6, 'status'])  
