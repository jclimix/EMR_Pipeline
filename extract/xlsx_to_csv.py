import pandas as pd

# Load all sheets in the Excel file
excel_path = 'data/source/Data Eng Data Set.xlsx'
sheets = pd.read_excel(excel_path, sheet_name=None)  # sheet_name=None loads all sheets

# Export each sheet to a separate CSV
for sheet_name, df in sheets.items():
    sheet_name = sheet_name.strip()
    sheet_name = sheet_name.lower()
    sheet_name = sheet_name.replace(" ", "_")
    csv_path = f"data/raw/{sheet_name}.csv"  # Create a filename based on the sheet name
    df.to_csv(csv_path, index=False)
