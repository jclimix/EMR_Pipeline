import pandas as pd

def extract_excel_data(excel_path):
    """
    Extracts all sheets from an Excel file and saves each sheet as a CSV.
    Each sheet is saved to 'data/raw/' with a filename based on the sheet name,
    converted to lowercase and underscores to replace spaces.
    """
    sheets = pd.read_excel(excel_path, sheet_name=None)

    for sheet_name, df in sheets.items():
        sheet_name = sheet_name.strip().lower().replace(" ", "_")
        csv_path = f"data/raw/{sheet_name}.csv"
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    excel_path = 'data/source/Data Eng Data Set.xlsx'
    extract_excel_data(excel_path)
