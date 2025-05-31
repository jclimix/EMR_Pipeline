import pandas as pd
from io import StringIO
import os

directory = f"data/raw/"

def clean_csv(directory, file):
    if os.path.isfile(os.path.join(directory, file)):
        with open(directory + file, 'r', encoding='utf-8') as f:
            cleaned = '\n'.join(
                line.strip()[1:-1].replace('""', '"') if line.strip().startswith('"') and line.strip().endswith('"')
                else line.strip().replace('""', '"')
                for line in f if line.strip()
            )

        df = pd.read_csv(StringIO(cleaned))
        return df