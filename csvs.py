import pandas as pd
import  importlib

# Load the Excel file
excel_file = 'C:/Users/ershi/Downloads/zomato-schema.xlsx'

# Load all sheets into a dictionary of DataFrames
sheets = pd.read_excel(excel_file, sheet_name=None)

# Iterate over the sheets and save each one as a CSV file
for sheet_name, df in sheets.items():
    csv_file = f'{sheet_name}.csv'  # Create a CSV file name based on sheet name
    df.to_csv(csv_file, index=False)
    print(f'Sheet {sheet_name} saved as {csv_file}')
