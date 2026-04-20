import pandas as pd
import os

# This finds the 'scripts' folder where the file lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# This moves up one and into 'data' regardless of your VS Code settings
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

files_to_check = [
    "Nazario.csv", "Enron.csv", "Nigerian_Fraud.csv", 
    "CEAS_08.csv"
]

print(f"{'File Name':<20} | {'Rows':<8} | {'Columns'}")
print("-" * 60)

for file_name in files_to_check:
    # Use join to create a path like "../data/Enron.csv"
    file_path = os.path.join(DATA_DIR, file_name)
    
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path, nrows=5)
        # Using a fast way to count rows for a security audit
        row_count = sum(1 for _ in open(file_path, encoding='utf-8', errors='ignore')) - 1 
        
        cols = df_temp.columns.tolist()
        print(f"{file_name:<20} | {row_count:<8} | {cols}")
    else:
        print(f"{file_name:<20} | ERROR    | Not found in {DATA_DIR}")