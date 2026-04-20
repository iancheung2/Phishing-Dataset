import pandas as pd
import os

# 1. Path Setup (Script-Relative)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "master_phishing_dataset.csv")

# Files and their intended "roles"
phishing_files = ["Nazario.csv", "Nigerian_Fraud.csv"]
safe_files = ["Enron.csv", "CEAS_08.csv"]

all_dfs = []

# Helper function to find the right column (some use 'body', some use 'text')
def get_text_col(df):
    for col in ['body', 'text', 'message', 'Message']:
        if col in df.columns:
            return col
    return None

# 2. Load all Phishing emails
for f in phishing_files:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        text_col = get_text_col(df)
        if text_col:
            df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
            all_dfs.append(df)
            print(f"Loaded {len(df)} phishing emails from {f}")

# 3. Load a subset of Safe emails
for f in safe_files:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=10000) 
        text_col = get_text_col(df)
        if text_col:
            df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
            all_dfs.append(df)
            print(f"Loaded {len(df)} safe emails from {f}")

# 4. Combine, Clean, and Deduplicate
master_df = pd.concat(all_dfs, ignore_index=True)
master_df.dropna(subset=['text'], inplace=True)

# SECURITY BEST PRACTICE: Remove exact duplicates
# Attackers often reuse templates; we don't want the model 'memorizing' the same email
initial_count = len(master_df)
master_df.drop_duplicates(subset=['text'], inplace=True)
print(f"Removed {initial_count - len(master_df)} duplicate emails.")

# Shuffle
master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Final Check & Save
print("\nFinal Class Distribution:")
print(master_df['label'].value_counts())

master_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nMaster dataset saved to: {OUTPUT_PATH}")