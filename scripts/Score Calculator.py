import pandas as pd
import os

# 1. Setup Paths (Script-Relative)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "master_phishing_dataset.csv")
METRICS_PATH = os.path.join(SCRIPT_DIR, "..", "metrics", "feature_analysis.csv")

# 2. Load the dataset
if not os.path.exists(DATA_PATH):
    print(f"Error: Could not find dataset at {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH)

def extract_security_features(text):
    text = str(text).lower()
    
    # Define our "Tone" and "Urgency" markers
    urgency_words = ['immediate', 'urgent', 'action required', 'suspended', 'expire', 'limited time']
    threat_words = ['password', 'login', 'verify', 'account', 'security', 'unauthorized', 'leaked']
    
    # Calculate scores
    urgency_score = sum(1 for word in urgency_words if word in text)
    threat_score = sum(1 for word in threat_words if word in text)
    
    # Check for "Too good to be true"
    greed_score = sum(1 for word in ['winner', 'congratulations', 'inheritance', 'transfer', 'claim'] if word in text)
    
    return pd.Series([urgency_score, threat_score, greed_score])

# 3. Run the analysis
print("Analyzing security features... this may take a moment.")
df[['urgency', 'threat', 'greed']] = df['text'].apply(extract_security_features)

# 4. Compare the Averages (The "Evidence" for your report)
stats = df.groupby('label')[['urgency', 'threat', 'greed']].mean()

print("\n--- AVERAGE SCORES BY CATEGORY ---")
print("(Label 1 = Phish, Label 0 = Safe)")
print(stats)

# 5. Save the results to your metrics folder
stats.to_csv(METRICS_PATH)
print(f"\nAnalysis saved to {METRICS_PATH}")