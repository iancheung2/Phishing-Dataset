import os
import json
from transformers import AutoConfig, AutoModelForSequenceClassification

# 1. Setup Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "phishing_model_v1")

def verify_model_labels():
    print(f"--- Auditing Model at: {MODEL_PATH} ---\n")
    
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model folder not found. Please run the Training Loop first.")
        return

    # Method A: Direct JSON Inspection (The "Ground Truth")
    config_file = os.path.join(MODEL_PATH, "config.json")
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Method B: Library Inspection (How your code 'sees' it)
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    print("Verification Results:")
    print(f"  - ID to Label Map: {config.id2label}")
    print(f"  - Label to ID Map: {config.label2id}")
    print(f"  - Number of Labels: {config.num_labels}")
    
    # Logic Check
    if config.id2label[0] == "LABEL_0":
        print("\n⚠️  NOTICE: Your model is using default generic labels (LABEL_0, LABEL_1).")
        print("   In your testing script, map 0 -> SAFE and 1 -> PHISH.")
    else:
        print("\n✅ SUCCESS: Your model has custom labels baked into the config.")

if __name__ == "__main__":
    verify_model_labels()