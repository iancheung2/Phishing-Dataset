import os
import torch
from transformers import pipeline

# 1. Setup Script-Relative Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "phishing_model_v1")

# 2. Hardware-Agnostic Device Selection
# This ensures it works on your M2 Mac (mps) and your teammates' PCs (cuda or cpu)
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = 0 # CUDA uses index
else:
    device = -1 # CPU

print(f"--- Loading model on device: {device} ---")

# 3. Initialize Pipeline
# We point to the local folder containing your config.json and tokenizer.json
pipe = pipeline("text-classification", model=MODEL_PATH, device=device)

def test_email(text):
    result = pipe(text)
    # Explicitly mapping the generic labels found in your config.json
    label_map = {
        "LABEL_0": "✅ SAFE", 
        "LABEL_1": "⚠️ PHISH"
    }
    
    raw_label = result[0]['label']
    # .get() ensures that if the model ever uses '0' or 'SAFE', it won't crash
    friendly_label = label_map.get(raw_label, raw_label)
    
    print(f"Testing: {text[:60]}...")
    print(f"Result: {friendly_label} | Confidence: {result[0]['score']:.2%}\n")

# --- Test Cases ---
if __name__ == "__main__":
    # Test 1: High Urgency
    test_email("URGENT: Your bank account has been locked. Click here to verify your identity.")

    # Test 2: Legitimate Business
    test_email("Hey team, just a reminder that the meeting has been moved to Room 402.")

    # Test 3: The "Soft" Social Engineering Phish
    test_email("I'm the new IT intern, can you please click this link to update your directory profile?")