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

    # Test 4: Account Compromise + QR MFA Reset
    test_email("Security Alert: Suspicious login detected. Scan the QR code below using your mobile device to re-authenticate immediately or your account will be disabled within 15 minutes.")

    # Test 5: Payroll Breach Panic
    test_email("URGENT: Payroll discrepancy detected in your last deposit. Please scan the attached QR code to confirm your banking details before processing is frozen.")

    # Test 6: CEO Impersonation + Mobile Push
    test_email("Hey, I need you to quickly review a secure document. Use your phone to scan this QR code—it won’t open properly on desktop.")

    # Test 7: Minimal Content QR Attack
    test_email("Document ready for review.\n\nScan QR to access securely.")

    # Test 8: Attachment-Based QR Lure
    test_email("Please see attached invoice. For security reasons, access the document via the QR code in the PDF.")

    # Test 9: HTML QR Style Prompt
    test_email("To continue, scan the secure login code below. This step is required due to new compliance policies.")

    # Test 10: Fake IT Security Upgrade
    test_email("IT Notice: We’ve upgraded our VPN login. Scan the QR code with your phone to sync your credentials.")

    # Test 11: MFA Reset Scam
    test_email("Your multi-factor authentication has expired. Re-enroll by scanning this QR code on your mobile device.")

    # Test 12: Password Expiration
    test_email("Your password expires today. For security, use your phone to scan this QR code to set a new one.")

    # Test 13: Legit QR but Non-Urgent
    test_email("For convenience, you can scan this QR code to download our mobile app, or visit our website directly.")

    # Test 14: Internal IT Announcement
    test_email("IT will be rolling out optional mobile login via QR codes next quarter. No action is required at this time.")

    # Test 15: Helpful Coworker Tone
    test_email("Hey! I couldn’t access the shared folder on desktop, but scanning this QR worked for me—can you try?")

    # Test 16: New Employee Pretext
    test_email("Hi, I just joined IT onboarding. They told me to send this QR code for directory verification—can you complete it?")

    # Test 17: Casual Slack-like Style
    test_email("quick thing — can you scan this on your phone and confirm access? desktop version is glitching")

    # Test 18: Vendor Payment Update
    test_email("We’ve updated our payment portal. Please scan the QR code to securely re-enter your billing details.")

    # Test 19: Invoice Fraud
    test_email("Invoice overdue notice. To avoid late fees, scan the QR code to complete payment securely.")

    # Test 20: Gift Card Scam Evolution
    test_email("Can you grab gift cards for the client? Use the QR code to access the approved vendor portal.")

    # Test 21: Mobile-Only Access Push
    test_email("This secure link cannot be opened on desktop. Please scan the QR code using your phone.")

    # Test 22: App-Based Authentication Lure
    test_email("To continue, scan this QR code with your banking app to verify your identity.")

    # Test 23: “Safer on Phone” Deception
    test_email("For your security, this process must be completed on a mobile device. Scan the QR code to proceed.")

    # Test 24: Personalized Spear Phish
    test_email("Hi Alex, following up on your PTO request—HR needs you to confirm details. Please scan the QR code to finalize before approval.")

    # Test 25: Project Context Injection
    test_email("Regarding the Q2 budget review, finance uploaded a secure version. Scan the QR code to access the latest revision.")

    # Test 26: Calendar + Meeting Hook
    test_email("Before the 3PM meeting, please scan this QR code to access the updated agenda on your phone.")

    # Test 27: Dual Path Attack
    test_email("You can either click the link below or scan the QR code for faster access on your mobile device.")

    # Test 28: “Backup Access” Trick
    test_email("If the link doesn’t work, scan the QR code instead to access your secure message.")

    # Test 29: Fear + Authority
    test_email("Compliance Notice: Failure to scan and verify via QR code may result in account suspension.")

    # Test 30: Scarcity / Deadline
    test_email("This secure document will expire in 10 minutes. Scan the QR code now to retain access.")

    # Test 31: Curiosity Bait
    test_email("You’ve received a confidential message. Scan the QR code to view it securely.")

    # Test 32: Mixed Legit + Suspicious
    test_email("Here’s the official HR portal: hr.company.com. Alternatively, scan this QR code for quick mobile access.")

    # Test 33: QR Mention Without Malice
    test_email("The conference badge includes a QR code for check-in at the venue.")

    # Test 34: Obfuscated Intent
    test_email("Authentication token embedded below. Please scan to continue session.")