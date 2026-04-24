import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os # NEW: Added for path management
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm.auto import tqdm
from torch.optim import AdamW
import json

# --- (PhishingDataset class remains exactly the same) ---
class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def get_device():
    if torch.cuda.is_available():
        # Best for: Teammates with NVIDIA GPUs (Windows/Linux)
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Best for: Your M2 Pro and other Apple Silicon Macs
        device = torch.device("mps")
    else:
        # Fallback for: Intel Macs or laptops without dedicated GPUs
        device = torch.device("cpu")
    
    print(f"--- Environment Ready: {device} ---")
    return device

def main():
    # 1. Setup & Path Management (NEW)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "master_phishing_dataset.csv")
    MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "phishing_model_v1")
    METRICS_DIR = os.path.join(SCRIPT_DIR, "..", "metrics")

    # Create metrics folder if it doesn't exist
    os.makedirs(METRICS_DIR, exist_ok=True)

    device = get_device()

    # Load from the correct relative path
    df = pd.read_csv(DATA_PATH).dropna()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    train_loader = DataLoader(PhishingDataset(train_texts, train_labels, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(PhishingDataset(val_texts, val_labels, tokenizer), batch_size=16)

    # 2. Professional Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # 3. History Tracking for Graphs
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 4. Evaluation with Security Metrics
        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        # Calculate Epoch Results
        avg_train = total_train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        epoch_f1 = f1_score(all_labels, all_preds)
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1"].append(epoch_f1)

        print(f"\nEpoch {epoch+1} Results:")
        print(classification_report(all_labels, all_preds, target_names=['Safe', 'Phish']))

    # 5. Save Metrics & Graphs to the NEW Metrics folder
    pd.DataFrame(history).to_csv(os.path.join(METRICS_DIR, "training_metrics.csv"), index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(METRICS_DIR, "loss_curve.png"))
    print(f"Graphs and Metrics Saved to {METRICS_DIR}")

    # 6. Save Model and Tokenizer to the NEW Model folder
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    print(f"Model and Tokenizer saved to {MODEL_DIR}")

if __name__ == '__main__':
    main()