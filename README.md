# Phishing Detection System (DistilBERT)

This repository contains a machine learning pipeline designed to detect phishing attempts in email text using a fine-tuned DistilBERT model.

## 📂 Project Structure
- **/data**: Contains raw CSV datasets (Enron, Nazario, CEAS, etc.) and the generated `master_phishing_dataset.csv`.
- **/scripts**: Python scripts for data processing, training, and testing.
- **/phishing_model_v1**: Model configuration and tokenizer files (Weights generated locally).
- **/metrics**: Performance logs, loss curves, and feature analysis results.

## 🚀 Getting Started

### 1. Environment Setup
Ensure you have Python 3.9+ installed. Install the required AI and security libraries using pip:

```bash
pip install torch pandas transformers scikit-learn tqdm matplotlib
```

### 2. Data Preparation

The raw datasets are already included in the /data folder. If you want to refresh the master dataset after adding new data or if you lose that file:

Navigate to the root directory.

Run the dataset builder:

```Bash
python scripts/Master\ Dataset\ Maker.py
```
This script handles data cleaning and deduplication.

### 3. Training the Model

To train the model on your local hardware:

```Bash
python scripts/Training\ Loop.py
```
Hardware Detection: The script automatically detects MPS (Apple Silicon), CUDA (NVIDIA), or CPU.

Output: This will generate the model.safetensors file inside /phishing_model_v1 and save training plots in /metrics.

### 4. Verification & Testing

Once training is complete, you can verify the results:

Audit Labels: Run 
```Bash 
python scripts/Label_Verification.py
``` 
to confirm ID mapping (0=Safe, 1=Phish).

Run Inference: Use the testing script to evaluate custom email strings:

```Bash
python scripts/Model\ Testing.py
```