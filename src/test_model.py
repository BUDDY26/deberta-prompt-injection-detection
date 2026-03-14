"""
Test script for evaluating the fine-tuned model on the training graph and eval loss and accuacy graph for each model step trained, with the name of the dataset dataset
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# Configuration
# MODEL_PATH = "deberta-pi-full-final"  # Path to your fine-tuned model
MODEL_PATH = "deberta-pi-full-stage3-final"  # Path to your fine-tuned model
BATCH_SIZE = 16
MAX_LENGTH = 256

print("="*80)
print("Testing Model on NVIDIA Aegis AI Content Safety Dataset")
print("="*80 + "\n")

# 1) Load the fine-tuned model and tokenizer
print(f"Loading model from: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on device: {device}\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to load from stage 1 model...")
    MODEL_PATH = "deberta-pi-full-stage1-final"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on device: {device}\n")

# 2) Load the NVIDIA Aegis dataset
print("Loading NVIDIA Aegis AI Content Safety Dataset...")
try:
    dataset = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Use test split if available, otherwise use validation or train
    if "test" in dataset:
        test_ds = dataset["test"]
    elif "validation" in dataset:
        test_ds = dataset["validation"]
    else:
        # Take a sample from train for testing
        test_ds = dataset["train"].train_test_split(test_size=0.1, seed=42)["test"]
    
    print(f"Test set size: {len(test_ds)}")
    print(f"Dataset columns: {test_ds.column_names}\n")
    
    # Examine first few examples to understand structure
    print("Sample data points:")
    for i in range(min(3, len(test_ds))):
        print(f"\nExample {i}:")
        print(test_ds[i])
    print()
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 3) Preprocess the dataset
print("Preprocessing dataset...")

def get_text_and_label(example):
    """
    Extract text and label from the NVIDIA Aegis dataset.
    The Aegis dataset structure:
    - 'prompt': the text to classify
    - 'prompt_label': safety label for the prompt
    - 'response_label': safety label for the response (if present)
    """
    # Extract text field - use 'prompt' which is the user input to classify
    text = example.get('prompt', '')
    
    # Extract and map label from prompt_label field
    label = None
    
    # Check for prompt_label (primary field for this dataset)
    if 'prompt_label' in example:
        raw_label = example['prompt_label']
        # Handle string labels
        if isinstance(raw_label, str):
            raw_label_lower = raw_label.lower()
            # Map various unsafe/harmful labels to 1
            if any(keyword in raw_label_lower for keyword in ['unsafe', 'harmful', 'toxic', 'injection', 'attack', 'malicious', 'jailbreak']):
                label = 1
            # Map safe/benign labels to 0
            elif any(keyword in raw_label_lower for keyword in ['safe', 'benign', 'harmless', 'legitimate']):
                label = 0
            else:
                # If it's a string number
                try:
                    label = int(raw_label)
                except:
                    label = 1 if 'unsafe' in raw_label_lower else 0
        else:
            # Numeric label - keep as is
            label = int(raw_label)
    
    # Fallback to 'label' field if prompt_label not found
    elif 'label' in example:
        raw_label = example['label']
        if isinstance(raw_label, str):
            raw_label_lower = raw_label.lower()
            if any(keyword in raw_label_lower for keyword in ['unsafe', 'harmful', 'toxic', 'injection', 'attack', 'malicious', 'jailbreak']):
                label = 1
            elif any(keyword in raw_label_lower for keyword in ['safe', 'benign', 'harmless', 'legitimate']):
                label = 0
            else:
                try:
                    label = int(raw_label)
                except:
                    label = 1 if 'unsafe' in raw_label_lower else 0
        else:
            label = int(raw_label)
    
    # Check for other possible label fields
    if label is None:
        for label_field in ['is_safe', 'safe', 'is_harmful', 'harmful', 'toxicity', 'safety_label']:
            if label_field in example:
                raw_label = example[label_field]
                if isinstance(raw_label, bool):
                    # If is_safe or safe -> invert (True=safe=0, False=unsafe=1)
                    if 'safe' in label_field:
                        label = 0 if raw_label else 1
                    # If is_harmful or harmful -> direct (True=harmful=1, False=safe=0)
                    else:
                        label = 1 if raw_label else 0
                else:
                    label = int(raw_label)
                break
    
    # Default to 0 if no label found
    if label is None:
        label = 0
        print(f"Warning: Could not determine label for example, defaulting to 0 (safe)")
        print(f"Example keys: {example.keys()}")
        print(f"Example values: {example}")
    
    return {'text': text, 'label': label}

# Extract text and labels
processed_data = [get_text_and_label(example) for example in test_ds]
texts = [item['text'] for item in processed_data]
labels = [item['label'] for item in processed_data]

print(f"Processed {len(texts)} examples")
print(f"Label distribution: Safe (0): {labels.count(0)}, Unsafe (1): {labels.count(1)}\n")

# 4) Run inference
print("Running inference...")
predictions = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        outputs = model(**inputs)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(batch_preds)

# 5) Calculate metrics
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80 + "\n")

# Overall accuracy
accuracy = accuracy_score(labels, predictions)
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Precision, Recall, F1-Score
precision, recall, f1, support = precision_recall_fscore_support(
    labels, predictions, average='binary', pos_label=1, zero_division=0
)
print("Binary Classification Metrics (Positive class = Unsafe/Injection):")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}\n")

# Per-class metrics
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    labels, predictions, average=None, labels=[0, 1], zero_division=0
)
print("Per-Class Metrics:")
print(f"  Class 0 (Safe):     Precision={precision_per_class[0]:.4f}, Recall={recall_per_class[0]:.4f}, F1={f1_per_class[0]:.4f}, Support={support_per_class[0]}")
print(f"  Class 1 (Unsafe):   Precision={precision_per_class[1]:.4f}, Recall={recall_per_class[1]:.4f}, F1={f1_per_class[1]:.4f}, Support={support_per_class[1]}\n")

# Confusion Matrix
cm = confusion_matrix(labels, predictions)
print("Confusion Matrix:")
print("                Predicted")
print("              Safe  Unsafe")
print(f"Actual Safe   {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"      Unsafe  {cm[1][0]:5d}  {cm[1][1]:5d}\n")

# Detailed classification report
print("Detailed Classification Report:")
print(classification_report(
    labels, 
    predictions, 
    target_names=['Safe', 'Unsafe'],
    digits=4,
    zero_division=0
))

# 6) Random Sample Examples
print("\n" + "="*80)
print("RANDOM SAMPLE TEST EXAMPLES")
print("="*80 + "\n")

# Select 10 random indices
import random
random.seed(42)
num_random_samples = min(10, len(texts))
random_indices = random.sample(range(len(texts)), num_random_samples)

print(f"Showing {num_random_samples} random test examples:\n")

for idx in random_indices:
    print(f"Example {idx}:")
    print(f"  Text: {texts[idx][:200]}..." if len(texts[idx]) > 200 else f"  Text: {texts[idx]}")
    print(f"  True Label: {'Safe' if labels[idx] == 0 else 'Unsafe'}")
    print(f"  Predicted:  {'Safe' if predictions[idx] == 0 else 'Unsafe'}")
    print(f"  Correct: {'✓' if labels[idx] == predictions[idx] else '✗'}")
    print()

# 7) Error Analysis - Show some misclassified examples
print("\n" + "="*80)
print("ERROR ANALYSIS - Sample Misclassifications")
print("="*80 + "\n")

misclassified_indices = [i for i in range(len(labels)) if labels[i] != predictions[i]]
num_samples = min(20, len(misclassified_indices))

if misclassified_indices:
    print(f"Total misclassifications: {len(misclassified_indices)}")
    print(f"Showing first {num_samples} examples:\n")
    
    for idx in misclassified_indices[:num_samples]:
        print(f"Example {idx}:")
        print(f"  Text: {texts[idx][:200]}..." if len(texts[idx]) > 200 else f"  Text: {texts[idx]}")
        print(f"  True Label: {'Safe' if labels[idx] == 0 else 'Unsafe'}")
        print(f"  Predicted:  {'Safe' if predictions[idx] == 0 else 'Unsafe'}")
        print()
else:
    print("No misclassifications found! Perfect accuracy!")

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)

# 7) Save results to file
results_summary = {
    "model_path": MODEL_PATH,
    "dataset": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
    "test_set_size": len(texts),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "label_distribution": {"safe": labels.count(0), "unsafe": labels.count(1)}
}

print("\nSaving results to test_results_2.txt...")
with open("test_results_2.txt", "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("Model Evaluation Results\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Dataset: nvidia/Aegis-AI-Content-Safety-Dataset-2.0\n")
    f.write(f"Test Set Size: {len(texts)}\n")
    f.write(f"Device: {device}\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-Score:  {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"                Predicted\n")
    f.write(f"              Safe  Unsafe\n")
    f.write(f"Actual Safe   {cm[0][0]:5d}  {cm[0][1]:5d}\n")
    f.write(f"      Unsafe  {cm[1][0]:5d}  {cm[1][1]:5d}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(
        labels, 
        predictions, 
        target_names=['Safe', 'Unsafe'],
        digits=4,
        zero_division=0
    ))

print("Results saved successfully!\n")
