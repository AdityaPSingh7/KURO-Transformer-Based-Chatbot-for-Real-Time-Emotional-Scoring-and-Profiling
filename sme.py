import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import pickle

# === Step 1: Load 3-class CSV ===
df = pd.read_csv("goemotions_3class.csv")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# === Step 2: Label Encoding ===
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)

# === Step 3: Convert to Hugging Face Dataset ===
train_dataset = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels_enc)})
test_dataset = Dataset.from_dict({"text": list(test_texts), "label": list(test_labels_enc)})

# === Step 4: Tokenize ===
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === Step 5: Load Model ===
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# === Step 6: Training Arguments ===
training_args = TrainingArguments(
    output_dir="./emotion_model",
    eval_strategy="epoch",  # use evaluation_strategy if on stable transformers
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=50,
    load_best_model_at_end=True,
)

# === Step 7: Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# === Step 8: Train ===
trainer.train()

# === Step 9: Save Model and Encoder ===
model.save_pretrained("./emotion_model")
tokenizer.save_pretrained("./emotion_model")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Training complete. Model saved to ./emotion_model")
