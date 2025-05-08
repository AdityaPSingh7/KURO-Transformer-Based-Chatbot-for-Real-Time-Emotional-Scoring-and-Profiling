import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pickle
from sklearn.model_selection import train_test_split

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# === Load Test Data ===
df = pd.read_csv("goemotions_3class.csv")
_, test_texts, _, test_labels = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42
)

# === Load Label Encoder ===
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

test_labels_enc = label_encoder.transform(test_labels)

# === Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained("./emotion_model")
model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")
model.to(device)
model.eval()

# === Tokenize Test Data ===
test_dataset = Dataset.from_dict({"text": list(test_texts)})
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === Get Predictions ===
predictions = []
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=16):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, axis=1)
        predictions.extend(preds.cpu().numpy())  # move to CPU for metrics

# === Accuracy ===
acc = accuracy_score(test_labels_enc, predictions)
print(f"✅ Accuracy: {acc:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(test_labels_enc, predictions)
class_names = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(test_labels_enc, predictions, target_names=class_names))

# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import pickle

# # === Load test data ===
# df = pd.read_csv("goemotions_3class.csv")
# df = df.sample(frac=1, random_state=42)  # shuffle
# test_df = df.iloc[int(0.8 * len(df)):]  # last 20% as test split

# # === Encode labels ===
# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# test_labels_enc = label_encoder.transform(test_df["sentiment"])

# # === Tokenize ===
# tokenizer = AutoTokenizer.from_pretrained("./emotion_model")

# test_dataset = Dataset.from_dict({
#     "text": list(test_df["text"]),
#     "label": list(test_labels_enc)
# })

# def tokenize(batch):
#     return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# test_dataset = test_dataset.map(tokenize, batched=True)
# test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# # === Load trained model ===
# model = AutoModelForSequenceClassification.from_pretrained("./emotion_model")

# # === Define metrics ===
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#     }

# # === Setup Trainer for evaluation only ===
# args = TrainingArguments(output_dir="./eval_temp", per_device_eval_batch_size=16)
# trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)

# # === Run evaluation ===
# results = trainer.evaluate(eval_dataset=test_dataset)

# # === Print results ===
# print("📊 Evaluation Results:")
# for k, v in results.items():
#     print(f"{k}: {v:.4f}")
