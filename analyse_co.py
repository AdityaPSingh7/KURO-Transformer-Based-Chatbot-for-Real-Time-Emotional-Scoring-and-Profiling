import torch
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import softmax
import matplotlib.pyplot as plt

# === Load custom 3-class sentiment model ===
model_path = "./emotion_model"
model_3class = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_3class = AutoTokenizer.from_pretrained(model_path)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Load GoEmotions 28-label model ===
model_28 = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
tokenizer_28 = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
goemotion_labels = model_28.config.id2label

# === Read input ===
with open("read.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# === Predict 3-class sentiment ===
def predict_sentiment(text):
    inputs = tokenizer_3class(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_3class(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return label_encoder.inverse_transform([pred])[0]

# === Predict 28-label emotion probabilities ===
def predict_emotions(text):
    inputs = tokenizer_28(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_28(**inputs)
        probs = softmax(outputs.logits, dim=1)[0].tolist()
    return probs

# === Process all lines ===
results = []
emotion_sums = [0.0] * 28

for line in lines:
    sentiment = predict_sentiment(line)
    emotion_probs = predict_emotions(line)
    results.append({"text": line, "sentiment": sentiment})
    emotion_sums = [a + b for a, b in zip(emotion_sums, emotion_probs)]

# === Calculate mental health score (out of 10) ===
score_map = {"negative": 0, "neutral": 5, "positive": 10}
avg_score = sum(score_map[r["sentiment"]] for r in results) / len(results)
mental_health_score = round(avg_score, 2)

# === Normalize emotion distribution ===
emotion_total = sum(emotion_sums)
emotion_percent = [(v / emotion_total) * 100 for v in emotion_sums]

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv("read_results.csv", index=False)

# === Display chart ===
plt.figure(figsize=(12, 6))
plt.bar([goemotion_labels[i] for i in range(28)], emotion_percent)
plt.xticks(rotation=90)
plt.ylabel("Percentage")
plt.title(f"Kuro's Analysis Report \n Your Mental Health Score: {mental_health_score} / 10")
plt.tight_layout()
plt.show()











# # from datasets import load_dataset

# # # Load the dataset
# # goemotions = load_dataset("go_emotions", split="train")
# # print(goemotions.features)
# # from datasets import load_dataset
# # import pandas as pd

# # # Load the dataset (e.g., training split)
# # dataset = load_dataset("go_emotions", split="train")

# # # Convert to pandas DataFrame
# # df = dataset.to_pandas()

# # # Optional: Convert label IDs to string names
# # label_list = dataset.features["labels"].feature.names

# # # Create a column with readable emotion labels
# # df["label_names"] = df["labels"].apply(lambda label_ids: [label_list[i] for i in label_ids])

# # # Save to CSV
# # df.to_csv("goemotions_train.csv", index=False)

# # print("Saved as goemotions_train.csv")
# from datasets import load_dataset
# import pandas as pd

# # Load dataset
# dataset = load_dataset("go_emotions", split="train")
# label_names = dataset.features["labels"].feature.names

# # Define emotion categories
# positive = [
#     'admiration', 'amusement', 'approval', 'caring', 'curiosity', 'desire',
#     'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization',
#     'relief', 'surprise'
# ] # use list above
# negative = [
#     'anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust',
#     'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
# ]

# neutral = ['neutral']

# def map_to_sentiment(label_ids):
#     emotion_names = [label_names[i] for i in label_ids]
#     pos = sum(em in positive for em in emotion_names)
#     neg = sum(em in negative for em in emotion_names)
#     neu = sum(em in neutral for em in emotion_names)

#     # Assign based on majority class
#     if pos > neg and pos > neu:
#         return "positive"
#     elif neg > pos and neg > neu:
#         return "negative"
#     elif neu > 0:
#         return "neutral"
#     else:
#         return "neutral"  # fallback in unclear cases

# # Convert to DataFrame
# df = dataset.to_pandas()
# df["sentiment"] = df["labels"].apply(map_to_sentiment)

# # Keep only what we need
# df_simple = df[["text", "sentiment"]]
# df_simple.to_csv("goemotions_3class.csv", index=False)

# print("Saved as goemotions_3class.csv ✅")


# # import os
# # from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
# # import torch
# # import torch.nn.functional as F

# # # --- Auto-detect latest checkpoint ---
# # def get_latest_checkpoint(results_dir="./results"):
# #     checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint")]
# #     if not checkpoints:
# #         raise FileNotFoundError("No checkpoints found in ./results.")
# #     latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
# #     return os.path.join(results_dir, latest)

# # model_path = get_latest_checkpoint()

# # # --- Load tokenizer & model ---
# # tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
# # model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
# # model.eval()

# # # --- Load and preprocess text ---
# # with open("read.txt", "r", encoding="utf-8") as f:
# #     text = f.read()

# # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# # with torch.no_grad():
# #     logits = model(**inputs).logits
# #     probs = F.softmax(logits, dim=-1)
# #     confidence, pred = torch.max(probs, dim=1)

# # # --- Output ---
# # id2label = {
# #     0: "Stress",
# #     1: "Depression",
# #     2: "Bipolar disorder",
# #     3: "Personality disorder",
# #     4: "Anxiety"
# # }
# # score = round(confidence.item() * 10, 1)
# # diagnosis = id2label[pred.item()]

# # print(f"Mental Health Score: {score}/10")
# # print(f"Detected Condition: {diagnosis}")

