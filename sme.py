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

# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from transformers import (
#     AutoTokenizer, AutoModelForSequenceClassification,
#     TrainingArguments, Trainer
# )
# from datasets import Dataset

# # === Step 1: Load CSV ===
# df = pd.read_csv("goemotions_3class.csv")
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df["text"], df["sentiment"], test_size=0.2, random_state=42
# )

# # === Step 2: Label Encoding ===
# label_encoder = LabelEncoder()
# train_labels_enc = label_encoder.fit_transform(train_labels)
# test_labels_enc = label_encoder.transform(test_labels)

# # === Step 3: Convert to Hugging Face Dataset ===
# train_dataset = Dataset.from_dict({
#     "text": list(train_texts),
#     "label": list(train_labels_enc)
# })
# test_dataset = Dataset.from_dict({
#     "text": list(test_texts),
#     "label": list(test_labels_enc)
# })

# # === Step 4: Tokenize ===
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# def tokenize(batch):
#     return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# train_dataset = train_dataset.map(tokenize, batched=True)
# test_dataset = test_dataset.map(tokenize, batched=True)

# train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
# test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# # === Step 5: Load Model ===
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# # === Step 6: Training Arguments ===

# training_args = TrainingArguments(
    
#     output_dir="./emotion_model",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     logging_steps=50,
#     load_best_model_at_end=True,
    
# )

# # === Step 7: Trainer ===
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

# # === Step 8: Train! ===
# trainer.train()

# # === Step 9: Save Model and Label Encoder ===
# model.save_pretrained("./emotion_model")
# tokenizer.save_pretrained("./emotion_model")

# import pickle
# with open("label_encoder.pkl", "wb") as f:
#     pickle.dump(label_encoder, f)

# print("✅ Training complete. Model saved to ./emotion_model")

































#GOLD
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import pandas as pd
# import numpy as np
# import tkinter as tk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt

# # Step 3: Load Model and Tokenizer
# tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# # Step 4: Sentiment Score Function
# def sentiment_score(text):
#     tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
#     result = model(tokens)
#     return int(torch.argmax(result.logits)) + 1  # Sentiment 1–5

# # Step 5: Read Text File
# with open('read.txt', 'r', encoding='utf-8') as f:
#     lines = [line.strip() for line in f.readlines() if line.strip()]

# # Step 6: Analyze Sentiment
# df = pd.DataFrame(lines, columns=['text'])
# df['sentiment'] = df['text'].apply(sentiment_score)

# # Step 7: Mental Health Score (0–10)
# def mental_health_score_fn(scores):
#     avg = np.mean(scores)
#     return round((avg - 1) * (10 / 4), 2)

# mh_score = mental_health_score_fn(df['sentiment'])

# # Step 8: GUI with Bar Chart and Score
# def display_results():
#     # Create window
#     root = tk.Tk()
#     root.title("Mental Health Sentiment Analysis")
#     root.geometry("700x500")
    
#     # Title Label
#     tk.Label(root, text=f"Your Mental Health Score is: {mh_score} / 10", font=("Helvetica", 16), pady=10).pack()

#     # Sentiment Distribution
#     sentiment_counts = df['sentiment'].value_counts().sort_index()
#     labels = ['Very Poor', 'Poor', 'Neutral', 'Good', 'Very Good']
#     values = [sentiment_counts.get(i, 0) for i in range(1, 6)]
#     bar_colors = ['#ff4d4d', '#ff9999', '#ffd11a', '#99e699', '#4da6ff']

#     # Create bar chart
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.bar(labels, values, color=bar_colors)
#     ax.set_ylabel("Number of Sentences")
#     ax.set_title("Sentiment Breakdown (1 to 5 Stars)", fontsize=14)

#     # Attach matplotlib figure to tkinter
#     canvas = FigureCanvasTkAgg(fig, master=root)
#     canvas.draw()
#     canvas.get_tk_widget().pack(pady=10)

#     root.mainloop()

# # Run
# display_results()

# # Step 1: Install required packages (uncomment if not installed)
# # !pip install torch torchvision torchaudio
# # !pip install transformers pandas numpy

# # Step 2: Import Libraries
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import pandas as pd
# import numpy as np

# # Step 3: Load Pretrained Model & Tokenizer
# tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# # Step 4: Define Sentiment Scoring Function
# def sentiment_score(text):
#     tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
#     result = model(tokens)
#     return int(torch.argmax(result.logits)) + 1  # Model returns score between 1-5

# # Step 5: Read Text File
# with open('read.txt', 'r', encoding='utf-8') as file:
#     content = file.read()

# # Step 6: Process Content by Lines
# lines = content.split('\n')
# lines = [line.strip() for line in lines if line.strip() != '']

# # Step 7: Apply Sentiment Scoring
# df = pd.DataFrame(lines, columns=["text"])
# df["sentiment_score"] = df["text"].apply(sentiment_score)

# # Step 8: Calculate Mental Health Score (0-10)
# def convert_to_mental_health_score(sentiment_scores):
#     avg_sentiment = np.mean(sentiment_scores)
#     return round((avg_sentiment - 1) * (10 / 4), 2)  # scales 1-5 to 0-10

# mental_health_score = convert_to_mental_health_score(df["sentiment_score"])

# # Step 9: Print Results
# print("Sentiment Scores by Line:")
# print(df)
# print("\nOverall Mental Health Score (0–10):", mental_health_score)
