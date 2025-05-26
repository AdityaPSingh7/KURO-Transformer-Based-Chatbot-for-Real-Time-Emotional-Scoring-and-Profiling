# KURO — Transformer-Based Chatbot for Real-Time Emotional Scoring and Profiling

KURO is a transformer-powered chatbot that enables real-time conversation **and** post-session emotional analysis.

It uses  
- **DistilBERT** fine-tuned for 3-class sentiment (Positive, Negative, Neutral)  
- **RoBERTa** (`SamLowe/roberta-base-go_emotions`) for 28-label emotion detection  

---

## 🚀 Features
- Django web chat interface  
- Automatic chat log (`read.txt`) after each session  
- Sentiment analysis (DistilBERT)  
- Emotion classification (RoBERTa)  
- Mental-health score (0 – 10)  
- Emotion-distribution visualization  

---

## 🛠️ Setup

### 1  Create & activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```
### 2 Install dependencies
```bash
pip install -r requirements.txt
```

### 🧪 Train the sentiment model
```bash
python sme.py
```
Uses goemotions_3class.csv (mapped from the original GoEmotions):

| Class        | Labels                                                                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Positive** | admiration, amusement, approval, caring, curiosity, desire, excitement, gratitude, joy, love, optimism, pride, realization, relief, surprise |
| **Negative** | anger, annoyance, confusion, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness                 |
| **Neutral**  | neutral                                                                                                                                      |

### 💬 Run the chatbot


