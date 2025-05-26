# KURO — Transformer-Based Chatbot for Real-Time Emotional Scoring and Profiling

KURO is a transformer-powered chatbot that supports real-time conversation and post-session emotional analysis.  
It employs:

- **DistilBERT** fine-tuned for 3-class sentiment (Positive, Negative, Neutral)  
- **RoBERTa** (`SamLowe/roberta-base-go_emotions`) for 28-label emotion detection  

---

## 🚀 Features
- Django web chat interface  
- Automatic chat log (`read.txt`) after each session  
- Sentiment analysis (DistilBERT)  
- Emotion classification (RoBERTa)  
- Mental-health score (0–10)  
- Emotion-distribution visualization  

---

## 🛠️ Setup

### 1  Create & activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
2 Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🧪 Train the sentiment model
bash
Copy
Edit
python sme.py
Uses goemotions_3class.csv (mapped from the original GoEmotions):

Class	Labels
Positive	admiration, amusement, approval, caring, curiosity, desire, excitement, gratitude, joy, love, optimism, pride, realization, relief, surprise
Negative	anger, annoyance, confusion, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
Neutral	neutral

💬 Run the chatbot
Terminal 1 — start Django server
bash
Copy
Edit
cd my_chatbot
python manage.py runserver
Browse to http://127.0.0.1:8000

Chat as usual

Send buye to end; log saved to read.txt

Terminal 2 — run analysis
bash
Copy
Edit
python analyse_co.py
Outputs:

Sentiment label

28-emotion distribution chart

Mental-health score

🗂️ Project structure
bash
Copy
Edit
KURO/
├── my_chatbot/            # Django backend
├── goemotions_train.csv   # Original data
├── goemotions_3class.csv  # 3-class data
├── sme.py                 # Training script
├── analyse_co.py          # Analysis script
├── requirements.txt       # Dependencies
├── read.txt               # Chat log
└── README.md              # This file
📬 Contact
Aditya Pratap Singh
LinkedIn • as441438@gmail.com
