# KURO-Transformer-Based-Chatbot-for-Real-Time-Emotional-Scoring-and-Profiling

--> Create a venv using pip venv command
--> Using requirements.txt download all necessary packages
--> train the DistilBert by executing the sme.py file 


Now, split termials in two parts:
--> Use one terminal to change directory to /my_chatbot and use 'python manage.py runserver' command to start the Django chat interface
    ->when you type and send buye, the conversation ends and a read.txt file is generated
    
--> Use the second terminal to execute analyse_co.py file which will read the read.txt file and give the anaysis



We have modified the goemotions_train.csv and classified into 3 classes (positive, negative and neutral) on basis of:
      positive = ['admiration', 'amusement', 'approval', 'caring', 'curiosity', 'desire',
                  'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization',
                  'relief', 'surprise']

   negative = [
    'anger', 'annoyance', 'confusion', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
 ]

# neutral = ['neutral']

Thanks for reading, hopw you like it:)

Feel free to reach out on: www.linkedin.com/in/aditya-pratap-singh-8b901a273
                                            or
                                    as441438@gmail.com
