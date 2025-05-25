import openai
import os
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import FileSystemStorage 

# Replace with your OpenAI API key
OPENAI_API_KEY = "your_openai_api_key_here"
openai.api_key = OPENAI_API_KEY

@csrf_exempt
def chat_response(request):
    session_id = request.session.session_key
    if not session_id:
        request.session.create()
        session_id = request.session.session_key

    if 'conversation' not in request.session:
        request.session['conversation'] = []
        request.session['user_name'] = None
        request.session['user_messages'] = []

    if request.method == "POST":
        user_message = request.POST.get('message', '').strip()
        
        # Add user message to user_messages list
        user_messages = request.session.get('user_messages', [])
        user_messages.append(user_message)
        request.session['user_messages'] = user_messages

        if "my name is" in user_message.lower():
            name = user_message.split("my name is")[-1].strip().capitalize()
            request.session['user_name'] = name
            bot_message = f"Nice to meet you, {name}! How are you feeling today?"
        else:
            name = request.session.get('user_name')
            conversation_history = request.session['conversation']
            conversation_history.append({"role": "user", "content": user_message})

            if "bye" in user_message.lower():
                try:
                    # Save conversation to file
                    try:
                        # Ensure directory exists
                        log_dir = settings.CONVERSATION_LOG_DIR
                        os.makedirs(log_dir, exist_ok=True)
                        log_path = os.path.join(log_dir, "read.txt")
                        with open(log_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(request.session["user_messages"]))
                        # log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'conversation_logs')
                        # os.makedirs(log_dir, exist_ok=True)
                        
                        # # Write only user messages to file
                        # with open(os.path.join(log_dir, 'read.txt'), 'w', encoding='utf-8') as f:
                        #     f.write('\n'.join(user_messages))
                    except Exception as e:
                        print(f"Error saving file: {e}")
                    
                    # Analyze conversation
                    full_conversation = [
                        {"role": "system", "content": "You are a caring, mature, and empathetic psychiatrist. Your goal is to learn more about the user's life, mental health, and personality. Ask leading and relevant questions about their life and work."}
                    ] + conversation_history
                    
                    analysis_prompt = "Based on the following conversation, give a score out of 10 for the user's mental health and highlight key points about their mental health and personality in 3 points of not more than 6 words each:\n"
                    analysis_prompt += "\n".join([f"{msg['role']}: {msg['content']}" for msg in full_conversation])
                    
                    analysis_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system", "content": "You are a mental health analysis assistant."},
                                 {"role": "user", "content": analysis_prompt}]
                    )

                    mental_health_analysis = analysis_response['choices'][0]['message']['content'].strip()
                    bot_message = (
                        "I respect your decision to end the conversation. "
                        "Remember, support is available whenever you need it. Take care of yourself!\n\n"
                        f"**Mental Health Analysis:** {mental_health_analysis}"
                    )

                    # Clear the session
                    request.session['conversation'] = []
                    request.session['user_messages'] = []
                except Exception as e:
                    bot_message = f"Error during analysis: {str(e)}"
            else:
                # Regular chat flow
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a caring, mature, and empathetic psychiatrist. Your goal is to learn more about the user's life, mental health, and personality. Ask leading and relevant questions about their life and work."},
                            *conversation_history
                        ]
                    )
                    bot_message = response['choices'][0]['message']['content'].strip()
                except Exception as e:
                    bot_message = f"Error: {str(e)}"

                # Add bot response to conversation
                conversation_history.append({"role": "assistant", "content": bot_message})

            request.session['conversation'] = conversation_history

        return JsonResponse({"response": bot_message})

    return render(request, 'chat/chat.html')