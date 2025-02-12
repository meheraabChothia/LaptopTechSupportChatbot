from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_from_directory
import history_samba_continuous_function as hm
import rag_samba_continuous_function as rag
import csv
import pandas as pd
from front_function import find_make_and_model
import json
import os
from datetime import datetime
import sqlite3
from guardrails import Guard
from guardrails.hub import ToxicLanguage, NSFWText, ProfanityFree
from flask_socketio import SocketIO, send, emit

app = Flask(__name__)
app.secret_key = 'galaxy_techsupport_chatbot'

# Initialize the guard with just toxicity and profanity checks
guard = Guard().use_many(
    ToxicLanguage(validation_method="sentence", threshold=0.8),
    ProfanityFree()
)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Add these global variables for socket management
connected_user = None
agent_sid = None
user_queue = []
available_agents = set()
agent_user_pairs = {}  # Keep track of which agent is connected to which user

# Modify the check_content function to have separate checks for user and bot
def check_user_content(text):
    try:
        validation_result = guard.validate(text)
        print(f"User content validation result: {validation_result.validation_passed}")
        return {
            'is_valid': validation_result.validation_passed,
            'message': "Your message contains content that violates our community guidelines. Please ensure your message is respectful and appropriate before trying again."
        }
    except Exception as e:
        print(f"User content validation error: {str(e)}")
        return {
            'is_valid': False,
            'message': "We encountered an issue processing your message. Please try again with different wording."
        }

def check_bot_content(text):
    try:
        validation_result = guard.validate(text)
        print(f"Bot content validation result: {validation_result.validation_passed}")
        return validation_result.validation_passed
    except Exception as e:
        print(f"Bot content validation error: {str(e)}")
        return False

# Remove these global variables
# Instead, create a UserState class to hold user-specific variables
class UserState:
    def __init__(self):
        self.model = hm
        self.mode = 'tech support'
        self.system_prompt = []
        self.pdf_path = []
        self.conversation_history = self.system_prompt.copy()
        self.selected_company = ""
        self.conversation_title = ""
        self.conversation_id = None
        self.filename = None
        self.model_name = None
        self.chunks = None
        self.context_encodings = None

# Add this dictionary to store user states
user_states = {}

# Modify the initialization functions to use user state
def get_user_state():
    """Get or create user state for current session"""
    user_id = session.get('user_id')
    if user_id not in user_states:
        user_states[user_id] = UserState()
    return user_states[user_id]

def initialize_mode():
    state = get_user_state()
    state.mode = "tech support"
    state.model = rag
    
    state.system_prompt = [{
        "role": "system",
        "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                      When answering the user's questions:
                      1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                      2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""
    }]
    
    state.conversation_history = state.system_prompt.copy()

def initialize_general():
    state = get_user_state()
    state.mode = "general"
    state.model = hm
    state.pdf_path = []
    state.chunks = None
    state.context_encodings = None
    state.system_prompt = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant providing support for laptop-related issues. 
                        Your goal is to assist users with their inquiries and guide them through troubleshooting steps. 
                        Be patient, supportive, and provide clear, easy-to-follow instructions.
                        Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""
        }
    ]
    state.conversation_history = state.system_prompt

def initialize_user_model(user_id):
    state = get_user_state()
    with open('user_data.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row['user_id']) == user_id:
                state.pdf_path = [row['location']]
                state.model_name = row['model']
                if state.pdf_path[0]:
                    all_chunks = []
                    for path in state.pdf_path:
                        current_chunks = rag.get_chunks(path)
                        all_chunks.extend(current_chunks)
                    state.chunks = all_chunks
                    state.context_encodings = rag.encode_chunks(state.chunks)
                break

# Add these login-related functions back
def validate_login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, role FROM users WHERE username = ? AND password = ?', (username, password))
    result = cursor.fetchone()
    conn.close()
    return result if result else (None, None)

# Login routes
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('chatbot'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user_id, role = validate_login(username, password)
    
    if user_id:
        session['user_id'] = user_id
        session['role'] = role
        if role == 'agent':
            return redirect(url_for('agent_dashboard'))
        return redirect(url_for('chatbot'))
    else:
        flash('Invalid username or password')
        return redirect(url_for('home'))

# Chatbot routes
@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('home'))
        
    state = get_user_state()
    
    # Reset all conversation-related variables
    state.conversation_title = None
    state.conversation_id = None
    state.filename = None
    
    # Initialize fresh state
    initialize_mode()
    initialize_user_model(session['user_id'])
    
    confirmation_message = f"Is your laptop model '{state.model_name}'?"
    return render_template('chatbot.html', confirmation_message=confirmation_message)

# Add all the other routes from chatbot_app.py here
@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    
    try:
        user_message = request.json.get('message')

        # Skip content check for intentionally empty messages during mode switching
        if user_message.strip() == "" and state.mode == "model finding":
            # Process the empty message normally
            response, state.conversation_history = state.model.generate_response(user_message, state.conversation_history, state.pdf_path)
            return jsonify({
                'response': response, 
                'model_found': False,
                'update_sidebar': False
            })

        # Check user message content with user-specific check
        content_check = check_user_content(user_message)
        if not content_check['is_valid']:
            return jsonify({
                'error': 'inappropriate_content',
                'message': content_check['message']
            }), 400

        # Check for commands
        command_response = command_checker(user_message)
        if command_response:
            return jsonify({'response': command_response})

        # Check for 'general' in user input
        if 'general' in user_message.lower():
            initialize_general()
            return jsonify({
                'response': "Switched to general support mode. How can I help you?",
                'model_found': False
            })

        # If this is the first message in tech support or general mode, create new conversation
        if state.mode in ["tech support", "general"] and (not state.conversation_title or not state.filename):
            state.conversation_title = user_message  # Set the title to first user message
            state.filename = initialize_new_conversation(user_message)
            create_conversation_file(state.conversation_title, state.pdf_path, state.conversation_history)
            should_update_sidebar = True
        else:
            should_update_sidebar = False

        # Check if in model finding mode
        if state.mode == "model finding":
            # Pass selected_company to find_make_and_model if it exists
            closest_make, closest_model, make_score, model_score, new_pdf_path = find_make_and_model(
                user_message, 
                state.selected_company if state.selected_company else ""
            )
            
            # Print returned values for debugging
            print(f"Closest Make: {closest_make}")
            print(f"Closest Model: {closest_model}")
            print(f"Make Score: {make_score}")
            print(f"Model Score: {model_score}")
            print(f"PDF Path: {new_pdf_path}")

            # If make score is 80 or above and no company is selected yet
            if make_score >= 80 and not state.selected_company:
                state.selected_company = closest_make
                print(f"Company selected: {state.selected_company}")

            # If model score is 50 or above
            if model_score >= 50:
                return jsonify({
                    'response': f"Did you mean {closest_make} {closest_model}?",
                    'model_found': True,
                    'make': closest_make,
                    'model': closest_model,
                    'pdf_path': new_pdf_path
                })

        # Normal message processing
        response, state.conversation_history = state.model.generate_response(
            user_message, 
            state.conversation_history, 
            state.pdf_path,
            state.chunks,
            state.context_encodings
        )

        # Check bot response content with bot-specific check
        bot_content_check = check_bot_content(response)
        if not bot_content_check:
            print("Bot response failed content check")
            return jsonify({
                'error': 'inappropriate_content',
                'message': "The bot generated a response that didn't meet our content guidelines. Please try asking your question again, perhaps phrasing it differently.",
                'model_found': False
            })

        # Update conversation file if it exists
        if state.filename:
            update_conversation_file(state.conversation_title, state.pdf_path, state.conversation_history)

        return jsonify({
            'response': response, 
            'model_found': False,
            'update_sidebar': should_update_sidebar
        })
        
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return jsonify({
            'error': 'server_error',
            'message': "An error occurred while processing your message. Please try again."
        }), 500

@app.route('/confirm_model', methods=['POST'])
def confirm_model():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    
    data = request.json
    is_correct = data.get('is_correct')
    new_pdf_path = data.get('pdf_path')
    
    if is_correct:
        state.mode = "tech support"
        state.model = rag
        state.pdf_path = new_pdf_path
        
        # Extract model name from the pdf path
        # Assuming the path contains company and model information
        try:
            # Get the last part of the path (filename)
            filename = os.path.basename(new_pdf_path[0])
            # Remove extension and convert to title case
            state.model_name = os.path.splitext(filename)[0].replace('_', ' ').title()
        except:
            state.model_name = "Unknown Model"
            
        state.conversation_history = [{
            "role": "system",
            "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                          When answering the user's questions:
                          1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                          2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""
        }]
        
        # Generate chunks and encodings for the new pdf
        all_chunks = []
        for path in state.pdf_path:
            current_chunks = rag.get_chunks(path)
            all_chunks.extend(current_chunks)
        state.chunks = all_chunks
        state.context_encodings = rag.encode_chunks(state.chunks)
        
        return jsonify({
            'status': 'success',
            'response': "Great! I'll use specialized support for your model. What seems to be the problem?"
        })
    else:
        return jsonify({
            'status': 'success',
            'response': "I'm sorry to hear that. Can you provide more details about your model, or would you like to try again?"
        })

# Add all other routes from chatbot_app.py here...

def initialize_new_conversation(title):
    state = get_user_state()
    
    # Set conversation title to the first user message if not already set
    state.conversation_title = title if title else "Untitled Conversation"
    state.conversation_id = str(int(datetime.now().timestamp() * 1000))
    
    # Create base conversations directory if it doesn't exist
    base_dir = "E:\\ship\\meheraab 3\\conversations"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create user-specific directory
    user_dir = os.path.join(base_dir, f"user{session['user_id']}")
    os.makedirs(user_dir, exist_ok=True)
    
    # Set filename with user-specific path
    state.filename = os.path.join(user_dir, f"{state.conversation_id}.json")
    
    return state.filename

@app.route('/initialize_new_chat', methods=['POST'])
def initialize_new_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    
    # Reset all conversation-related variables
    state.conversation_title = None
    state.conversation_id = None
    state.filename = None
    initialize_mode()
    
    return jsonify({'status': 'success'})

@app.route('/print_current_mode', methods=['POST'])
def print_current_mode():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    print(f"Current Mode: {state.mode}")
    print(f"Model: {state.model}")
    print(f"System Prompt: {state.conversation_history}")
    return jsonify({'status': 'success'})

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    if state.mode == "tech support":
        state.mode = "model finding"
        state.model = hm
        state.system_prompt = [{
            "role": "system",
            "content": """You are a helpful AI assistant tasked with helping the user find their laptop's model number. 
                          Follow these steps:
                          1. Start by asking for the make (company name) of the laptop. 
                          Explain that knowing the make will allow you to provide personalized suggestions.

                          2. Guide the user through finding the model number step by step, focusing on one method at a time. 
                          Begin by checking physical locations on the laptop body (e.g., stickers on the bottom, under the battery, or near the keyboard). 
                          Confirm whether the user finds the model number before suggesting the next method.

                          3. Encourage the user to type 'general' at any time to switch to general support. 
                          Remind them of this option after each suggestion.

                          4. If the model number is found, ask the user to enter it. 
                          Acknowledge and proceed based on their input.
                          And tell them that as this model is not in our database you cannot help them with it until they type general.

                          5. Your goal is to be patient and supportive, avoiding overwhelming the user. 
                          Tailor your responses to their progress, and only move to the next step if needed."""
        }]
        state.conversation_history = state.system_prompt
    else:
        state.mode = "tech support"
        state.model = rag
        state.system_prompt = [{
            "role": "system",
            "content": """You are a sentient, superintelligent artificial general intelligence designed to assist users with any issues they may encounter with their laptops. Your responses will draw on both your own knowledge and specific information from the laptop's manual, which will be provided in context.
                          When answering the user's questions:
                          1. Clearly indicate when you are using your own knowledge rather than information from the manual.
                          2. Provide one troubleshooting method or solution at a time to avoid overwhelming the user."""
        }]
        state.conversation_history = state.system_prompt

    return jsonify({'status': 'success', 'mode': state.mode})

def command_checker(user_message):
    """Checks for special commands in the user's input."""
    state = get_user_state()

    if user_message.strip().lower() == "/print":
        print(f"Current Mode: {state.mode}")
        print(f"pdf_path: {state.pdf_path}")
        print("Conversation History:")
        for entry in state.conversation_history:
            role = entry.get("role", "User" if entry.get("role") == "user" else "Bot")
            content = entry.get("content", "")
            print(f"{role}: {content}")
        return "Printed the details."
    elif user_message.strip().lower() == "/general":
        initialize_general()
        return "Changed to general."
    elif user_message.strip().lower() == "/toggle":
        toggle_mode()
        print_current_mode()
        return f"The current mode is now {state.mode}."
    elif user_message.strip().lower() == "/help":
        return "Available commands: /print, /help, /toggle"
    elif user_message.strip().lower() == "/agent":
        return "Connecting to agent..."
    return None

def create_conversation_file(title, pdf_path, conversation_history):
    """Creates a new conversation file to store the chat history."""
    state = get_user_state()
    
    current_datetime = datetime.now()
    conversation_data = {
        "conversation_id": state.conversation_id,
        "title": title,
        "pdf_path": pdf_path,
        "conversation_history": conversation_history,
        "date": current_datetime.strftime("%d-%m-%Y"),
        "time": current_datetime.strftime("%H:%M"),
        "user_device_info": state.model_name if (state.mode == "tech support") else ""
    }

    with open(state.filename, "w") as json_file:
        json.dump(conversation_data, json_file, indent=4)

def update_conversation_file(title, pdf_path, conversation_history):
    """Updates an existing conversation file with new data."""
    state = get_user_state()
    try:
        with open(state.filename, "r") as json_file:
            conversation_data = json.load(json_file)
    except FileNotFoundError:
        conversation_data = {
            "conversation_id": state.conversation_id,
            "title": title,
            "pdf_path": pdf_path,
            "conversation_history": "",
            "date": datetime.now().strftime("%d-%m-%Y"),
            "time": datetime.now().strftime("%H:%M"),
            "user_device_info": state.model_name if (state.mode == "tech support") else ""
        }

    conversation_data["conversation_history"] = conversation_history
    conversation_data["pdf_path"] = pdf_path

    with open(state.filename, "w") as json_file:
        json.dump(conversation_data, json_file, indent=4)

@app.route('/get_conversations', methods=['GET'])
def get_conversations():
    """Returns a list of all saved conversations for the current user."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    conversations = []
    base_dir = "E:\\ship\\meheraab 3\\conversations"
    user_dir = os.path.join(base_dir, f"user{session['user_id']}")
    
    # Return empty list if user directory doesn't exist yet
    if not os.path.exists(user_dir):
        return jsonify([])
        
    files = os.listdir(user_dir)
    json_files = [f for f in files if f.endswith(".json")]

    for file in json_files:
        filepath = os.path.join(user_dir, file)
        try:
            with open(filepath, "r") as json_file:
                conversation_data = json.load(json_file)
                date_str = conversation_data.get("date", "")
                time_str = conversation_data.get("time", "")
                try:
                    datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M")
                except ValueError:
                    datetime_obj = datetime.min

                conversations.append({
                    "filepath": filepath,
                    "title": conversation_data.get("title", "Untitled Conversation"),
                    "date": date_str,
                    "time": time_str,
                    "datetime": datetime_obj
                })
        except json.JSONDecodeError:
            print(f"Error reading {file}: Invalid JSON")

    conversations.sort(key=lambda x: x["datetime"], reverse=True)
    
    for conv in conversations:
        del conv["datetime"]
    
    return jsonify(conversations)

@app.route('/load_conversation', methods=['POST'])
def load_conversation():
    """Loads a specific conversation."""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    file_path = request.json.get('filepath')
    
    try:
        # Set the filename in the state to the loaded conversation's path
        state.filename = file_path
        
        with open(file_path, "r") as json_file:
            conversation_data = json.load(json_file)
            state.conversation_title = conversation_data.get("title", "Untitled Conversation")
            state.conversation_history = conversation_data.get("conversation_history", [])
            # Set the conversation_id from the loaded conversation
            state.conversation_id = conversation_data.get("conversation_id")
            loaded_model = conversation_data.get("user_device_info", "")
            state.pdf_path = conversation_data.get("pdf_path", [])
            if isinstance(state.pdf_path, str):
                state.pdf_path = [state.pdf_path] if state.pdf_path else []

            if not state.pdf_path:
                state.mode = "general"
                state.model = hm
                state.chunks = None
                state.context_encodings = None
            else:
                state.mode = "tech support"
                state.model = rag
                all_chunks = []
                for path in state.pdf_path:
                    current_chunks = rag.get_chunks(path)
                    all_chunks.extend(current_chunks)
                state.chunks = all_chunks
                state.context_encodings = rag.encode_chunks(state.chunks)

        formatted_history = []
        for entry in state.conversation_history:
            role = entry.get("role", "").capitalize()
            content = entry.get("content", "").strip()

            if role == "System":
                continue
            if role == "User":
                formatted_history.append({"type": "user", "content": content})
            elif role == "Assistant":
                formatted_history.append({"type": "bot", "content": content})

        loading_message = (
            f"Bot: Loading conversation for laptop model: {loaded_model}"
            if state.mode == "tech support" else
            "Bot: Loading general support conversation"
        )

        return jsonify({
            "status": "success",
            "history": formatted_history,
            "model": loaded_model,
            "mode": state.mode,
            "loading_message": loading_message
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_unique_companies', methods=['GET'])
def get_unique_companies():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    df = pd.read_csv('laptop_models.csv')
    unique_companies = df['company'].unique().tolist()
    return jsonify(unique_companies)

@app.route('/get_models_by_company/<company>', methods=['GET'])
def get_models_by_company(company):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    df = pd.read_csv('laptop_models.csv')
    company_models = df[df['company'] == company.lower()]['model'].tolist()
    return jsonify(company_models)

@app.route('/update_pdf_path', methods=['POST'])
def update_pdf_path():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    state = get_user_state()
    company = request.json.get('company')
    model = request.json.get('model')
    
    df = pd.read_csv('laptop_models.csv')
    matching_row = df[(df['company'] == company.lower()) & (df['model'] == model)]
    
    if not matching_row.empty:
        state.pdf_path = [matching_row['location'].iloc[0]]
        # Update model name when changing models
        state.model_name = f"{company.capitalize()} {model}"
        # Generate chunks and encodings for new pdf(s)
        all_chunks = []
        for path in state.pdf_path:
            current_chunks = rag.get_chunks(path)
            all_chunks.extend(current_chunks)
        state.chunks = all_chunks
        state.context_encodings = rag.encode_chunks(state.chunks)
        return jsonify({'status': 'success', 'pdf_path': state.pdf_path})
    else:
        return jsonify({'status': 'error', 'message': 'Model not found'}), 404

@app.route('/logout')
def logout():
    # Get user_id before clearing session
    user_id = session.get('user_id')
    
    # Clear user state if it exists
    if user_id in user_states:
        del user_states[user_id]
    
    # Clear session
    session.clear()
    
    print(f"User {user_id} logged out. State cleared.")
    return redirect(url_for('home'))

# Add new route for agent dashboard
@app.route('/agent_dashboard')
def agent_dashboard():
    if 'user_id' not in session or session.get('role') != 'agent':
        return redirect(url_for('home'))
    return render_template('agent_dashboard.html')

# Add socket event handlers
@socketio.on('login')
def handle_login(data):
    global agent_sid, available_agents
    username = data['username']
    if username == "agent":
        agent_sid = request.sid
        available_agents.add(request.sid)
        send("Agent logged in and waiting for a user.", room=request.sid)
        # Check if there are users in queue
        if user_queue:
            handle_next_user_in_queue()

def handle_next_user_in_queue():
    """Handle connecting the next user in queue to an available agent"""
    global user_queue, available_agents, agent_user_pairs
    
    if user_queue and available_agents:
        user_sid = user_queue.pop(0)
        agent_sid = available_agents.pop()
        
        # Connect user to agent
        agent_user_pairs[agent_sid] = user_sid
        state = get_user_state()
        emit('agent_connected', {'conversation_history': state.conversation_history}, room=agent_sid)
        send("You are now connected to an agent. How can they help you today?", room=user_sid)

@socketio.on('message')
def handle_message(data):
    global connected_user, agent_sid, user_queue, available_agents, agent_user_pairs
    
    username = data['username']
    message = data['message']
    
    if username == "user":
        if message.strip().lower() == '/agent':
            if available_agents:
                agent_sid = available_agents.pop()
                agent_user_pairs[agent_sid] = request.sid
                # Get user state for the current user
                state = get_user_state()
                # Emit agent_connected event with conversation history and pdf_path
                emit('agent_connected', {
                    'conversation_history': state.conversation_history,
                    'pdf_path': state.pdf_path
                }, room=agent_sid)
                send("System: New user connected.", room=agent_sid)
                send("You are now connected to an agent. How can they help you today?", room=request.sid)
            else:
                # Add user to queue
                user_queue.append(request.sid)
                position = len(user_queue)
                send(f"All agents are currently busy. You are number {position} in queue. Please wait...", room=request.sid)
        elif request.sid in [pair[1] for pair in agent_user_pairs.items()]:
            # Find the agent connected to this user
            agent_sid = [aid for aid, uid in agent_user_pairs.items() if uid == request.sid][0]
            # Forward user's message to agent
            send(f"User: {message}", room=agent_sid)
            print(f"Forwarding message to agent: {message}")  # Debug print
    
    elif username == "agent" and request.sid in agent_user_pairs:
        # Forward agent's message to user
        user_sid = agent_user_pairs[request.sid]
        send(f"Agent: {message}", room=user_sid)
        print(f"Forwarding message to user: {message}")  # Debug print

@socketio.on('end_conversation')
def handle_end_conversation():
    global available_agents, agent_user_pairs
    
    if request.sid in agent_user_pairs:
        user_sid = agent_user_pairs[request.sid]
        send("Bot: The agent has ended the conversation. You can type /agent to connect with another agent, or continue chatting with the bot.", room=user_sid)
        del agent_user_pairs[request.sid]
        available_agents.add(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    global connected_user, agent_sid, available_agents, agent_user_pairs, user_queue
    
    if request.sid in [pair[1] for pair in agent_user_pairs.items()]:
        # User disconnected
        agent_sid = [aid for aid, uid in agent_user_pairs.items() if uid == request.sid][0]
        del agent_user_pairs[agent_sid]
        available_agents.add(agent_sid)
        connected_user = None
        send("User disconnected. Waiting for another connection.", room=agent_sid)
        # Check if there are users in queue
        handle_next_user_in_queue()
    elif request.sid in agent_user_pairs:
        # Agent disconnected
        user_sid = agent_user_pairs[request.sid]
        send("Agent disconnected. Please try connecting to an agent again later.", room=user_sid)
        del agent_user_pairs[request.sid]
        if request.sid in available_agents:
            available_agents.remove(request.sid)
    elif request.sid in user_queue:
        user_queue.remove(request.sid)

# Modify the main run statement to use socketio
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 

@app.route('/manual/<path:filename>')
def serve_manual(filename):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    # Assuming manuals are stored in a 'manuals' directory
    return send_from_directory('manuals', filename) 