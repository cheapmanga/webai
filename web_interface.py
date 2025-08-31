# web_interface.py
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import re
import math
import threading

# Import your chatbot logic
from ai_chatbot import AIVoiceChatbot

app = Flask(__name__)

# Initialize the chatbot
bot = AIVoiceChatbot()

# Store chat history
chat_history = []

# Ensure generated_images directory exists
IMAGE_FOLDER = 'generated_images'
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Global task tracker
current_task = {"running": False}


def extract_image_path(response):
    """
    Extract the generated image path from bot's response.
    Looks for any generated image file path in the response.
    """
    match = re.search(r'generated[_\\/]images[/\\]image_[^ \n"\)]+\.png', response)
    if match:
        path = match.group(0).replace('\\', '/')
        if os.path.exists(path):
            return path
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/generated_images/<filename>')
def serve_generated_image(filename):
    """Serve generated images from the generated_images folder."""
    return send_from_directory('generated_images', filename)


@app.route("/cancel", methods=["POST"])
def cancel():
    """Cancel ongoing image generation."""
    if current_task["running"]:
        bot.cancel_generation = True
        return jsonify({"status": "canceled", "message": "🛑 Image generation canceled."})
    return jsonify({"status": "idle", "message": "No active task to cancel."})


@app.route("/chat", methods=["POST"])
def chat():
    # Prevent multiple concurrent tasks
    if current_task["running"]:
        return jsonify({
            "response": "Still processing a request. Click 'Stop' to cancel it first.",
            "show_stop": True
        })

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please say something.", "show_stop": False})

    # Add user message to chat history
    chat_history.append({"sender": "user", "text": user_message})

    # Check if this is an image generation command
    is_image_command = any(cmd in user_message.lower() for cmd in [
        'generate image of', 'create image of', 'draw', 'paint'
    ]) and 'regenerate' not in user_message.lower()

    prompt = ""
    if is_image_command:
        prompt = re.sub(
            r'^(generate|create|draw|paint)\s+image of\s*',
            '', user_message, flags=re.I
        ).strip()
        if not prompt:
            return jsonify({"response": "Please describe what you want me to draw.", "show_stop": False})

    try:
        if is_image_command:
            # Start background generation
            current_task["running"] = True
            bot.cancel_generation = False  # Reset cancellation flag

            def run_generation():
                try:
                    result = bot.generate_image(prompt)
                    # We can't send this back directly; client will need to send another message to continue
                finally:
                    current_task["running"] = False

            thread = threading.Thread(target=run_generation, daemon=True)
            thread.start()

            # Respond immediately to show feedback
            return jsonify({
                "response": f"🎨 Generating image of *{prompt}*...\n\n⏳ This may take 1–3 minutes.\n\nClick **Stop** below to cancel.",
                "image_url": None,
                "show_stop": True
            })

        elif user_message.lower() == 'regenerate':
            if bot.last_prompt:
                current_task["running"] = True
                bot.cancel_generation = False

                def run_regeneration():
                    try:
                        result = bot.generate_image(bot.last_prompt)
                    finally:
                        current_task["running"] = False

                thread = threading.Thread(target=run_regeneration, daemon=True)
                thread.start()

                return jsonify({
                    "response": f"🔄 Regenerating image for: *{bot.last_prompt}*\n\n⏳ Please wait...",
                    "image_url": None,
                    "show_stop": True
                })
            else:
                return jsonify({
                    "response": "I don't have a previous image to regenerate.",
                    "show_stop": False
                })

        else:
            # Handle all other fast commands
            response = handle_non_image_commands(user_message)
            image_path = extract_image_path(response)
            image_url = None
            if image_path and os.path.exists(image_path):
                filename = os.path.basename(image_path)
                image_url = url_for('serve_generated_image', filename=filename)

            return jsonify({
                "response": response,
                "image_url": image_url,
                "show_stop": False
            })

    except Exception as e:
        current_task["running"] = False
        return jsonify({
            "response": f"An error occurred: {str(e)}",
            "show_stop": False
        })


def handle_non_image_commands(user_message):
    """Handle non-image commands (fast responses)"""
    try:
        response = bot.get_response(user_message)

        if user_message.lower() in ['quit', 'exit', 'bye']:
            return "Goodbye! Have a great day!"
        elif user_message.lower() == 'list knowledge':
            return bot.list_knowledge()
        elif user_message.lower().startswith('forget'):
            keyword = user_message.split(' ', 1)[1].strip() if len(user_message.split(' ', 1)) > 1 else None
            return bot.forget(keyword)
        elif user_message.lower() == 'learn':
            return "The 'learn' command is not supported in the web UI yet."
        else:
            # Try scientific or conversion commands
            if any(kw in user_message.lower() for kw in ['convert', 'calculate', 'sqrt', '^', '**']) or \
               re.match(r'(sqrt|sin|cos|tan|log|ln|exp)\s*\(', user_message, re.I):
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                try:
                    bot.handle_scientific_or_conversion(user_message)
                finally:
                    sys.stdout = old_stdout
                output = captured_output.getvalue().strip()
                return output.replace("ChatBot: ", "") if output else "I couldn't compute that."
            else:
                learned_answer, score = bot.find_learned_response(user_message, threshold=0.6)
                if learned_answer:
                    return learned_answer
                elif "not sure" in response:
                    return "I'm not sure how to respond to that."
                return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    print("🌍 Starting AI ChatBot Web Interface...")
    print("👉 Open http://localhost:5000 in your browser")
    print("💡 Tip: Type 'generate image of a fantasy castle' to test image generation.")
    app.run(debug=True, port=5000)