# ai_chatbot.py
import random
import re
import datetime
import json
import os
import sys
import subprocess
import pyperclip  # For copying image to clipboard
import math      # For scientific operations
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


class AIVoiceChatbot:
    def __init__(self):
        self.knowledge_file = "chatbot_knowledge.json"
        self.image_folder = "generated_images"
        os.makedirs(self.image_folder, exist_ok=True)

        # Image generation model (loaded on first use)
        self.image_pipe = None

        # Track last prompt for regenerate
        self.last_prompt = None

        # Load knowledge
        self.knowledge_base = {}
        self.conversation_history = []
        self.load_knowledge()

        # Define intents
        self.intents = self.create_intents()

        # Synonyms for better matching
        self.synonyms = self.create_synonyms()

    def cancellation_callback(self, step: int, timestep: int, latents: torch.FloatTensor):
        """
        Callback used by Stable Diffusion to check if generation should be canceled.
        """
        if hasattr(self, "cancel_generation") and self.cancel_generation:
            raise KeyboardInterrupt("Image generation canceled by user.")

    def create_intents(self):
        return [
            {
                "tag": "greeting",
                "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! I'm happy to see you!"
                ]
            },
            {
                "tag": "goodbye",
                "patterns": ["bye", "goodbye", "see you", "quit", "exit"],
                "responses": [
                    "Goodbye! Have a wonderful day!",
                    "See you later! Come back soon!"
                ]
            },
            {
                "tag": "thanks",
                "patterns": ["thank you", "thanks", "appreciate it"],
                "responses": [
                    "You're very welcome!",
                    "Happy to help!"
                ]
            },
            {
                "tag": "name",
                "patterns": ["what is your name", "who are you"],
                "responses": [
                    "I'm ChatBot, your AI assistant!",
                    "You can call me ChatBot!"
                ]
            },
            {
                "tag": "feelings",
                "patterns": ["how are you", "how are you doing"],
                "responses": [
                    "I'm functioning perfectly, thank you!",
                    "I'm doing great! Ready to help you."
                ]
            },
            {
                "tag": "time",
                "patterns": ["what time is it", "current time"],
                "responses": ["The current time is {time}."]
            },
            {
                "tag": "date",
                "patterns": ["what date is it", "today's date"],
                "responses": ["Today's date is {date}."]
            },
            {
                "tag": "joke",
                "patterns": ["tell me a joke", "make me laugh"],
                "responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my wife she was drawing her eyebrows too high. She looked surprised."
                ]
            },
            {
                "tag": "help",
                "patterns": ["help", "what can you do"],
                "responses": [
                    "I can chat, learn, generate images, calculate, and convert units!"
                ]
            }
        ]

    def create_synonyms(self):
        return {
            "what": ["what", "which", "whats"],
            "is": ["is", "are", "was", "were"],
            "the": ["the", "this", "that"],
            "how": ["how", "in what way"],
            "many": ["many", "much", "number"],
            "tall": ["tall", "height", "high"],
            "long": ["long", "length", "duration"]
        }

    def preprocess_input(self, text):
        return re.sub(r'[^\w\s]', '', text.lower())

    def calculate_similarity(self, input_text, stored_question):
        input_clean = self.preprocess_input(input_text)
        stored_clean = self.preprocess_input(stored_question)
        input_words = input_clean.split()
        stored_words = stored_clean.split()
        if not stored_words:
            return 0
        match_count = sum(1 for w1 in input_words for w2 in stored_words if w1 == w2)
        return match_count / len(stored_words)

    def match_intent(self, user_input):
        best_match = None
        highest_score = 0
        for intent in self.intents:
            for pattern in intent["patterns"]:
                score = self.calculate_similarity(user_input, pattern)
                if score > highest_score and score > 0.3:
                    highest_score = score
                    best_match = intent
        return best_match

    def find_learned_response(self, user_input, threshold=0.6):
        best_match = None
        highest_score = 0
        input_clean = self.preprocess_input(user_input)
        for question, answer in self.knowledge_base.items():
            score = self.calculate_similarity(input_clean, question)
            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = answer
        return best_match, highest_score

    def get_response(self, user_input):
        intent = self.match_intent(user_input)
        if intent:
            response = random.choice(intent["responses"])
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            response = response.replace("{time}", current_time)
            response = response.replace("{date}", current_date)
            return response
        return "I'm not sure how to respond to that."

    def load_image_model(self):
        if self.image_pipe is None:
            print("🖼️ Loading image model (Stable Diffusion)...")
            try:
                self.image_pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32
                )
                self.image_pipe = self.image_pipe.to("cpu")
                print("✅ Image model loaded!")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                return False
        return True

    def open_file(self, filepath):
        try:
            if os.name == 'nt':  # Windows
                os.startfile(filepath)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(['open', filepath])
            else:  # Linux
                subprocess.call(['xdg-open', filepath])
        except Exception as e:
            print(f"Failed to open {filepath}: {e}")

    def copy_image_to_clipboard(self, image_path):
        try:
            image = Image.open(image_path)
            pyperclip.copy(image)
            return True
        except Exception as e:
            print(f"Failed to copy image to clipboard: {e}")
            return False

    def generate_image(self, prompt):
        if not self.load_image_model():
            return "Sorry, I couldn't load the image generator."

        try:
            # Set cancellation flag
            self.cancel_generation = False

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.image_folder, f"image_{timestamp}.png")

            print(f"🎨 Generating: '{prompt}' (this may take 1-3 minutes)...")

            # Generate image with callback
            image = self.image_pipe(
                prompt,
                callback=self.cancellation_callback,
                callback_steps=1
            ).images[0]

            if self.cancel_generation:
                print("🛑 Image generation was canceled.")
                return "Image generation was canceled."

            # Save image
            image.save(filename)
            self.last_prompt = prompt

            # Open image
            try:
                image.show()
            except Exception:
                self.open_file(filename)

            # Copy to clipboard
            print("📋 Copying image to clipboard...")
            if self.copy_image_to_clipboard(filename):
                print("✅ Image copied to clipboard!")
            else:
                print("⚠️ Could not copy to clipboard.")

            result = f"🖼️ Image generated, opened, and copied to clipboard: {filename}"
            print(f"ChatBot: {result}")
            return result

        except KeyboardInterrupt:
            return "🛑 Image generation was canceled by user."

        except Exception as e:
            if self.cancel_generation:
                return "🛑 Image generation was canceled."
            error = f"❌ Image generation failed: {e}"
            print(f"ChatBot: {error}")
            return error

    def add(self, n1, n2):
        return n1 + n2

    def sub(self, n1, n2):
        return n1 - n2

    def mul(self, n1, n2):
        return n1 * n2

    def div(self, n1, n2):
        if n2 == 0:
            return None
        return n1 / n2

    def calculator(self):
        print("\n🧮 Calculator Mode")
        print("Please select operation -\n"
              "1. Add\n"
              "2. Subtract\n"
              "3. Multiply\n"
              "4. Divide\n")

        try:
            sel = int(input("Select operation (1-4): ").strip())
            n1 = float(input("Enter first number: ").strip())
            n2 = float(input("Enter second number: ").strip())

            if sel == 1:
                result = self.add(n1, n2)
                print(f"✅ {n1} + {n2} = {result}")
            elif sel == 2:
                result = self.sub(n1, n2)
                print(f"✅ {n1} - {n2} = {result}")
            elif sel == 3:
                result = self.mul(n1, n2)
                print(f"✅ {n1} * {n2} = {result}")
            elif sel == 4:
                result = self.div(n1, n2)
                if result is not None:
                    print(f"✅ {n1} / {n2} = {result}")
                else:
                    print("❌ Error: Division by zero is not allowed.")
            else:
                print("❌ Invalid selection. Please choose 1-4.")
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")
        print("")  # New line for clarity

    def scientific_operation(self, op, value):
        try:
            if op == "sqrt":
                return math.sqrt(value)
            elif op == "sin":
                return math.sin(math.radians(value))  # Degrees input
            elif op == "cos":
                return math.cos(math.radians(value))
            elif op == "tan":
                return math.tan(math.radians(value))
            elif op == "log":
                return math.log10(value)  # Base 10 log
            elif op == "ln":
                return math.log(value)  # Natural log
            elif op == "exp":
                return math.exp(value)
            else:
                return None
        except Exception:
            return None

    def convert_units(self, value, from_unit, to_unit):
        length_units = {
            "meter": 1.0,
            "m": 1.0,
            "centimeter": 0.01,
            "cm": 0.01,
            "millimeter": 0.001,
            "mm": 0.001,
            "kilometer": 1000.0,
            "km": 1000.0,
            "inch": 0.0254,
            "in": 0.0254,
            "foot": 0.3048,
            "ft": 0.3048,
            "yard": 0.9144,
            "yd": 0.9144,
            "mile": 1609.34,
            "mi": 1609.34
        }

        weight_units = {
            "kilogram": 1.0,
            "kg": 1.0,
            "gram": 0.001,
            "g": 0.001,
            "pound": 0.453592,
            "lb": 0.453592,
            "ounce": 0.0283495,
            "oz": 0.0283495
        }

        temp_units = {
            "c": "celsius",
            "celsius": "celsius",
            "f": "fahrenheit",
            "fahrenheit": "fahrenheit",
            "k": "kelvin",
            "kelvin": "kelvin"
        }

        # Length
        if from_unit in length_units and to_unit in length_units:
            meters = value * length_units[from_unit]
            result = meters / length_units[to_unit]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"

        # Weight
        if from_unit in weight_units and to_unit in weight_units:
            kg = value * weight_units[from_unit]
            result = kg / weight_units[to_unit]
            return f"{value} {from_unit} = {result:.4f} {to_unit}"

        # Temperature
        from_norm = temp_units.get(from_unit.lower(), "")
        to_norm = temp_units.get(to_unit.lower(), "")
        if from_norm and to_norm:
            value = float(value)
            if from_norm == "celsius":
                if to_norm == "fahrenheit":
                    res = (value * 9/5) + 32
                elif to_norm == "kelvin":
                    res = value + 273.15
                else:
                    res = value
            elif from_norm == "fahrenheit":
                if to_norm == "celsius":
                    res = (value - 32) * 5/9
                elif to_norm == "kelvin":
                    res = (value - 32) * 5/9 + 273.15
                else:
                    res = value
            elif from_norm == "kelvin":
                if to_norm == "celsius":
                    res = value - 273.15
                elif to_norm == "fahrenheit":
                    res = (value - 273.15) * 9/5 + 32
                else:
                    res = value
            return f"{value}°{from_unit[0].upper()} = {res:.4f}°{to_unit[0].upper()}"

        return None

    def handle_scientific_or_conversion(self, user_input):
        user_input = user_input.lower().strip()

        # Convert: "convert 10 meters to feet"
        conv_match = re.match(
            r'convert\s+([\-]?\d+\.?\d*)\s*([a-zA-Z°]+)\s+to\s+([a-zA-Z°]+)', user_input)
        if conv_match:
            value = float(conv_match.group(1))
            from_unit = conv_match.group(2).strip("°")
            to_unit = conv_match.group(3).strip("°")
            result = self.convert_units(value, from_unit, to_unit)
            if result:
                print(f"ChatBot: {result}")
                return True
            else:
                print("ChatBot: ❌ Unsupported unit conversion.")
                return True

        # Scientific: sqrt(16), sin(30), log(100), etc.
        sci_match = re.match(r'(sqrt|sin|cos|tan|log|ln|exp)\s*\(\s*([\-]?\d+\.?\d*)\s*\)', user_input)
        if sci_match:
            op = sci_match.group(1)
            val = float(sci_match.group(2))
            result = self.scientific_operation(op, val)
            if result is not None:
                print(f"ChatBot: {op}({val}) = {result:.6f}")
            else:
                print(f"ChatBot: ❌ Invalid input for {op}.")
            return True

        # Power: 2^8 or 5**3
        pow_match = re.match(r'calculate\s+([\-]?\d+\.?\d*)\s*\^\s*([\-]?\d+\.?\d*)', user_input) or \
                   re.match(r'calculate\s+([\-]?\d+\.?\d*)\s*\*\*\s*([\-]?\d+\.?\d*)', user_input)
        if pow_match:
            base = float(pow_match.group(1))
            exp = float(pow_match.group(2))
            result = math.pow(base, exp)
            print(f"ChatBot: {base} ^ {exp} = {result}")
            return True

        # Square root shortcut: calculate sqrt(25)
        sqrt_match = re.match(r'calculate\s+sqrt\s*\(\s*([\-]?\d+\.?\d*)\s*\)', user_input)
        if sqrt_match:
            val = float(sqrt_match.group(1))
            if val >= 0:
                result = math.sqrt(val)
                print(f"ChatBot: √{val} = {result:.6f}")
            else:
                print("ChatBot: ❌ Cannot compute square root of negative number.")
            return True

        return False

    def learn(self, q, a):
        key = self.preprocess_input(q)
        self.knowledge_base[key] = a
        self.save_knowledge()
        return f"I've learned a new response for questions like '{q}'"

    def forget(self, keyword=None):
        if keyword:
            keyword_clean = self.preprocess_input(keyword)
            removed = [k for k in self.knowledge_base if keyword_clean in k]
            for k in removed:
                del self.knowledge_base[k]
            return f"Forgot {len(removed)} items related to '{keyword}'."
        self.knowledge_base.clear()
        self.save_knowledge()
        return "I've forgotten everything I learned."

    def list_knowledge(self):
        if not self.knowledge_base:
            return "I haven't learned anything yet."
        result = []
        for q, a in self.knowledge_base.items():
            wrapped_answer = '\n  '.join([a[i:i+80] for i in range(0, len(a), 80)]) if len(a) > 80 else a
            result.append(f"• Example: '{q}'")
            result.append(f"  Answer:\n  {wrapped_answer}\n")
        return "\n".join(result)

    def save_knowledge(self):
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Save error: {e}")

    def load_knowledge(self):
        if os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"🧠 Loaded {len(self.knowledge_base)} learned facts.")
            except Exception as e:
                print(f"Load error: {e}")

    def chat(self):
        print("🤖 Welcome to your AI ChatBot with Image Generation & Smart Calculator!")
        print("💡 Commands:")
        print("   - 'generate image of [description]'")
        print("   - 'regenerate'")
        print("   - 'learn' / 'forget' / 'list knowledge'")
        print("   - 'calculate' → opens menu")
        print("   - 'sqrt(25)', 'sin(30)', 'convert 10m to ft', 'calculate 2^8'")
        print("   - 'quit'")
        print("-" * 70)

        welcome = "Hi! I can learn, create images, do math, and convert units!"
        print(f"ChatBot: {welcome}")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                # Exit
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("ChatBot: Goodbye! Have a great day!")
                    break

                # Regenerate last image
                if user_input.lower() == 'regenerate':
                    if self.last_prompt:
                        print(f"🔄 Regenerating image for: '{self.last_prompt}'")
                        response = self.generate_image(self.last_prompt)
                        print(f"ChatBot: {response}")
                    else:
                        print("ChatBot: I don't have a previous image to regenerate.")
                    continue

                # Image generation
                img_match = re.search(r'generate image of|create image of|draw|paint', user_input, re.I)
                if img_match:
                    prompt = re.sub(r'^(generate|create|draw|paint)\s+(image of\s*)?', '', user_input, flags=re.I).strip()
                    if prompt:
                        response = self.generate_image(prompt)
                        print(f"ChatBot: {response}")
                    else:
                        print("ChatBot: Please describe what you want me to draw.")
                    continue

                # Scientific, conversion, or expression commands
                if user_input.lower().startswith(('convert', 'calculate')) or \
                   any(op in user_input.lower() for op in ['sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', '^', '**']):
                    if self.handle_scientific_or_conversion(user_input):
                        continue

                # Interactive calculator (only if user types exactly 'calculate')
                if user_input.lower() == 'calculate':
                    self.calculator()
                    continue

                # Learn command
                if user_input.lower() == 'learn':
                    q = input("Teach me a question: ").strip()
                    if not q:
                        print("ChatBot: Question cannot be empty.")
                        continue
                    print("Now enter the answer. Type 'END' on a new line to finish:")
                    lines = []
                    while True:
                        try:
                            line = input()
                            if line.strip().lower() == 'end':
                                break
                            lines.append(line)
                        except EOFError:
                            break
                    a = "\n".join(lines).strip()
                    if a:
                        print(f"ChatBot: {self.learn(q, a)}")
                    else:
                        print("ChatBot: Answer cannot be empty.")
                    continue

                # Forget command
                if user_input.lower().startswith('forget'):
                    parts = user_input.split(' ', 1)
                    keyword = parts[1].strip() if len(parts) > 1 else None
                    print(f"ChatBot: {self.forget(keyword)}")
                    continue

                # List knowledge
                if user_input.lower() == 'list knowledge':
                    print(f"ChatBot:\n{self.list_knowledge()}")
                    continue

                # Check learned knowledge
                learned_answer, score = self.find_learned_response(user_input, threshold=0.6)
                if learned_answer:
                    print(f"ChatBot: {learned_answer}")
                    continue

                # Default response
                response = self.get_response(user_input)
                print(f"ChatBot: {response}")

            except KeyboardInterrupt:
                print("\nChatBot: Goodbye!")
                break
            except Exception as e:
                print(f"ChatBot: An error occurred: {e}")


if __name__ == "__main__":
    try:
        bot = AIVoiceChatbot()
        bot.chat()
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        input("\nPress Enter to exit...")