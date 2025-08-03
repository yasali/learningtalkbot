#!/usr/bin/env python3
"""
Swedish Voice Chatbot - Fixed Version
====================================

Fixes:
1. Better Swedish speech recognition settings
2. Improved GPT-SW3 prompt formatting
3. Fallback to better Swedish models
4. More robust audio processing
"""

import gradio as gr
import tempfile
import os
import threading
import time
from pathlib import Path
import numpy as np

# Core ML libraries
try:
    import whisper
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from gtts import gTTS
    import pygame
    import soundfile as sf
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("📦 Install with: pip install openai-whisper transformers torch gtts pygame soundfile")
    exit(1)

# Global models and state
whisper_model = None
chat_pipeline = None
current_model_name = None
conversation_context = []
is_speaking = False

def initialize_models():
    """Initialize all AI models at startup"""
    global whisper_model, chat_pipeline, current_model_name
    
    print("🎤 Loading Whisper model for Swedish speech recognition...")
    try:
        # Use small model for better Swedish recognition
        whisper_model = whisper.load_model("small")
        print("✅ Whisper model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return False
    
    print("\n🧠 Loading Swedish language model...")
    
    # Prioritize models that work better for Swedish conversation
    model_options = [
        "neph1/bellman-7b-mistral-instruct-v0.2",    # Best Swedish conversational model
        "AI-Sweden-Models/gpt-sw3-126m",             # Official but can be quirky
        "microsoft/DialoGPT-small",                  # Good fallback
        "distilgpt2"                                 # Basic fallback
    ]
    
    # Check for Hugging Face token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token and os.path.exists('.hf_token'):
        with open('.hf_token', 'r') as f:
            hf_token = f.read().strip()
    
    # Try each model until one works
    for model_name in model_options:
        try:
            print(f"Trying model: {model_name}")
            
            # Load tokenizer and model
            if hf_token and "AI-Sweden-Models" in model_name:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline with better settings
            chat_pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_new_tokens=60,  # Shorter responses
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            current_model_name = model_name
            print(f"✅ Successfully loaded: {model_name}")
            
            # Test the model with a simple Swedish prompt
            if "bellman" in model_name.lower():
                test_prompt = "[INST]Hej, vad heter du?[/INST]"
            elif "gpt-sw3" in model_name.lower():
                test_prompt = "Användare: Hej, vad heter du?\nAssistent:"
            else:
                test_prompt = "Hej, vad heter du?"
                
            test_response = chat_pipeline(test_prompt, max_new_tokens=20, num_return_sequences=1)
            print(f"🧪 Test response: {test_response[0]['generated_text'][:100]}...")
            
            break
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    if chat_pipeline is None:
        print("❌ Could not load any language model!")
        return False
    
    print(f"\n🎉 All models loaded successfully!")
    print(f"📱 Speech Recognition: Whisper (small) - better Swedish")
    print(f"🧠 Language Model: {current_model_name}")
    print(f"🔊 Text-to-Speech: Google TTS")
    
    return True

def transcribe_audio(audio_file):
    """Convert audio to text using Whisper with better Swedish settings"""
    try:
        if audio_file is None:
            return ""
        
        print(f"🎤 Transcribing Swedish audio...")
        
        # Load and transcribe with specific Swedish settings
        result = whisper_model.transcribe(
            audio_file, 
            language="sv",                    # Force Swedish
            task="transcribe",               # Not translate
            temperature=0.0,                 # More deterministic
            no_speech_threshold=0.6,         # Better silence detection
            logprob_threshold=-1.0,          # Less restrictive
            condition_on_previous_text=False # Don't rely on context
        )
        
        text = result["text"].strip()
        
        # Clean up common transcription artifacts
        text = text.replace("Thanks for watching!", "")
        text = text.replace("Tack för att du tittar!", "")
        text = text.replace("♪", "")
        
        if text and len(text) > 2:
            print(f"🗣️ User said: '{text}'")
            return text
        else:
            print("🔇 No clear speech detected")
            return ""
            
    except Exception as e:
        print(f"❌ Speech recognition error: {e}")
        return ""

def get_chatbot_response(user_input):
    """Generate AI response with improved formatting"""
    global conversation_context
    
    try:
        if not user_input.strip():
            return "Jag hörde inget tydligt. Kan du säga det igen?"
        
        # Clean up user input
        user_input = user_input.strip()
        
        print(f"🤖 Generating response for: '{user_input}'")
        
        # Create appropriate prompt based on model type
        if "bellman" in current_model_name.lower():
            # Bellman/Mistral works best with instruction format
            prompt = f"[INST]Du är en vänlig svensk AI-assistent. Svara kort och naturligt på svenska.\n\nAnvändare: {user_input}[/INST]"
            
        elif "gpt-sw3" in current_model_name.lower():
            # GPT-SW3 needs careful prompting to avoid weird formatting
            prompt = f"Detta är en naturlig konversation på svenska.\n\nAnvändare: {user_input}\nAssistent:"
            
        elif "DialoGPT" in current_model_name:
            # DialoGPT works best with direct input
            prompt = user_input
            
        else:
            # Generic format
            prompt = f"Svara på svenska: {user_input}\nSvar:"
        
        # Generate response with better settings
        response = chat_pipeline(
            prompt,
            max_new_tokens=40,           # Shorter responses
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=chat_pipeline.tokenizer.eos_token_id,
            eos_token_id=chat_pipeline.tokenizer.eos_token_id
        )
        
        # Extract and clean generated text
        generated = response[0]['generated_text']
        
        # Clean up response based on model type
        if "bellman" in current_model_name.lower():
            # Extract text after [/INST]
            if "[/INST]" in generated:
                bot_response = generated.split("[/INST]")[-1].strip()
            else:
                bot_response = generated.replace(prompt, "").strip()
                
        elif "gpt-sw3" in current_model_name.lower():
            # Remove the prompt and clean up GPT-SW3 formatting
            bot_response = generated.replace(prompt, "").strip()
            
            # Remove common GPT-SW3 artifacts
            lines = bot_response.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                # Skip lines that look like metadata
                if not any(x in line.lower() for x in ['datum:', 'användare:', 'assistent:', 'kubernetes', 'docker', '20']):
                    if line and len(line) > 3:
                        clean_lines.append(line)
            
            bot_response = ' '.join(clean_lines[:2])  # Take first 2 clean lines max
            
        elif "DialoGPT" in current_model_name:
            # DialoGPT often generates after a newline
            if len(generated) > len(user_input):
                bot_response = generated[len(user_input):].strip()
            else:
                bot_response = "Kan du säga det igen?"
                
        else:
            # Generic cleanup
            bot_response = generated.replace(prompt, "").strip()
            for prefix in ["Svar:", "Assistant:", "AI:", "Bot:"]:
                if bot_response.startswith(prefix):
                    bot_response = bot_response[len(prefix):].strip()
        
        # Final cleanup
        bot_response = bot_response.replace('\n', ' ').strip()
        
        # Ensure it's reasonable Swedish
        if len(bot_response) < 3:
            bot_response = "Ursäkta, kan du upprepa det?"
        elif len(bot_response) > 150:
            # Truncate at sentence boundary
            sentences = bot_response.split('.')
            bot_response = sentences[0] + '.' if sentences[0] else bot_response[:100] + '...'
        
        # Remove any remaining weird characters or formatting
        import re
        bot_response = re.sub(r'[^\w\såäöÅÄÖ.,!?-]', '', bot_response)
        bot_response = bot_response.strip()
        
        if not bot_response:
            bot_response = "Jag förstod inte riktigt. Kan du fråga något annat?"
        
        print(f"🤖 AI Response: '{bot_response}'")
        return bot_response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Ursäkta, jag hade tekniska problem. Försök igen."

def speak_text(text):
    """Convert text to speech and play it"""
    global is_speaking
    
    try:
        if not text or len(text.strip()) == 0:
            return
        
        is_speaking = True
        print(f"🔊 Speaking: '{text}'")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Create TTS audio with slower speed for clarity
        tts = gTTS(text=text, lang='sv', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
            # Play the audio
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            pygame.mixer.music.unload()
            
        # Remove temporary file
        try:
            os.unlink(tmp_file.name)
        except:
            pass
    
    except Exception as e:
        print(f"❌ Error with text-to-speech: {e}")
    finally:
        is_speaking = False

def process_conversation_turn(audio):
    """Process one turn of conversation with better error handling"""
    try:
        if audio is None:
            return "", "Ingen ljudfil. Försök igen.", [], "🎤 Redo att lyssna"
        
        # Step 1: Convert speech to text
        print("🎤 Processing audio...")
        user_text = transcribe_audio(audio)
        
        if not user_text:
            return "", "Kunde inte höra svenska ord. Tala tydligare.", [], "🎤 Redo att lyssna"
        
        # Step 2: Get AI response
        print("🤖 Generating Swedish response...")
        bot_response = get_chatbot_response(user_text)
        
        # Step 3: Speak the response (in background)
        print("🔊 Starting speech...")
        speech_thread = threading.Thread(target=speak_text, args=(bot_response,))
        speech_thread.daemon = True
        speech_thread.start()
        
        # Step 4: Update conversation history
        new_exchange = [user_text, bot_response]
        
        return user_text, bot_response, [new_exchange], "🎤 Redo att lyssna"
        
    except Exception as e:
        error_msg = f"Fel: {str(e)[:50]}"
        print(f"❌ {error_msg}")
        return "", "Tekniskt fel. Försök igen.", [], "❌ Fel - försök igen"

def create_fixed_interface():
    """Create an improved conversational interface"""
    
    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .mic-section {
        text-align: center;
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
    }
    .status-display {
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border: 2px solid #1976D2;
        color: #1976D2;
    }
    """
    
    with gr.Blocks(css=css, title="🇸🇪 Svensk Röstchatbot (Förbättrad)") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🇸🇪 Svensk Röstchatbot</h1>
            <h3>Förbättrad version för naturlig svenska</h3>
            <p>✅ Bättre taligenkänning • ✅ Naturligare svar • ✅ Snabbare</p>
        </div>
        """)
        
        # Status display
        status_display = gr.Textbox(
            value="🎤 Redo att lyssna på svenska",
            label="Status",
            interactive=False,
            elem_classes=["status-display"]
        )
        
        # Microphone section
        gr.HTML("""
        <div class="mic-section">
            <h2>🎤 Tala svenska här!</h2>
            <p>Säg något kort och tydligt</p>
        </div>
        """)
        
        # Audio input
        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="",
            elem_id="main_microphone"
        )
        
        # Latest exchange
        with gr.Row():
            with gr.Column(scale=1):
                last_user_text = gr.Textbox(
                    label="🗣️ Du sa (svenska):",
                    interactive=False,
                    lines=2,
                    placeholder="Din svenska visas här..."
                )
            
            with gr.Column(scale=1):
                last_ai_response = gr.Textbox(
                    label="🤖 AI svarade (svenska):",
                    interactive=False,
                    lines=2,
                    placeholder="AI:ns svenska svar visas här..."
                )
        
        # Full conversation
        conversation_history = gr.Chatbot(
            label="💬 Fullständig konversation",
            height=350,
            type="messages",
            placeholder="🇸🇪 Börja prata svenska för att starta konversationen!"
        )
        
        # Hidden state
        chat_state = gr.State([])
        
        # Process audio input
        def handle_audio_input(audio, history):
            user_text, ai_response, new_exchange, status = process_conversation_turn(audio)
            
            # Update full history
            if new_exchange and new_exchange[0] and new_exchange[0][0]:
                history.extend([
                    {"role": "user", "content": new_exchange[0][0]},
                    {"role": "assistant", "content": new_exchange[0][1]}
                ])
            
            return user_text, ai_response, history, history, status
        
        # Connect audio to processing
        audio_input.change(
            fn=handle_audio_input,
            inputs=[audio_input, chat_state],
            outputs=[last_user_text, last_ai_response, conversation_history, chat_state, status_display]
        )
        
        # Usage tips
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #E8F5E8; border-radius: 10px; border-left: 4px solid #4CAF50;">
            <h3>💡 Tips för bästa resultat:</h3>
            <ul style="font-size: 1.1em; line-height: 1.6;">
                <li><strong>🗣️ Tala tydligt</strong> - inte för snabbt eller långsamt</li>
                <li><strong>📱 Kortare meningar</strong> - 5-10 ord fungerar bäst</li>
                <li><strong>⏳ Vänta</strong> - låt AI:n svara färdigt innan du pratar igen</li>
                <li><strong>🔄 Försök igen</strong> - om det blev fel, säg samma sak igen</li>
            </ul>
            <h4>🎯 Bra testfraser:</h4>
            <p><em>"Hej, vad heter du?" • "Hur mår du?" • "Vad gillar du?" • "Berätta om Sverige"</em></p>
        </div>
        """)
    
    return demo

def main():
    """Main function"""
    print("🇸🇪 FÖRBÄTTRAD SVENSK RÖSTCHATBOT")
    print("=" * 50)
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    print("\n🚀 Starting improved interface...")
    print("🎯 Optimized for Swedish conversation!")
    
    # Create and launch interface
    demo = create_fixed_interface()
    
    try:
        print("🌐 Starting server...")
        print("📱 Open: http://localhost:7861")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,  # Different port
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Hej då!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()