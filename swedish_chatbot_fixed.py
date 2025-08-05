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
# Add new state management for better audio control
is_listening = False
should_stop = False
audio_lock = threading.Lock()
current_audio_thread = None

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
    global is_listening, should_stop
    
    try:
        if audio_file is None or should_stop:
            return ""
        
        is_listening = True
        print(f"🎤 Transcribing Swedish audio...")
        
        # Quick check for audio file validity
        try:
            import soundfile as sf
            data, sample_rate = sf.read(audio_file)
            if len(data) < sample_rate * 0.5:  # Less than 0.5 seconds
                print("🔇 Audio too short, skipping")
                return ""
        except Exception:
            print("⚠️ Could not validate audio file")
        
        if should_stop:
            return ""
        
        # Load and transcribe with specific Swedish settings
        result = whisper_model.transcribe(
            audio_file, 
            language="sv",                    # Force Swedish
            task="transcribe",               # Not translate
            temperature=0.0,                 # More deterministic
            no_speech_threshold=0.6,         # Better silence detection
            logprob_threshold=-1.0,          # Less restrictive
            condition_on_previous_text=False, # Don't rely on context
            verbose=False                    # Reduce output noise
        )
        
        if should_stop:
            return ""
        
        text = result["text"].strip()
        
        # Clean up common transcription artifacts
        text = text.replace("Thanks for watching!", "")
        text = text.replace("Tack för att du tittar!", "")
        text = text.replace("♪", "")
        text = text.replace("  ", " ")  # Multiple spaces
        
        if text and len(text) > 2:
            print(f"🗣️ User said: '{text}'")
            return text
        else:
            print("🔇 No clear speech detected")
            return ""
            
    except Exception as e:
        print(f"❌ Speech recognition error: {e}")
        return ""
    finally:
        is_listening = False

def get_chatbot_response(user_input):
    """Generate AI response with improved formatting and context"""
    global conversation_context, should_stop
    
    try:
        if not user_input.strip() or should_stop:
            return "Jag hörde inget tydligt. Kan du säga det igen?"
        
        # Clean up user input
        user_input = user_input.strip()
        
        print(f"🤖 Generating response for: '{user_input}'")
        
        # Add to conversation context for better continuity
        conversation_context.append({"role": "user", "content": user_input})
        
        # Keep context manageable (last 6 turns)
        if len(conversation_context) > 6:
            conversation_context = conversation_context[-6:]
        
        # Create context-aware prompt based on model type
        if "bellman" in current_model_name.lower():
            # Bellman/Mistral works best with instruction format
            context_str = ""
            if len(conversation_context) > 1:
                recent_context = conversation_context[-3:]  # Last 3 exchanges
                context_str = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in recent_context[:-1]])
                context_str = f"Tidigare konversation:\n{context_str}\n\n"
            
            prompt = f"[INST]Du är en vänlig svensk AI-assistent som har naturliga konversationer. Svara kort och naturligt på svenska.\n\n{context_str}Användare: {user_input}[/INST]"
            
        elif "gpt-sw3" in current_model_name.lower():
            # GPT-SW3 needs careful prompting to avoid weird formatting
            context_str = ""
            if len(conversation_context) > 1:
                context_str = f"Tidigare sa användaren: {conversation_context[-2]['content'] if len(conversation_context) >= 2 else ''}\n"
            
            prompt = f"Detta är en naturlig konversation på svenska.\n{context_str}\nAnvändare: {user_input}\nAssistent:"
            
        elif "DialoGPT" in current_model_name:
            # DialoGPT works best with conversational context
            if len(conversation_context) > 1:
                # Build conversation history
                context_parts = []
                for msg in conversation_context[-4:]:  # Last 4 messages
                    if msg['role'] == 'user':
                        context_parts.append(msg['content'])
                    else:
                        context_parts.append(msg['content'])
                prompt = " ".join(context_parts[-3:]) + f" {user_input}"
            else:
                prompt = user_input
            
        else:
            # Generic format with some context
            recent_context = conversation_context[-2]['content'] if len(conversation_context) >= 2 else ""
            context_str = f"Tidigare: {recent_context}\n" if recent_context else ""
            prompt = f"{context_str}Svara på svenska: {user_input}\nSvar:"
        
        if should_stop:
            return "Konversationen avbröts."
        
        # Generate response with better settings
        response = chat_pipeline(
            prompt,
            max_new_tokens=40,     # Shorter for more natural conversation
            do_sample=True,
            temperature=0.7,       # Slightly more creative
            top_p=0.85,           # More focused
            repetition_penalty=1.1,
            pad_token_id=chat_pipeline.tokenizer.eos_token_id,
            eos_token_id=chat_pipeline.tokenizer.eos_token_id,
            return_full_text=False
        )
        
        if should_stop:
            return "Konversationen avbröts."
        
        # Extract and clean response
        bot_response = response[0]['generated_text'].strip()
        
        # Model-specific cleanup
        if "bellman" in current_model_name.lower():
            # Remove instruction markers
            bot_response = bot_response.replace("[/INST]", "").replace("[INST]", "")
            
        elif "gpt-sw3" in current_model_name.lower():
            # Remove common GPT-SW3 artifacts
            bot_response = bot_response.replace("Assistent:", "").replace("AI:", "")
            
        elif "DialoGPT" in current_model_name:
            # Clean up DialoGPT repetition
            sentences = bot_response.split('.')
            if len(sentences) > 1:
                bot_response = sentences[0] + '.'
        
        # General cleanup
        bot_response = bot_response.replace(user_input, "")  # Remove echo
        bot_response = bot_response.replace("Användare:", "")
        bot_response = bot_response.replace("Svar:", "")
        
        # Remove weird formatting and repetition
        lines = bot_response.split('\n')
        bot_response = lines[0].strip() if lines else bot_response.strip()
        
        # Final cleanup
        bot_response = bot_response.replace('\n', ' ').strip()
        
        # Ensure it's reasonable Swedish
        if len(bot_response) < 3:
            bot_response = "Ursäkta, kan du upprepa det?"
        elif len(bot_response) > 100:  # Shorter responses for better flow
            # Truncate at sentence boundary
            sentences = bot_response.split('.')
            bot_response = sentences[0] + '.' if sentences[0] else bot_response[:80] + '...'
        
        # Remove any remaining weird characters or formatting
        import re
        bot_response = re.sub(r'[^\w\såäöÅÄÖ.,!?-]', '', bot_response)
        bot_response = bot_response.strip()
        
        if not bot_response:
            bot_response = "Jag förstod inte riktigt. Kan du fråga något annat?"
        
        # Add AI response to context
        conversation_context.append({"role": "assistant", "content": bot_response})
        
        print(f"🤖 AI Response: '{bot_response}'")
        return bot_response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Ursäkta, jag hade tekniska problem. Försök igen."

def speak_text(text):
    """Convert text to speech and play it with better control"""
    global is_speaking, should_stop, current_audio_thread
    
    try:
        if not text or len(text.strip()) == 0 or should_stop:
            return
        
        with audio_lock:
            is_speaking = True
        
        print(f"🔊 Speaking: '{text}'")
        
        # Initialize pygame mixer with better settings
        try:
            pygame.mixer.quit()  # Clean shutdown if already running
        except:
            pass
        
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        # Create TTS audio with slower speed for clarity
        tts = gTTS(text=text, lang='sv', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            
            if should_stop:
                return
            
            # Play the audio with interruption capability
            pygame.mixer.music.load(tmp_file.name)
            pygame.mixer.music.play()
            
            # Wait for playback to complete or interruption
            while pygame.mixer.music.get_busy() and not should_stop:
                time.sleep(0.1)
            
            # Stop playback if interrupted
            if should_stop:
                pygame.mixer.music.stop()
            
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
        with audio_lock:
            is_speaking = False

def process_conversation_turn(audio):
    """Process one turn of conversation with better error handling and interruption support"""
    global current_audio_thread, should_stop
    
    try:
        if audio is None:
            return "", "Ingen ljudfil. Försök igen.", [], "🎤 Redo att lyssna"
        
        # Reset stop flag for new conversation turn
        should_stop = False
        
        # Step 1: Convert speech to text
        print("🎤 Processing audio...")
        user_text = transcribe_audio(audio)
        
        if should_stop:
            return "", "Konversationen avbröts.", [], "🎤 Redo att lyssna"
        
        if not user_text:
            return "", "Kunde inte höra svenska ord. Tala tydligare.", [], "🎤 Redo att lyssna"
        
        # Step 2: Get AI response
        print("🤖 Generating Swedish response...")
        bot_response = get_chatbot_response(user_text)
        
        if should_stop:
            return user_text, "Konversationen avbröts.", [], "🎤 Redo att lyssna"
        
        # Step 3: Speak the response (in background with better management)
        print("🔊 Starting speech...")
        
        # Stop any existing speech
        if current_audio_thread and current_audio_thread.is_alive():
            should_stop = True
            current_audio_thread.join(timeout=1.0)  # Wait briefly for cleanup
        
        # Reset stop flag and start new speech
        should_stop = False
        current_audio_thread = threading.Thread(target=speak_text, args=(bot_response,))
        current_audio_thread.daemon = True
        current_audio_thread.start()
        
        # Step 4: Update conversation history
        new_exchange = [user_text, bot_response]
        
        return user_text, bot_response, [new_exchange], "🎤 Lyssnar... (prata igen eller vänta på svar)"
        
    except Exception as e:
        error_msg = f"Fel: {str(e)[:50]}"
        print(f"❌ {error_msg}")
        return "", f"Tekniskt fel: {error_msg}", [], "🎤 Redo att lyssna"

def stop_conversation():
    """Stop current conversation and audio playback"""
    global should_stop, current_audio_thread, is_speaking, is_listening
    
    print("⏹️ Stopping conversation...")
    should_stop = True
    
    # Stop audio playback
    try:
        pygame.mixer.music.stop()
    except:
        pass
    
    # Wait for threads to finish
    if current_audio_thread and current_audio_thread.is_alive():
        current_audio_thread.join(timeout=1.0)
    
    # Reset states
    with audio_lock:
        is_speaking = False
        is_listening = False
    
    return "Konversationen stoppad. Klicka igen för att börja prata."

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
        
        # Control buttons
        with gr.Row():
            stop_button = gr.Button("⏹️ Stoppa konversation", variant="secondary", size="sm")
            reset_button = gr.Button("🔄 Rensa historik", variant="secondary", size="sm")
        
        # Microphone section
        gr.HTML("""
        <div class="mic-section">
            <h2>🎤 Tala svenska här!</h2>
            <p>Säg något kort och tydligt - vänta på svar innan du pratar igen</p>
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
        
        def handle_stop():
            """Handle stop button click"""
            stop_msg = stop_conversation()
            return "", "", stop_msg
        
        def handle_reset():
            """Handle reset button click"""
            global conversation_context
            conversation_context = []
            stop_conversation()
            return "", "", [], [], "🎤 Historik rensad - redo att börja på nytt"
        
        # Connect audio to processing
        audio_input.change(
            fn=handle_audio_input,
            inputs=[audio_input, chat_state],
            outputs=[last_user_text, last_ai_response, conversation_history, chat_state, status_display]
        )
        
        # Connect control buttons
        stop_button.click(
            fn=handle_stop,
            outputs=[last_user_text, last_ai_response, status_display]
        )
        
        reset_button.click(
            fn=handle_reset,
            outputs=[last_user_text, last_ai_response, conversation_history, chat_state, status_display]
        )
        
        # Usage tips
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #E8F5E8; border-radius: 10px; border-left: 4px solid #4CAF50;">
            <h3>💡 Tips för bästa resultat:</h3>
            <ul style="font-size: 1.1em; line-height: 1.6;">
                <li><strong>🗣️ Tala tydligt</strong> - inte för snabbt eller långsamt</li>
                <li><strong>📱 Kortare meningar</strong> - 5-10 ord fungerar bäst</li>
                <li><strong>⏳ Vänta på svar</strong> - låt AI:n svara färdigt innan du pratar igen</li>
                <li><strong>⏹️ Använd Stoppa</strong> - för att avbryta pågående konversation</li>
                <li><strong>🔄 Rensa historik</strong> - för att börja ett nytt ämne</li>
                <li><strong>🔄 Försök igen</strong> - om transkriptionen blev fel</li>
            </ul>
            <h4>🎯 Bra testfraser:</h4>
            <p><em>"Hej, vad heter du?" • "Hur mår du idag?" • "Vad gillar du?" • "Berätta om Sverige" • "Vad gör du?"</em></p>
            <h4>⚠️ Viktigt:</h4>
            <p><em>Vänta tills AI:n slutat prata innan du säger något nytt. Använd Stoppa-knappen om något går fel.</em></p>
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