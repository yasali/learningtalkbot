#!/usr/bin/env python3
"""
Swedish Voice Chatbot - Conversational UI
=========================================

Natural conversation flow:
1. You talk → AI listens and responds
2. Wait for AI to finish speaking
3. Talk again → AI responds again
4. Repeat naturally

Optimized for real conversations in Swedish!
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
        whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return False
    
    print("\n🧠 Loading Swedish language model...")
    
    # Try Swedish models first, then fallback to general models
    model_options = [
        "AI-Sweden-Models/gpt-sw3-126m",             
        "neph1/bellman-7b-mistral-instruct-v0.2",    
        "distilgpt2",                                 
        "microsoft/DialoGPT-small"                    
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
            
            # Create pipeline
            chat_pipeline = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            current_model_name = model_name
            print(f"✅ Successfully loaded: {model_name}")
            break
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    if chat_pipeline is None:
        print("❌ Could not load any language model!")
        return False
    
    print(f"\n🎉 All models loaded successfully!")
    print(f"📱 Speech Recognition: Whisper (base)")
    print(f"🧠 Language Model: {current_model_name}")
    print(f"🔊 Text-to-Speech: Google TTS")
    
    return True

def transcribe_audio(audio_file):
    """Convert audio to text using Whisper"""
    try:
        if audio_file is None:
            return ""
        
        print(f"🎤 Transcribing audio...")
        result = whisper_model.transcribe(audio_file, language="sv")
        text = result["text"].strip()
        
        if text:
            print(f"🗣️ User said: {text}")
            return text
        else:
            return ""
            
    except Exception as e:
        print(f"❌ Speech recognition error: {e}")
        return ""

def get_chatbot_response(user_input):
    """Generate AI response using the loaded language model"""
    global conversation_context
    
    try:
        if not user_input.strip():
            return "Jag hörde inget. Kan du säga något?"
        
        # Add to conversation context
        conversation_context.append(f"Användare: {user_input}")
        
        # Keep only last 6 exchanges (3 back-and-forth)
        if len(conversation_context) > 6:
            conversation_context = conversation_context[-6:]
        
        # Create context-aware prompt
        if "bellman" in current_model_name.lower():
            context = "\n".join(conversation_context[-4:])  # Last 2 exchanges
            prompt = f"[INST]{context}\nAnvändare: {user_input}[/INST]"
        elif "gpt-sw3" in current_model_name.lower():
            context = "\n".join(conversation_context[-4:])
            prompt = f"{context}\nAssistent:"
        else:
            prompt = user_input
        
        print(f"🤖 Generating response...")
        
        # Generate response
        response = chat_pipeline(
            prompt, 
            max_length=len(prompt.split()) + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=chat_pipeline.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated = response[0]['generated_text']
        
        # Clean up response based on model type
        if "bellman" in current_model_name.lower():
            if "[/INST]" in generated:
                bot_response = generated.split("[/INST]")[-1].strip()
            else:
                bot_response = generated.replace(prompt, "").strip()
        else:
            bot_response = generated.replace(prompt, "").strip()
            
            # Clean up common artifacts
            for prefix in ["Svar:", "Assistent:", "AI:", "Bot:"]:
                if bot_response.startswith(prefix):
                    bot_response = bot_response[len(prefix):].strip()
        
        # Ensure reasonable length
        if len(bot_response) > 200:
            sentences = bot_response.split('.')
            bot_response = sentences[0] + '.' if sentences else bot_response[:200] + '...'
        
        if not bot_response or len(bot_response) < 5:
            bot_response = "Kan du upprepa det på ett annat sätt?"
        
        # Add AI response to context
        conversation_context.append(f"Assistent: {bot_response}")
        
        print(f"🤖 AI Response: {bot_response}")
        return bot_response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Ursäkta, jag hade problem att förstå. Försök igen."

def speak_text(text):
    """Convert text to speech and play it"""
    global is_speaking
    
    try:
        if not text or len(text.strip()) == 0:
            return
        
        is_speaking = True
        print(f"🔊 Speaking: {text}")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Create TTS audio
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
    """Process one turn of conversation"""
    try:
        if audio is None:
            return "", "Ingen ljudfil hittades. Försök igen.", [], "🎤 Redo att lyssna"
        
        # Update status
        status = "🎤 Lyssnar på din röst..."
        
        # Step 1: Convert speech to text
        user_text = transcribe_audio(audio)
        
        if not user_text:
            return "", "Kunde inte höra vad du sa. Försök igen.", [], "🎤 Redo att lyssna"
        
        # Step 2: Get AI response
        status = "🤖 Tänker på ett svar..."
        bot_response = get_chatbot_response(user_text)
        
        # Step 3: Update conversation history
        new_exchange = [user_text, bot_response]
        
        # Step 4: Speak the response (in background)
        status = "🔊 AI pratar..."
        speech_thread = threading.Thread(target=speak_text, args=(bot_response,))
        speech_thread.daemon = True
        speech_thread.start()
        
        # Wait a moment for speech to start
        time.sleep(0.5)
        status = "🎤 Redo att lyssna"
        
        return user_text, bot_response, [new_exchange], status
        
    except Exception as e:
        error_msg = f"Fel: {str(e)}"
        print(f"❌ {error_msg}")
        return "", error_msg, [], "❌ Fel - försök igen"

def create_conversational_interface():
    """Create a conversational UI optimized for natural talking"""
    
    # Custom CSS for conversational UI
    css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .conversation-area {
        background: #f8fafc;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 2px solid #e2e8f0;
    }
    .mic-section {
        text-align: center;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
    }
    .status-display {
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: #e0f2fe;
        border: 2px solid #0277bd;
    }
    """
    
    with gr.Blocks(css=css, title="🇸🇪 Svenska Samtalet") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🇸🇪 Svenska Samtalet</h1>
            <h3>Naturlig konversation på svenska</h3>
            <p>Prata → Lyssna → Prata → Lyssna</p>
        </div>
        """)
        
        # Status display
        status_display = gr.Textbox(
            value="🎤 Redo att lyssna",
            label="Status",
            interactive=False,
            elem_classes=["status-display"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Large microphone section
                gr.HTML("""
                <div class="mic-section">
                    <h2>🎤 Tryck och prata</h2>
                    <p>Säg något på svenska!</p>
                </div>
                """)
                
                # Audio input - larger and more prominent
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="",
                    elem_id="main_microphone",
                    show_label=False
                )
        
        # Conversation display
        gr.HTML("""
        <div class="conversation-area">
            <h3>💬 Senaste utbyte:</h3>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                last_user_text = gr.Textbox(
                    label="🗣️ Du sa:",
                    interactive=False,
                    lines=2,
                    placeholder="Din svenska kommer att visas här..."
                )
            
            with gr.Column(scale=1):
                last_ai_response = gr.Textbox(
                    label="🤖 AI svarade:",
                    interactive=False,
                    lines=2,
                    placeholder="AI:ns svenska svar kommer att visas här..."
                )
        
        # Full conversation history
        conversation_history = gr.Chatbot(
            label="💬 Fullständig konversation",
            height=400,
            type="messages",
            placeholder="Din konversation på svenska kommer att visas här...\n\n🎤 Börja prata för att starta!"
        )
        
        # Hidden state for conversation
        chat_state = gr.State([])
        
        # Auto-process when audio is recorded
        def handle_audio_input(audio, history):
            user_text, ai_response, new_exchange, status = process_conversation_turn(audio)
            
            # Update full history
            if new_exchange and new_exchange[0]:
                history.extend([
                    {"role": "user", "content": new_exchange[0][0]},
                    {"role": "assistant", "content": new_exchange[0][1]}
                ])
            
            return user_text, ai_response, history, history, status
        
        # Connect audio input to processing
        audio_input.change(
            fn=handle_audio_input,
            inputs=[audio_input, chat_state],
            outputs=[last_user_text, last_ai_response, conversation_history, chat_state, status_display]
        )
        
        # Instructions
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #f0f9ff; border-radius: 10px; border-left: 4px solid #0ea5e9;">
            <h3>🎯 Så här fungerar det:</h3>
            <ol style="font-size: 1.1em; line-height: 1.6;">
                <li><strong>🎤 Prata:</strong> Klicka på mikrofonen och säg något på svenska</li>
                <li><strong>⏳ Vänta:</strong> AI:n lyssnar, tänker och svarar på svenska</li>
                <li><strong>🔊 Lyssna:</strong> Hör AI:ns svar (spelas automatiskt)</li>
                <li><strong>🔄 Upprepa:</strong> Prata igen för att fortsätta samtalet!</li>
            </ol>
            <p><em>💡 Tips: Tala tydligt och vänta tills AI:n slutat prata innan du säger något nytt.</em></p>
        </div>
        """)
    
    return demo

def main():
    """Main function to run the conversational chatbot"""
    print("🇸🇪 SVENSKA SAMTALET - NATURLIG KONVERSATION")
    print("=" * 60)
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    print("\n🚀 Starting conversational interface...")
    print("📱 The chatbot will open in your browser automatically.")
    print("🎤 Just click the microphone and start talking!")
    print("🔄 Natural conversation flow: Talk → Listen → Talk → Listen")
    
    # Create and launch interface
    demo = create_conversational_interface()
    
    try:
        print("🌐 Starting local server...")
        print("📱 Open in your browser: http://localhost:7860")
        print("🎤 Click microphone button to start conversation")
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            quiet=False
        )
    except KeyboardInterrupt:
        print("\n👋 Conversation ended. Hej då!")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")

if __name__ == "__main__":
    main()