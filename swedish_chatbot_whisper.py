#!/usr/bin/env python3
"""
Swedish Voice Chatbot using Whisper (instead of Vosk)
==================================================

A completely free, open-source Swedish voice chatbot that:
- Uses Whisper for speech-to-text (much more reliable than Vosk)
- Uses Swedish language models for AI responses  
- Uses gTTS for text-to-speech
- Runs completely offline (after initial downloads)

Requirements:
- transformers, torch, openai-whisper
- gtts, pygame (for audio)
- gradio (for web interface)
- HF token for Swedish models (optional)
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

# Global models - loaded once at startup
whisper_model = None
chat_pipeline = None
current_model_name = None

def initialize_models():
    """Initialize all AI models at startup"""
    global whisper_model, chat_pipeline, current_model_name
    
    print("🎤 Loading Whisper model for Swedish speech recognition...")
    try:
        # Load Whisper model - base is good balance of speed/accuracy
        whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load Whisper: {e}")
        return False
    
    print("\n🧠 Loading Swedish language model...")
    
    # Try Swedish models first, then fallback to general models
    model_options = [
        "AI-Sweden-Models/gpt-sw3-126m",             # Official Swedish model (requires token)
        "neph1/bellman-7b-mistral-instruct-v0.2",    # Open Swedish model
        "distilgpt2",                                 # General English model
        "microsoft/DialoGPT-small"                    # Conversational model
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
            
            # Test the model
            test_response = chat_pipeline("Hej, hur mår du?", max_length=50, num_return_sequences=1)
            print(f"🧪 Test response: {test_response[0]['generated_text'][:100]}...")
            
            break
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    if chat_pipeline is None:
        print("❌ Could not load any language model!")
        print("💡 Solutions:")
        print("1. Set HF_TOKEN for Swedish models: export HF_TOKEN='your_token'")
        print("2. Install missing dependencies: pip install transformers torch")
        print("3. Check internet connection for model downloads")
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
            return "Ingen ljudfil hittades."
        
        print(f"🎤 Transcribing audio: {audio_file}")
        
        # Load and transcribe audio
        result = whisper_model.transcribe(audio_file, language="sv")
        text = result["text"].strip()
        
        if text:
            print(f"🗣️ User said: {text}")
            return text
        else:
            return "Kunde inte höra vad du sa. Försök igen."
            
    except Exception as e:
        print(f"❌ Speech recognition error: {e}")
        return "Fel vid ljudigenkänning. Försök igen."

def get_chatbot_response(user_input):
    """Generate AI response using the loaded language model"""
    try:
        if not user_input.strip():
            return "Jag hörde inget. Kan du säga något?"
        
        # Create appropriate prompt based on model type
        if "DialoGPT" in current_model_name:
            # DialoGPT format
            prompt = user_input
        elif "bellman" in current_model_name.lower():
            # Bellman/Mistral format  
            prompt = f"[INST]{user_input}[/INST]"
        elif "gpt-sw3" in current_model_name.lower():
            # GPT-SW3 format
            prompt = f"Användare: {user_input}\nAssistent:"
        else:
            # General format
            prompt = f"Användare: {user_input}\nSvar:"
        
        print(f"🤖 Generating response for: {user_input}")
        
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
        if "DialoGPT" in current_model_name:
            # For DialoGPT, the response is after the input
            if len(generated) > len(user_input):
                bot_response = generated[len(user_input):].strip()
            else:
                bot_response = "Jag förstår inte riktigt."
        elif "bellman" in current_model_name.lower():
            # For Bellman, extract after [/INST]
            if "[/INST]" in generated:
                bot_response = generated.split("[/INST]")[-1].strip()
            else:
                bot_response = generated.replace(prompt, "").strip()
        else:
            # For other models, remove the prompt
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
        
        print(f"🤖 AI Response: {bot_response}")
        return bot_response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Ursäkta, jag hade problem att förstå. Försök igen."

def speak_text(text):
    """Convert text to speech and play it"""
    try:
        if not text or len(text.strip()) == 0:
            return
        
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

def process_voice_chat(audio):
    """Main function that processes voice input and returns AI response"""
    try:
        # Step 1: Convert speech to text
        user_text = transcribe_audio(audio)
        
        if not user_text or "fel vid" in user_text.lower() or "kunde inte" in user_text.lower():
            return user_text, "Kunde inte förstå ljudet. Försök tala tydligare."
        
        # Step 2: Get AI response
        bot_response = get_chatbot_response(user_text)
        
        # Step 3: Speak the response (in background thread to avoid blocking UI)
        speech_thread = threading.Thread(target=speak_text, args=(bot_response,))
        speech_thread.daemon = True
        speech_thread.start()
        
        return user_text, bot_response
        
    except Exception as e:
        error_msg = f"Fel i chatbot: {str(e)}"
        print(f"❌ {error_msg}")
        return "Fel vid bearbetning", error_msg

def create_gradio_interface():
    """Create the Gradio web interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
    }
    .header {
        text-align: center;
        background: linear-gradient(90deg, #0066cc, #004499);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .model-info {
        background: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    """
    
    with gr.Blocks(css=css, title="🇸🇪 Svensk Röstchatbot") as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🇸🇪 Svensk Röstchatbot (Whisper Edition)</h1>
            <p>Prata med mig på svenska! Använder Whisper för taligenkänning.</p>
        </div>
        """)
        
        # Model info
        gr.HTML(f"""
        <div class="model-info">
            <h3>🤖 Aktiva AI-modeller:</h3>
            <ul>
                <li><strong>🎤 Taligenkänning:</strong> OpenAI Whisper (base)</li>
                <li><strong>🧠 Språkmodell:</strong> {current_model_name or 'Laddas...'}</li>
                <li><strong>🔊 Talsyntes:</strong> Google Text-to-Speech</li>
            </ul>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Audio input
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Klicka för att spela in din röst",
                    elem_id="audio_input"
                )
                
                # Process button
                process_btn = gr.Button(
                    "💬 Bearbeta röstmeddelande", 
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Text outputs
                user_text_output = gr.Textbox(
                    label="🗣️ Vad du sa:",
                    interactive=False,
                    lines=2
                )
                
                bot_response_output = gr.Textbox(
                    label="🤖 AI-svar:",
                    interactive=False,
                    lines=4
                )
        
        # Conversation history
        with gr.Row():
            conversation_history = gr.Chatbot(
                label="💬 Konversationshistorik",
                height=300
            )
        
        # State for conversation
        chat_history = gr.State([])
        
        def update_conversation(audio, history):
            """Process audio and update conversation history"""
            user_text, bot_response = process_voice_chat(audio)
            
            # Add to history
            if user_text and bot_response:
                history.append([user_text, bot_response])
            
            return user_text, bot_response, history, history
        
        # Connect the interface
        process_btn.click(
            fn=update_conversation,
            inputs=[audio_input, chat_history],
            outputs=[user_text_output, bot_response_output, conversation_history, chat_history]
        )
        
        # Auto-process when audio is recorded
        audio_input.change(
            fn=update_conversation,
            inputs=[audio_input, chat_history],
            outputs=[user_text_output, bot_response_output, conversation_history, chat_history]
        )
        
        # Usage instructions
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 8px;">
            <h3>📋 Hur man använder:</h3>
            <ol>
                <li><strong>Klicka på mikrofonen</strong> och spela in ditt meddelande på svenska</li>
                <li><strong>Vänta</strong> medan AI:n bearbetar din röst</li>
                <li><strong>Lyssna</strong> på AI:ns svar (spelas automatiskt)</li>
                <li><strong>Fortsätt konversationen</strong> genom att spela in mer!</li>
            </ol>
            <p><em>💡 Tips: Tala tydligt och vänta på att inspelningen slutar innan du börjar prata.</em></p>
        </div>
        """)
    
    return demo

def main():
    """Main function to run the chatbot"""
    print("🇸🇪 SVENSK RÖSTCHATBOT MED WHISPER")
    print("=" * 50)
    
    # Initialize models
    if not initialize_models():
        print("❌ Failed to initialize models. Exiting.")
        return
    
    print("\n🚀 Starting web interface...")
    print("📱 The chatbot will open in your browser automatically.")
    print("🎤 Click the microphone to start talking!")
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    try:
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,  # Set to True for public sharing
            inbrowser=True,  # Open in browser automatically
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped. Hej då!")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")

if __name__ == "__main__":
    main()