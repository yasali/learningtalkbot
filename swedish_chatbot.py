import gradio as gr
from gtts import gTTS
import os
import time
import queue
import sounddevice as sd
import vosk
import json
import tempfile
import threading
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- Global Configurations ---
print("Initializing Swedish Voice Chatbot...")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Vosk Speech-to-Text configuration
MODEL_DIR = "vosk-model-small-sv-0.15"
SAMPLERATE = 16000
AUDIO_DEVICE = None  # Use default device
audio_queue = queue.Queue()

# Global variables for models
vosk_model = None
chatbot_pipeline = None
is_listening = False

def download_vosk_model():
    """Download Vosk Swedish model if not present"""
    import urllib.request
    import zipfile
    
    if not os.path.exists(MODEL_DIR):
        print("Downloading Vosk Swedish model (this may take a few minutes)...")
        url = "https://alphacephei.com/vosk/models/vosk-model-small-sv-0.15.zip"
        
        try:
            urllib.request.urlretrieve(url, "vosk-model.zip")
            print("Extracting model...")
            with zipfile.ZipFile("vosk-model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("vosk-model.zip")
            print("Vosk model downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading Vosk model: {e}")
            print(f"Please download manually from: {url}")
            print(f"Extract to: {MODEL_DIR}")
            raise
    else:
        print("Vosk model already exists.")

def initialize_models():
    """Initialize all AI models"""
    global vosk_model, chatbot_pipeline
    
    # Initialize Vosk model
    try:
        download_vosk_model()
        vosk_model = vosk.Model(MODEL_DIR)
        print("Vosk model loaded successfully!")
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        raise
    
    # Initialize language model
    try:
        print("Loading language model (this may take several minutes on first run)...")
        
        # Try Swedish models first, then fallback to general models
        model_options = [
            "AI-Sweden/gpt-sw3-126m",      # Swedish model (requires token)
            "distilgpt2",                  # General English model
            "microsoft/DialoGPT-small"     # Conversational model
        ]
        
        # Check for Hugging Face token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        
        chatbot_pipeline = None
        for model_name in model_options:
            try:
                print(f"Trying model: {model_name}")
                
                # Load tokenizer and model
                if hf_token and "AI-Sweden" in model_name:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Set pad token if not available
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Create pipeline
                chatbot_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device == "cuda" else -1,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                print(f"✅ Model {model_name} loaded successfully!")
                break
                
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {str(e)[:100]}...")
                continue
        
        if chatbot_pipeline is None:
            raise Exception("No language model could be loaded")
            
    except Exception as e:
        print(f"Error loading language model: {e}")
        print("\n💡 SOLUTIONS:")
        print("1. For Swedish models: Set HF_TOKEN environment variable")
        print("2. Install required packages: pip install transformers torch")
        print("3. Check internet connection for model downloads")
        raise

def get_text_from_mic():
    """Records audio from microphone and converts to text using Vosk"""
    global is_listening
    
    if vosk_model is None:
        return "Error: Vosk model not loaded"
    
    def callback(indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        audio_queue.put(bytes(indata))

    try:
        is_listening = True
        print("🎤 Lyssnar... Prata nu!")
        
        with sd.RawInputStream(
            samplerate=SAMPLERATE, 
            blocksize=8000, 
            device=AUDIO_DEVICE,
            dtype='int16', 
            channels=1, 
            callback=callback
        ):
            rec = vosk.KaldiRecognizer(vosk_model, SAMPLERATE)
            timeout = time.time() + 10  # 10 second timeout
            
            while is_listening and time.time() < timeout:
                try:
                    data = audio_queue.get(timeout=0.1)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get('text', '').strip()
                        if text:
                            print(f"Du sa: {text}")
                            is_listening = False
                            return text
                except queue.Empty:
                    continue
            
            # Get partial result if no complete sentence
            partial_result = json.loads(rec.FinalResult())
            text = partial_result.get('text', '').strip()
            if text:
                print(f"Du sa: {text}")
                return text
            else:
                return "Inget tal upptäckt"
                
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return f"Fel vid taligenkänning: {e}"
    finally:
        is_listening = False

def get_chatbot_response(user_input):
    """Generate response using local GPT-SW3 model"""
    if chatbot_pipeline is None:
        return "Error: Chatbot model not loaded"
    
    if not user_input or user_input.strip() == "":
        return "Jag hörde inget. Kan du upprepa?"
    
    try:
        # Create a conversational prompt in Swedish
        prompt = f"Användare: {user_input}\nAssistent: "
        
        # Generate response
        response = chatbot_pipeline(
            prompt,
            max_length=len(prompt.split()) + 50,  # Reasonable length
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=chatbot_pipeline.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        
        # Clean up the response
        if "Assistent:" in generated_text:
            bot_response = generated_text.split("Assistent:")[-1].strip()
        else:
            bot_response = generated_text[len(prompt):].strip()
        
        # Remove any unwanted tokens or repetitions
        bot_response = bot_response.split('\n')[0].strip()
        
        if not bot_response:
            bot_response = "Jag förstår inte riktigt. Kan du förklara mer?"
        
        return bot_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Ursäkta, jag hade problem att generera ett svar."

def speak_response(text):
    """Convert text to speech and play audio"""
    if not text or text.strip() == "":
        return
    
    try:
        print(f"🔊 Chatbot svarar: {text}")
        
        # Create TTS
        tts = gTTS(text=text, lang='sv', slow=False)
        
        # Use temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            
            # Play audio (try different methods for compatibility)
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except ImportError:
                try:
                    os.system(f"mpg123 {tmp_file.name}")
                except:
                    try:
                        os.system(f"ffplay -nodisp -autoexit {tmp_file.name}")
                    except:
                        print("Could not play audio. Install pygame, mpg123, or ffmpeg for audio playback.")
            
            # Clean up
            try:
                os.unlink(tmp_file.name)
            except:
                pass
                
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# --- Gradio UI ---
def chat_with_voice(history):
    """Main function for voice chat interaction"""
    global is_listening
    
    try:
        # 1. Listen for speech
        user_text = get_text_from_mic()
        
        if user_text and user_text != "Inget tal upptäckt" and not user_text.startswith("Fel"):
            # 2. Get chatbot response
            bot_response = get_chatbot_response(user_text)
            
            # 3. Speak the response in a separate thread
            speech_thread = threading.Thread(target=speak_response, args=(bot_response,))
            speech_thread.daemon = True
            speech_thread.start()
            
            # 4. Update chat history
            history.append((user_text, bot_response))
        else:
            # Handle errors or no speech detected
            history.append(("", user_text))
    
    except Exception as e:
        error_msg = f"Fel: {str(e)}"
        print(error_msg)
        history.append(("", error_msg))
    
    return history, ""

def stop_listening():
    """Stop current listening session"""
    global is_listening
    is_listening = False
    return "Stoppade lyssnandet"

# Initialize models when script starts
print("Starting model initialization...")
initialize_models()
print("All models loaded successfully!")

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Svensk Röst-Chatbot",
    css="""
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    """
) as demo:
    
    gr.Markdown("""
    # 🇸🇪 Svensk Röst-Chatbot
    
    En helt gratis chatbot som förstår och talar svenska! Använder:
    - **Vosk** för tal-till-text (offline)
    - **GPT-SW3** för svenska språkförståelse (offline)
    - **gTTS** för text-till-tal
    
    Tryck på mikrofon-knappen och börja prata!
    """)
    
    chatbot = gr.Chatbot(
        label="Konversation",
        height=400,
        show_label=True
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            msg = gr.Textbox(
                placeholder="Tryck på mikrofonknappen och prata på svenska...",
                interactive=False,
                label="Status"
            )
        with gr.Column(scale=1):
            mic_button = gr.Button("🎤 Prata", variant="primary", size="lg")
        with gr.Column(scale=1):
            stop_button = gr.Button("⏹️ Stopp", variant="secondary", size="lg")
    
    gr.Markdown("""
    ### Instruktioner:
    1. **Tryck på 🎤 Prata** och vänta på "Lyssnar..."
    2. **Tala tydligt på svenska** (du har 10 sekunder)
    3. **Vänta** på chatbotens svar
    4. **Upprepa** för fortsatt konversation
    
    ### Tips:
    - Tala tydligt och inte för snabbt
    - Vänta tills "Lyssnar..." visas innan du pratar
    - Använd **Stopp** för att avbryta inspelning
    """)
    
    def start_conversation():
        return gr.update(value="🎤 Förbereder för inspelning...")
    
    # Event handlers
    mic_button.click(
        fn=start_conversation,
        outputs=msg
    ).then(
        fn=chat_with_voice,
        inputs=chatbot,
        outputs=[chatbot, msg]
    )
    
    stop_button.click(
        fn=stop_listening,
        outputs=msg
    )

if __name__ == "__main__":
    print("\n🚀 Startar svensk röst-chatbot!")
    print("Öppna webbläsaren på adressen som visas nedan...")
    demo.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )