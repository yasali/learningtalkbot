# 🚀 Swedish Voice Chatbot - Quick Start Guide

Get your Swedish voice chatbot running in 5 minutes! Follow these exact steps.

## ⚡ Super Quick Start (Recommended)

```bash
# 1. Clone or download the project
git clone <your-repo-url>
cd swedish-voice-chatbot

# 2. Download all models automatically
python download_models.py

# 3. Run the chatbot
python swedish_chatbot.py

# 4. Open browser to http://localhost:7860 and start talking!
```

That's it! Your Swedish voice chatbot is ready! 🎉

## 📋 Step-by-Step Instructions

### Step 1: Environment Setup
```bash
# Create virtual environment (recommended)
python3 -m venv chatbot_env
source chatbot_env/bin/activate  # Linux/Mac
# OR: chatbot_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Models

#### Option A: Automatic Download (Easiest)
```bash
python download_models.py
```

#### Option B: Manual Download
```bash
# Download Vosk Swedish model
wget https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
unzip vosk-model-sv-rhasspy-0.15.zip
rm vosk-model-sv-rhasspy-0.15.zip

# Download GPT-SW3 model (run in Python)
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('AI-Sweden-Models/gpt-sw3-126m'); AutoModelForCausalLM.from_pretrained('AI-Sweden-Models/gpt-sw3-126m')"
```

### Step 3: Launch Chatbot
```bash
python swedish_chatbot.py
```

### Step 4: Use the Chatbot
1. Open browser to: `http://localhost:7860`
2. Click the **🎤 Prata** button
3. Wait for "Lyssnar..." message
4. Speak clearly in Swedish
5. Wait for the AI response (text + audio)
6. Continue the conversation!

## 🎯 What Gets Downloaded

### Vosk Swedish Model
- **File**: `vosk-model-sv-rhasspy-0.15.zip` (~45MB)
- **Source**: https://alphacephei.com/vosk/models/
- **Purpose**: Converts Swedish speech to text
- **Location**: `./vosk-model-sv-rhasspy-0.15/`
- **Works**: Completely offline

### GPT-SW3 Swedish Model
- **Model**: `AI-Sweden-Models/gpt-sw3-126m` (~500MB)
- **Source**: https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m
- **Purpose**: Generates Swedish text responses
- **Location**: `~/.cache/huggingface/transformers/`
- **Works**: Locally after download

## 💡 Usage Examples

### Conversation Examples:
```
You: "Hej, vad heter du?"
Bot: "Hej! Jag är en svensk röstassistent. Vad kan jag hjälpa dig med?"

You: "Kan du berätta om Stockholm?"
Bot: "Stockholm är Sveriges huvudstad och ligger på fjorton öar mellan Östersjön och Mälaren..."

You: "Vad är klockan?"
Bot: "Jag kan inte se klockan just nu, men du kan kolla tiden på din dator eller telefon."
```

## 🛠 Troubleshooting

### Download Issues:
```bash
# If automatic download fails, try manual:
# 1. Check internet connection
# 2. Use alternative Vosk download:
curl -L -o vosk-model.zip https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip

# 3. For GPT-SW3, clear cache and retry:
rm -rf ~/.cache/huggingface/
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m')"
```

### Audio Issues:
```bash
# Linux: Install audio dependencies
sudo apt install portaudio19-dev alsa-utils

# Test microphone
arecord -l

# Test speakers
speaker-test -t wav -c 2
```

### Memory Issues:
```bash
# Use smaller model for low-memory systems
# Edit swedish_chatbot.py and change:
model_name = "AI-Sweden-Models/gpt-sw3-126m"  # Smallest version
```

## 📁 Project Structure After Setup

```
swedish-voice-chatbot/
├── swedish_chatbot.py          # Main application
├── download_models.py          # Model download script
├── requirements.txt            # Dependencies
├── setup.py                   # Auto setup
├── run_chatbot.sh            # Launch script
├── README.md                 # Documentation
├── MODEL_SETUP.md           # Detailed model guide
├── QUICKSTART.md            # This file
├── chatbot_env/             # Virtual environment
└── vosk-model-sv-rhasspy-0.15/  # Vosk model (after download)
    ├── am/
    ├── graph/
    ├── ivector/
    └── conf/
```

## 🔧 Alternative Launch Methods

### Method 1: Direct Python
```bash
python swedish_chatbot.py
```

### Method 2: Shell Script (Linux/Mac)
```bash
chmod +x run_chatbot.sh
./run_chatbot.sh
```

### Method 3: Setup Script
```bash
python setup.py  # Downloads everything and launches
```

## 📊 System Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- 1GB disk space
- Microphone
- Internet (for initial download)

### Recommended:
- Python 3.9+
- 8GB RAM
- GPU with CUDA
- Good quality microphone
- Fast internet (for downloads)

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Download script completes without errors
- ✅ `python swedish_chatbot.py` starts successfully
- ✅ Browser opens to chatbot interface
- ✅ Clicking 🎤 shows "Lyssnar..." message
- ✅ Speaking Swedish produces text transcription
- ✅ Bot responds with Swedish text and audio

## 🚨 Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| "No module named 'vosk'" | `pip install vosk` |
| "No module named 'transformers'" | `pip install transformers torch` |
| Vosk model not found | Run `python download_models.py` |
| GPT-SW3 download fails | Check internet, clear `~/.cache/huggingface/` |
| No audio output | Install `pygame`: `pip install pygame` |
| Microphone not working | Check permissions, test with other apps |
| Out of memory | Use smaller model: `AI-Sweden/gpt-sw3-126m` |

## 🎯 Next Steps

Once running:
1. **Test basic conversation** - Try simple Swedish phrases
2. **Adjust settings** - Edit `swedish_chatbot.py` for custom behavior
3. **Improve audio** - Use a better microphone for accuracy
4. **Upgrade model** - Try larger GPT-SW3 versions for better responses
5. **Customize UI** - Modify the Gradio interface

---

**🎊 Congratulations!** You now have a fully functional Swedish voice chatbot running locally on your machine! 🇸🇪🤖

**Need help?** Check `MODEL_SETUP.md` for detailed model information or `README.md` for comprehensive documentation.