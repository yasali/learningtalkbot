# 🇸🇪 Swedish Voice Chatbot

A completely free, open-source Swedish voice chatbot using Whisper, Swedish language models, and text-to-speech.

## ✨ Features

- 🎤 **Swedish Speech Recognition** - Whisper with Swedish optimization
- 🧠 **Swedish AI Responses** - Bellman model for natural conversation  
- 🔊 **Swedish Text-to-Speech** - Google TTS
- 💬 **Natural Conversation** - Optimized for back-and-forth chat
- 🌐 **Web Interface** - Easy-to-use Gradio interface
- 🔒 **Completely Free** - No API keys required for basic functionality

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd swedish-chatbot
```

### 2. Set Up Environment
```bash
# Automatic setup (recommended)
./setup_environment.sh

# Or manual setup:
python3 -m venv chatbot_env
source chatbot_env/bin/activate
pip install -r requirements.txt
```

### 3. Get Hugging Face Token (Optional)
For best Swedish models, get a free token from [Hugging Face](https://huggingface.co/settings/tokens):
```bash
export HF_TOKEN='your_token_here'
```

### 4. Run the Chatbot
```bash
# Easy way:
./start_fixed_chatbot.sh

# Or manual:
source chatbot_env/bin/activate
export HF_TOKEN='your_token_here'  # Optional
python swedish_chatbot_fixed.py
```

### 5. Use the Chatbot
- Open **http://localhost:7861** in your browser
- Click the microphone and speak Swedish
- Listen to the AI's Swedish response
- Continue the conversation naturally!

## 📋 System Requirements

### macOS (Homebrew):
```bash
brew install python@3.11 portaudio ffmpeg
```

### Linux (Ubuntu/Debian):
```bash
sudo apt install python3-venv python3-pip portaudio19-dev ffmpeg mpg123 alsa-utils
```

### Windows:
- Install Python 3.11+
- Install ffmpeg
- The setup script will handle the rest

## 🎯 Test Phrases

Try these Swedish phrases to test the chatbot:
- "Hej, vad heter du?"
- "Hur mår du idag?"
- "Berätta om Sverige"
- "Vad gillar du?"
- "Kan du hjälpa mig?"

## 📁 Repository Structure

```
swedish-chatbot/
├── swedish_chatbot_fixed.py      # Main chatbot (recommended)
├── swedish_chatbot_whisper.py    # Alternative version
├── requirements.txt              # Python dependencies
├── setup_environment.sh          # Environment setup script
├── start_fixed_chatbot.sh        # Quick start script
├── setup_token.py                # HF token setup helper
├── .gitignore                    # Excludes virtual env and models
├── README.md                     # This file
├── FIXED_VERSION_GUIDE.md        # Detailed fix explanations
└── docs/                         # Additional documentation
    ├── MODEL_SETUP.md
    ├── QUICKSTART.md
    └── FIXES_SUMMARY.md
```

## ⚠️ Important Notes

### Virtual Environment
- **The `chatbot_env/` folder is NOT included in git** (it's in `.gitignore`)
- **Never commit the virtual environment** - it's huge (several GB)
- **Each user creates their own environment** using `setup_environment.sh`

### AI Models
- **Models are downloaded automatically** on first run
- **Models are cached locally** (~2-7GB depending on which models load)
- **Model cache is NOT committed** to the repository

### Dependencies
- **Only `requirements.txt` is committed** - lists what to install
- **Virtual environment is created fresh** for each installation
- **This keeps the repository small** and portable

## 🔧 Troubleshooting

### Microphone Issues (macOS):
- Grant microphone permissions in System Preferences
- Allow your browser to access microphone

### Model Download Issues:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python setup_token.py  # Set up HF token
python swedish_chatbot_fixed.py
```

### Audio Issues:
```bash
# Install audio dependencies
brew install portaudio ffmpeg  # macOS
sudo apt install portaudio19-dev ffmpeg  # Linux
```

## 🌟 Features

### Speech Recognition
- Uses OpenAI Whisper "small" model
- Optimized Swedish language settings
- Better accuracy than basic Whisper

### Language Models (Priority Order)
1. **Bellman** (`neph1/bellman-7b-mistral-instruct-v0.2`) - Best Swedish conversation
2. **GPT-SW3** (`AI-Sweden-Models/gpt-sw3-126m`) - Official Swedish model (requires HF token)
3. **DialoGPT** (`microsoft/DialoGPT-small`) - Good fallback
4. **DistilGPT2** - Basic fallback

### Response Quality
- Short, natural responses (40 tokens max)
- Artifact filtering (removes "Datum:", "Kubernetes", etc.)
- Model-specific prompt optimization
- Context-aware conversation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. **DON'T commit the virtual environment** (it's already in `.gitignore`)
4. Only commit code, documentation, and requirements
5. Submit a pull request

## 📝 License

MIT License - feel free to use for personal or commercial projects.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- AI Sweden for GPT-SW3 models
- Bellman project for excellent Swedish conversation AI
- Gradio for the web interface
- Google for Text-to-Speech
