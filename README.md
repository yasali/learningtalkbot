# 🇸🇪 Svensk Röst-Chatbot (Swedish Voice Chatbot)

En helt gratis röst-aktiverad chatbot som förstår och talar svenska! Använder endast open-source modeller som körs lokalt på din dator.

*A completely free voice-activated chatbot that understands and speaks Swedish! Uses only open-source models running locally on your computer.*

## ✨ Funktioner (Features)

- 🎤 **Tal-till-text**: Offline taligenkänning med Vosk
- 🧠 **Svensk AI**: GPT-SW3 modell för svenskt språkförståelse
- 🔊 **Text-till-tal**: Google Text-to-Speech för naturligt ljud
- 🌐 **Webb-gränssnitt**: Enkelt att använda med Gradio
- 💻 **Helt lokalt**: Ingen data skickas till externa servrar (förutom TTS)
- 🆓 **Gratis**: Använder endast open-source komponenter

## 🚀 Snabbstart (Quick Start)

### 1. Automatisk Installation

```bash
# Klona projektet
git clone <repository-url>
cd swedish-voice-chatbot

# Kör setup-scriptet
python3 setup.py
```

### 2. Manuell Installation

```bash
# Skapa virtuell miljö
python3 -m venv chatbot_env
source chatbot_env/bin/activate  # Linux/Mac
# eller
chatbot_env\Scripts\activate     # Windows

# Installera dependencies
pip install -r requirements.txt

# Starta chatboten
python swedish_chatbot.py
```

### 3. Öppna i Webbläsare

Gå till `http://localhost:7860` och börja prata svenska!

## 📋 Systemkrav (System Requirements)

### Minimum:
- Python 3.8+
- 4GB RAM
- Mikrofon
- Internetanslutning (för text-till-tal)

### Rekommenderat:
- Python 3.9+
- 8GB+ RAM
- GPU med CUDA (för snabbare AI-inferens)
- Bra mikrofon för bättre taligenkänning

## 🛠️ Installation per Operativsystem

### Linux (Ubuntu/Debian)
```bash
# Installera systemberoenden
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg mpg123 alsa-utils

# Följ sedan snabbstart-instruktionerna
```

### macOS
```bash
# Installera Homebrew om det inte finns
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installera beroenden
brew install portaudio ffmpeg mpg123

# Följ sedan snabbstart-instruktionerna
```

### Windows
```bash
# Installera Python från python.org
# Följ snabbstart-instruktionerna
# Eventuellt behöver du installera audio-codecs manuellt
```

## 🎯 Användning (Usage)

1. **Starta chatboten**: `python swedish_chatbot.py`
2. **Öppna webbläsaren** på den visade URL:en
3. **Klicka på 🎤 Prata** knappen
4. **Vänta** på "Lyssnar..." meddelandet
5. **Tala tydligt på svenska** (10 sekunder timeout)
6. **Vänta** på chatbotens svar
7. **Upprepa** för fortsatt konversation

### Tips för Bästa Resultat:
- Tala tydligt och inte för snabbt
- Använd vardagssvenska
- Vänta tills "Lyssnar..." visas
- Håll bakgrundsljud till minimum
- Testa mikrofonnivån innan du börjar

## 🔧 Konfiguration

### Ändra AI-modell:
Redigera `swedish_chatbot.py` och ändra `model_name`:
```python
# För snabbare men mindre kvalitet:
model_name = "AI-Sweden/gpt-sw3-126m"

# För bättre kvalitet men långsammare:
model_name = "AI-Sweden/gpt-sw3-356m"
```

### Ändra Vosk-modell:
Ladda ner en annan modell från [Vosk Models](https://alphacephei.com/vosk/models) och uppdatera `MODEL_DIR`.

### Ändra TTS-språk:
```python
tts = gTTS(text=text, lang='sv', slow=False)  # 'sv' för svenska
```

## 🐛 Felsökning (Troubleshooting)

### Problem med Ljud
```bash
# Linux: Testa mikrofon
arecord -l
aplay -l

# Kontrollera volymnivåer
alsamixer
```

### Import-fel
```bash
# Reinstallera dependencies
pip install --upgrade -r requirements.txt

# Eller installera individuellt:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Slow Model Loading
```bash
# Använd mindre modell eller lägg till GPU-support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Permission Errors (Linux)
```bash
# Lägg till användare i audio-gruppen
sudo usermod -a -G audio $USER
# Logga ut och in igen
```

## 📂 Projektstruktur

```
swedish-voice-chatbot/
├── swedish_chatbot.py      # Huvudapplikation
├── requirements.txt        # Python-dependencies
├── setup.py               # Automatisk setup-script
├── README.md              # Denna fil
└── vosk-model-sv-*        # Vosk-modell (laddas ner automatiskt)
```

## 🔄 Teknisk Arkitektur

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Mikrofon  │───▶│     Vosk     │───▶│   GPT-SW3   │
│   (Audio)   │    │ (Speech-to-  │    │  (Svenska   │
│             │    │   Text)      │    │    AI)      │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│  Högtalare  │◀───│     gTTS     │◀──────────┘
│   (Audio)   │    │ (Text-to-    │
│             │    │  Speech)     │
└─────────────┘    └──────────────┘
```

## 🤝 Bidrag (Contributing)

1. Forka projektet
2. Skapa en feature-branch
3. Committa dina ändringar
4. Pusha till branchen
5. Öppna en Pull Request

## 📄 Licens (License)

Detta projekt är öppen källkod under MIT-licensen.

## 🙏 Acknowledgments

- **Vosk**: För utmärkt offline taligenkänning
- **AI Sweden**: För GPT-SW3 svenska språkmodellen
- **Hugging Face**: För transformers-biblioteket
- **Google**: För gTTS text-till-tal
- **Gradio**: För enkelt webb-gränssnitt

## 📞 Support

Om du stöter på problem:

1. Kolla [Issues](https://github.com/your-repo/issues) först
2. Skapa en ny Issue med detaljerad beskrivning
3. Inkludera systeminfo och felmeddelanden

---

**Lycka till med din svenska röst-chatbot! 🇸🇪🤖**
