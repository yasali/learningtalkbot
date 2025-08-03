# 🤖 Swedish Voice Chatbot - Model Setup Guide

This guide explains exactly which AI models to download and how to load them locally for the Swedish Voice Chatbot.

## 📋 Required Models

### 1. 🎤 Vosk Swedish Speech Recognition Model

**Model Name**: `vosk-model-sv-rhasspy-0.15`  
**Purpose**: Convert Swedish speech to text (offline)  
**Size**: ~45MB  
**License**: Apache 2.0  

#### Direct Download Links:
- **Primary**: https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
- **Mirror**: https://huggingface.co/alphacep/vosk-model-sv-rhasspy-0.15

#### Manual Download Steps:
```bash
# Method 1: Direct download
wget https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
unzip vosk-model-sv-rhasspy-0.15.zip
rm vosk-model-sv-rhasspy-0.15.zip

# Method 2: Using curl
curl -L -o vosk-model-sv-rhasspy-0.15.zip https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
unzip vosk-model-sv-rhasspy-0.15.zip
rm vosk-model-sv-rhasspy-0.15.zip
```

#### Expected Directory Structure:
```
vosk-model-sv-rhasspy-0.15/
├── am/
├── graph/
├── ivector/
├── conf/
└── README
```

### 2. 🧠 GPT-SW3 Swedish Language Model

**Model Name**: `AI-Sweden/gpt-sw3-126m` (recommended for most users)  
**Purpose**: Generate Swedish text responses  
**Size**: ~500MB  
**License**: MIT  

#### Available Model Sizes:
| Size | Model Name | Size | RAM Required | Quality | Speed |
|------|------------|------|--------------|---------|-------|
| Small | `AI-Sweden/gpt-sw3-126m` | ~500MB | 2GB | Basic | Fast |
| Medium | `AI-Sweden/gpt-sw3-356m` | ~1.4GB | 4GB | Good | Medium |
| Large | `AI-Sweden/gpt-sw3-1.3b` | ~5GB | 8GB+ | Best | Slow |

#### Automatic Download (Recommended):
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download and cache the model
model_name = "AI-Sweden/gpt-sw3-126m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

#### Manual Download Using Git LFS:
```bash
# Install git-lfs if not already installed
git lfs install

# Clone the model repository
git clone https://huggingface.co/AI-Sweden/gpt-sw3-126m

# The model will be in the gpt-sw3-126m/ directory
```

## 🚀 Automated Download Script

Use our automated download script to get both models:

```bash
# Run the download script
python download_models.py
```

This script will:
- Show model information and download sizes
- Download Vosk Swedish model with progress bar
- Download GPT-SW3 model via Hugging Face
- Test that both models load correctly
- Provide manual fallback instructions if needed

## 📁 Storage Locations

### Default Locations:
- **Vosk Model**: `./vosk-model-sv-rhasspy-0.15/` (in project directory)
- **GPT-SW3 Model**: `~/.cache/huggingface/transformers/` (Hugging Face cache)

### Custom Locations:
You can specify custom paths in the chatbot code:

```python
# Custom Vosk model path
MODEL_DIR = "/path/to/your/vosk-model-sv-rhasspy-0.15"

# Custom GPT-SW3 model path  
model_name = "/path/to/your/gpt-sw3-126m"
```

## 🔧 Loading Models in Code

### Loading Vosk Model:
```python
import vosk
import json

# Load the Vosk model
model_dir = "vosk-model-sv-rhasspy-0.15"
vosk_model = vosk.Model(model_dir)

# Create recognizer
rec = vosk.KaldiRecognizer(vosk_model, 16000)  # 16kHz sample rate

# Example usage
if rec.AcceptWaveform(audio_data):
    result = json.loads(rec.Result())
    text = result.get('text', '')
    print(f"Recognized: {text}")
```

### Loading GPT-SW3 Model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
model_name = "AI-Sweden/gpt-sw3-126m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example usage
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Example
response = generate_response("Hej, vad heter du?")
print(response)
```

## 🛠 Manual Installation Steps

### Step 1: Install Dependencies
```bash
pip install vosk transformers torch sounddevice
```

### Step 2: Download Vosk Model
```bash
# Create project directory
mkdir swedish-chatbot
cd swedish-chatbot

# Download Vosk model
wget https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
unzip vosk-model-sv-rhasspy-0.15.zip
rm vosk-model-sv-rhasspy-0.15.zip
```

### Step 3: Download GPT-SW3 Model
```python
# Run this Python script
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading GPT-SW3 model...")
tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-126m")
model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-126m")
print("Download complete!")
```

### Step 4: Test the Models
```python
# Test script
import vosk
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Test Vosk
print("Testing Vosk model...")
vosk_model = vosk.Model("vosk-model-sv-rhasspy-0.15")
print("✅ Vosk model loaded successfully!")

# Test GPT-SW3
print("Testing GPT-SW3 model...")
tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-126m")
model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-126m")
print("✅ GPT-SW3 model loaded successfully!")

print("🎉 All models ready!")
```

## 🐛 Troubleshooting

### Vosk Model Issues:
```bash
# If download fails, try alternative sources:
# 1. Hugging Face mirror
git lfs install
git clone https://huggingface.co/alphacep/vosk-model-sv-rhasspy-0.15

# 2. Check file integrity
ls -la vosk-model-sv-rhasspy-0.15/
# Should show folders: am, graph, ivector, conf

# 3. Test loading
python -c "import vosk; vosk.Model('vosk-model-sv-rhasspy-0.15')"
```

### GPT-SW3 Model Issues:
```bash
# Clear Hugging Face cache if corrupted
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip uninstall transformers
pip install transformers

# Test with minimal example
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m')"
```

### Memory Issues:
```bash
# For low memory systems, use smallest model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m')
model = AutoModelForCausalLM.from_pretrained('AI-Sweden/gpt-sw3-126m', torch_dtype='float16')
"
```

## 📊 Model Performance Comparison

| Model | Size | Speed | Quality | RAM Usage | Best For |
|-------|------|-------|---------|-----------|----------|
| gpt-sw3-126m | 500MB | Fast | Basic | 2GB | Testing, Low-end hardware |
| gpt-sw3-356m | 1.4GB | Medium | Good | 4GB | Balanced usage |
| gpt-sw3-1.3b | 5GB | Slow | Excellent | 8GB+ | High-quality responses |

## 🎯 Quick Start Commands

```bash
# Complete automated setup
python download_models.py

# Manual setup
wget https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip
unzip vosk-model-sv-rhasspy-0.15.zip
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m'); AutoModelForCausalLM.from_pretrained('AI-Sweden/gpt-sw3-126m')"

# Test everything
python -c "import vosk; from transformers import AutoTokenizer, AutoModelForCausalLM; vosk.Model('vosk-model-sv-rhasspy-0.15'); AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m'); print('✅ All models ready!')"
```

---

**Note**: All models are free and open-source. The first download may take 10-30 minutes depending on your internet speed, but subsequent loads will be much faster as models are cached locally.