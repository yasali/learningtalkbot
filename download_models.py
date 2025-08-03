#!/usr/bin/env python3
"""
Swedish Voice Chatbot - Model Download Script
Downloads and sets up all required AI models for the chatbot
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_with_progress(url, filename):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
            print(f"\r[{bar}] {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end='', flush=True)
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress bar

def download_vosk_model():
    """Download Vosk Swedish speech recognition model"""
    print("🎤 DOWNLOADING VOSK SWEDISH MODEL")
    print("=" * 50)
    
    model_dir = "vosk-model-sv-rhasspy-0.15"
    model_zip = "vosk-model-sv-rhasspy-0.15.zip"
    model_url = "https://alphacephei.com/vosk/models/vosk-model-sv-rhasspy-0.15.zip"
    
    if os.path.exists(model_dir):
        print(f"✅ Vosk model already exists in {model_dir}")
        return model_dir
    
    print(f"📥 Downloading from: {model_url}")
    print(f"📁 Model size: ~45MB")
    print()
    
    try:
        # Download the model
        download_with_progress(model_url, model_zip)
        
        # Extract the model
        print("📂 Extracting model...")
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up zip file
        os.remove(model_zip)
        
        print(f"✅ Vosk Swedish model downloaded and extracted to: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"❌ Error downloading Vosk model: {e}")
        print("\n🔗 MANUAL DOWNLOAD OPTION:")
        print(f"1. Download manually from: {model_url}")
        print(f"2. Extract to current directory")
        print(f"3. Ensure folder is named: {model_dir}")
        return None

def download_gpt_sw3_model():
    """Download GPT-SW3 model using Hugging Face"""
    print("\n🧠 DOWNLOADING GPT-SW3 SWEDISH MODEL")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Available GPT-SW3 models (choose based on your hardware)
        models = {
            "small": "AI-Sweden/gpt-sw3-126m",      # ~500MB, fast, basic responses
            "medium": "AI-Sweden/gpt-sw3-356m",     # ~1.4GB, better quality
            "large": "AI-Sweden/gpt-sw3-1.3b",      # ~5GB, best quality, requires good GPU
        }
        
        print("Available GPT-SW3 models:")
        for size, model_name in models.items():
            print(f"  {size.upper()}: {model_name}")
        
        # Default to medium model (good balance)
        model_name = models["small"]  # Start with smallest for compatibility
        
        print(f"\n📥 Downloading model: {model_name}")
        print("📁 This may take several minutes depending on your internet connection...")
        print("💾 Models will be cached in ~/.cache/huggingface/")
        
        # Download tokenizer
        print("\n🔤 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer downloaded successfully!")
        
        # Download model
        print("\n🤖 Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ GPT-SW3 model downloaded successfully!")
        
        return model_name, tokenizer, model
        
    except ImportError:
        print("❌ Error: transformers library not installed")
        print("💡 Please install with: pip install transformers torch")
        return None, None, None
    except Exception as e:
        print(f"❌ Error downloading GPT-SW3 model: {e}")
        print("\n🔗 MANUAL DOWNLOAD OPTION:")
        print("1. Install transformers: pip install transformers torch")
        print("2. Run Python and execute:")
        print("   from transformers import AutoTokenizer, AutoModelForCausalLM")
        print("   tokenizer = AutoTokenizer.from_pretrained('AI-Sweden/gpt-sw3-126m')")
        print("   model = AutoModelForCausalLM.from_pretrained('AI-Sweden/gpt-sw3-126m')")
        return None, None, None

def test_models():
    """Test that models can be loaded"""
    print("\n🧪 TESTING MODELS")
    print("=" * 50)
    
    # Test Vosk model
    try:
        import vosk
        model_dir = "vosk-model-sv-rhasspy-0.15"
        if os.path.exists(model_dir):
            vosk_model = vosk.Model(model_dir)
            print("✅ Vosk model loads successfully!")
        else:
            print("❌ Vosk model directory not found")
    except Exception as e:
        print(f"❌ Error loading Vosk model: {e}")
    
    # Test GPT-SW3 model
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "AI-Sweden/gpt-sw3-126m"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ GPT-SW3 model loads successfully!")
    except Exception as e:
        print(f"❌ Error loading GPT-SW3 model: {e}")

def show_model_info():
    """Show information about the models that will be downloaded"""
    print("🇸🇪 SWEDISH VOICE CHATBOT - MODEL INFORMATION")
    print("=" * 60)
    
    print("\n📋 MODELS TO DOWNLOAD:")
    print("\n1. 🎤 VOSK SWEDISH SPEECH RECOGNITION MODEL")
    print("   - Name: vosk-model-sv-rhasspy-0.15")
    print("   - Size: ~45MB")
    print("   - Purpose: Convert Swedish speech to text")
    print("   - Source: https://alphacephei.com/vosk/models/")
    print("   - License: Apache 2.0")
    print("   - Works: Completely offline")
    
    print("\n2. 🧠 GPT-SW3 SWEDISH LANGUAGE MODEL")
    print("   - Name: AI-Sweden/gpt-sw3-126m (small version)")
    print("   - Size: ~500MB")
    print("   - Purpose: Generate Swedish text responses")
    print("   - Source: Hugging Face (AI Sweden)")
    print("   - License: MIT")
    print("   - Works: Locally, no internet needed after download")
    
    print("\n📁 STORAGE LOCATIONS:")
    print(f"   - Vosk model: {os.path.abspath('vosk-model-sv-rhasspy-0.15')}")
    print(f"   - GPT-SW3 cache: {os.path.expanduser('~/.cache/huggingface/')}")
    
    print("\n💾 TOTAL DISK SPACE NEEDED: ~600MB")
    print("\n⏱️  DOWNLOAD TIME: 5-15 minutes (depending on internet speed)")

def main():
    """Main function to download all models"""
    show_model_info()
    
    response = input("\nProceed with download? (y/N): ").lower().strip()
    if response != 'y':
        print("Download cancelled.")
        sys.exit(0)
    
    print("\n🚀 STARTING MODEL DOWNLOADS...")
    
    # Download Vosk model
    vosk_path = download_vosk_model()
    
    # Download GPT-SW3 model
    model_name, tokenizer, model = download_gpt_sw3_model()
    
    # Test models
    test_models()
    
    print("\n🎉 MODEL DOWNLOAD COMPLETE!")
    print("=" * 50)
    
    if vosk_path and model_name:
        print("✅ All models downloaded successfully!")
        print("\n📋 WHAT'S NEXT:")
        print("1. Run the chatbot: python swedish_chatbot.py")
        print("2. Open browser to: http://localhost:7860")
        print("3. Click 🎤 Prata and speak Swedish!")
        
        print(f"\n📁 FILES CREATED:")
        print(f"   - Vosk model: {vosk_path}/")
        print(f"   - GPT-SW3 cache: ~/.cache/huggingface/transformers/")
    else:
        print("⚠️  Some models failed to download.")
        print("Please check the manual download instructions above.")

if __name__ == "__main__":
    main()