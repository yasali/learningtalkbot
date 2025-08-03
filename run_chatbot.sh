#!/bin/bash

# Swedish Voice Chatbot Launcher Script
# This script sets up the environment and runs the chatbot

echo "🇸🇪 Swedish Voice Chatbot Launcher"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "chatbot_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the setup first:"
    echo "   python3 setup.py"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source chatbot_env/bin/activate

# Check if main script exists
if [ ! -f "swedish_chatbot.py" ]; then
    echo "❌ swedish_chatbot.py not found!"
    echo "Make sure you're in the correct directory."
    exit 1
fi

# Check Python dependencies
echo "🧪 Checking dependencies..."
python -c "import vosk, transformers, gradio, gtts" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some dependencies are missing!"
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=./models_cache

# Check system audio
echo "🔊 Checking audio system..."
if command -v pulseaudio >/dev/null 2>&1; then
    pulseaudio --check -v
    if [ $? -ne 0 ]; then
        echo "⚠️  Starting PulseAudio..."
        pulseaudio --start --exit-idle-time=-1
    fi
fi

# Launch the chatbot
echo "🚀 Launching Swedish Voice Chatbot..."
echo "📱 The web interface will open in your browser automatically"
echo "🛑 Press Ctrl+C to stop the chatbot"
echo ""

python swedish_chatbot.py

echo ""
echo "👋 Chatbot stopped. Goodbye!"