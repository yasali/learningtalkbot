#!/bin/bash

echo "🇸🇪 SWEDISH CHATBOT - Environment Setup"
echo "======================================"
echo ""
echo "This script will create a clean virtual environment and install dependencies."
echo "The virtual environment is NOT included in the repository (it's in .gitignore)."
echo ""

# Check if we're on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - using Homebrew setup"
    
    # Install system dependencies with Homebrew
    echo "📦 Installing system dependencies..."
    brew update
    brew install python@3.11 portaudio ffmpeg
    
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux - using apt setup"
    
    # Install system dependencies with apt
    echo "📦 Installing system dependencies..."
    sudo apt update
    sudo apt install -y python3.13-venv python3-pip portaudio19-dev ffmpeg mpg123 alsa-utils libsdl2-dev libfreetype-dev
    
    PYTHON_CMD="python3"
else
    echo "❌ Unsupported operating system"
    exit 1
fi

# Remove old virtual environment if it exists
if [ -d "chatbot_env" ]; then
    echo "🗑️ Removing old virtual environment..."
    rm -rf chatbot_env
fi

# Create new virtual environment
echo "🆕 Creating new virtual environment..."
$PYTHON_CMD -m venv chatbot_env

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source chatbot_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "🚀 To use the chatbot:"
echo "1. source chatbot_env/bin/activate"
echo "2. export HF_TOKEN='your_huggingface_token'"
echo "3. python swedish_chatbot_fixed.py"
echo ""
echo "💡 Or use the convenient script:"
echo "./start_fixed_chatbot.sh"
echo ""
echo "📝 Remember: The chatbot_env/ folder is ignored by git and won't be committed."