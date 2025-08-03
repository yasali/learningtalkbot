#!/usr/bin/env python3
"""
Setup script for Swedish Voice Chatbot
Automatically installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True

def install_system_dependencies():
    """Install system-level dependencies based on OS"""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        # Install audio dependencies for Linux
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg mpg123 alsa-utils",
        ]
        for cmd in commands:
            if not run_command(cmd, f"Installing Linux dependencies: {cmd.split()[-1]}"):
                print("⚠️  Some system dependencies might not be installed. Audio might not work properly.")
                
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        # Check if Homebrew is installed
        if subprocess.run("which brew", shell=True, capture_output=True).returncode == 0:
            commands = [
                "brew install portaudio ffmpeg mpg123",
            ]
            for cmd in commands:
                run_command(cmd, f"Installing macOS dependencies")
        else:
            print("⚠️  Homebrew not found. Please install Homebrew first: https://brew.sh/")
            
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️  On Windows, you might need to install additional audio codecs manually")
        print("Consider installing: https://www.codecguide.com/download_kl.htm")
    
def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_name = "chatbot_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True
    
    if run_command(f"python3 -m venv {venv_name}", "Creating virtual environment"):
        print(f"✅ Virtual environment '{venv_name}' created successfully!")
        print(f"📝 To activate it manually:")
        if platform.system().lower() == "windows":
            print(f"   {venv_name}\\Scripts\\activate")
        else:
            print(f"   source {venv_name}/bin/activate")
        return True
    return False

def install_python_dependencies():
    """Install Python dependencies from requirements.txt"""
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Not running in a virtual environment!")
        print("It's recommended to use a virtual environment to avoid conflicts.")
        response = input("Continue anyway? (y/N): ").lower()
        if response != 'y':
            return False
    
    # Install pip packages
    commands = [
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    required_packages = [
        "speech_recognition",
        "vosk", 
        "gtts",
        "gradio",
        "transformers",
        "torch",
        "sounddevice"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Try running: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully!")
    return True

def main():
    """Main setup function"""
    print("🇸🇪 Swedish Voice Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    install_system_dependencies()
    
    # Create virtual environment
    print("\n🏠 Setting up virtual environment...")
    create_virtual_environment()
    
    # Install Python dependencies
    print("\n🐍 Installing Python dependencies...")
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies!")
        return False
    
    # Test imports
    print("\n🧪 Testing installation...")
    if not test_imports():
        print("❌ Some packages failed to import!")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the chatbot:")
    print("   python swedish_chatbot.py")
    print("\n2. Open your browser to the displayed URL")
    print("3. Click the microphone button and start talking in Swedish!")
    
    print("\n💡 Tips:")
    print("- Make sure your microphone is working")
    print("- The first run will download AI models (may take time)")
    print("- Speak clearly in Swedish")
    print("- Internet connection needed for text-to-speech")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)