#!/usr/bin/env python3
"""
Swedish Voice Chatbot - Hugging Face Token Setup
Helps you set up your Hugging Face token for accessing Swedish models
"""

import os
import sys

def setup_hf_token():
    """Guide user through Hugging Face token setup"""
    print("🔑 HUGGING FACE TOKEN SETUP")
    print("=" * 50)
    print("To access Swedish GPT-SW3 models, you need a Hugging Face token.")
    print()
    
    print("📋 STEPS TO GET YOUR TOKEN:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Log in or create a free account")
    print("3. Click 'New token'")
    print("4. Choose 'Read' permission")
    print("5. Copy the generated token")
    print()
    
    # Check if token already exists
    existing_token = os.environ.get("HF_TOKEN")
    if existing_token:
        print(f"✅ HF_TOKEN already set: {existing_token[:10]}...{existing_token[-5:]}")
        
        response = input("Do you want to update it? (y/N): ").lower().strip()
        if response != 'y':
            print("Keeping existing token.")
            return existing_token
    
    # Get token from user
    print("🔗 Enter your Hugging Face token:")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return None
    
    if len(token) < 20:
        print("⚠️ Token seems too short. Hugging Face tokens are usually longer.")
        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            return None
    
    # Offer setup methods
    print("\n📋 HOW TO SET UP YOUR TOKEN:")
    print("1. Environment variable (temporary)")
    print("2. Add to shell profile (permanent)")
    print("3. Hugging Face CLI login")
    print("4. Just test it now")
    
    choice = input("\nChoose method (1-4): ").strip()
    
    if choice == "1":
        # Set environment variable
        os.environ["HF_TOKEN"] = token
        print(f"✅ HF_TOKEN set for this session!")
        print("To make it permanent, add this to your ~/.bashrc or ~/.zshrc:")
        print(f"export HF_TOKEN='{token}'")
        
    elif choice == "2":
        # Add to shell profile
        shell_profiles = ["~/.bashrc", "~/.zshrc", "~/.profile"]
        
        print("Available shell profiles:")
        for i, profile in enumerate(shell_profiles, 1):
            expanded = os.path.expanduser(profile)
            exists = "✅" if os.path.exists(expanded) else "❌"
            print(f"  {i}. {profile} {exists}")
        
        profile_choice = input("Choose profile (1-3): ").strip()
        
        try:
            profile_idx = int(profile_choice) - 1
            profile_path = os.path.expanduser(shell_profiles[profile_idx])
            
            with open(profile_path, 'a') as f:
                f.write(f"\n# Hugging Face token for Swedish Voice Chatbot\n")
                f.write(f"export HF_TOKEN='{token}'\n")
            
            print(f"✅ Token added to {profile_path}")
            print("Restart your terminal or run: source ~/.bashrc")
            
        except (ValueError, IndexError, FileNotFoundError) as e:
            print(f"❌ Error writing to profile: {e}")
            
    elif choice == "3":
        # Use Hugging Face CLI
        try:
            import subprocess
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", token],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Logged in with Hugging Face CLI!")
            else:
                print(f"❌ CLI login failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ huggingface-cli not found. Install with: pip install huggingface-hub")
            
    elif choice == "4":
        # Just test
        os.environ["HF_TOKEN"] = token
        print("✅ Token set for testing!")
        
    else:
        print("Invalid choice. Token not saved.")
        return None
    
    return token

def test_token(token=None):
    """Test if the token works with Swedish models"""
    print("\n🧪 TESTING TOKEN...")
    
    if not token:
        token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ No token found to test")
        return False
    
    try:
        from transformers import AutoTokenizer
        from huggingface_hub import login
        
        # Login with token
        login(token=token)
        print("✅ Token authentication successful!")
        
        # Try to access Swedish model
        print("🇸🇪 Testing Swedish model access...")
        tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-126m", token=token)
        print("✅ Swedish model accessible!")
        
        return True
        
    except Exception as e:
        print(f"❌ Token test failed: {e}")
        print("\n💡 POSSIBLE ISSUES:")
        print("1. Invalid token")
        print("2. No internet connection")
        print("3. Model requires special permissions")
        print("4. Token doesn't have read permissions")
        return False

def main():
    """Main setup function"""
    print("🇸🇪 SWEDISH VOICE CHATBOT - TOKEN SETUP")
    print("=" * 60)
    
    # Setup token
    token = setup_hf_token()
    
    if token:
        # Test token
        if test_token(token):
            print("\n🎉 SUCCESS!")
            print("Your Hugging Face token is working correctly!")
            print("\n📋 NEXT STEPS:")
            print("1. Run: python download_models.py")
            print("2. Run: python swedish_chatbot.py")
            print("3. Enjoy your Swedish voice chatbot! 🤖")
        else:
            print("\n⚠️ Token setup completed but testing failed.")
            print("You can still try running the chatbot - it will use fallback models.")
    else:
        print("\n⚠️ No token set up.")
        print("The chatbot will use fallback models (non-Swedish).")
    
    print("\n💡 REMEMBER:")
    print("- Keep your token private")
    print("- Don't share it in code or screenshots")
    print("- You can regenerate it if compromised")

if __name__ == "__main__":
    main()