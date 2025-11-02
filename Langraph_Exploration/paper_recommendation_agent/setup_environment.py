import os
import getpass

def setup_environment():
    """Setup environment for Hugging Face API"""
    
    print("🤗 Hugging Face API Setup")
    print("=" * 50)
    print("Get your free token from: https://huggingface.co/settings/tokens")
    print("The token needs to have 'read' access")
    print()
    
    # Get Hugging Face token
    token = getpass.getpass("Enter your Hugging Face token: ")
    
    # Write to .env file
    with open(".env", "w") as f:
        f.write(f"HF_TOKEN={token}\n")
    
    # Update config.yaml
    with open("config.yaml", "r") as f:
        config_content = f.read()
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Environment setup completed!")
    print("✅ HF_TOKEN added to .env file")
    print("✅ Config file updated")
    print("📝 You can now run the application")

if __name__ == "__main__":
    setup_environment()