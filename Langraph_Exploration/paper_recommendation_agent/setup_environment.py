import os
import getpass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
def setup_environment():
    """Setup environment for Hugging Face API"""
    
    print("🤗 Hugging Face API Setup")
    print("=" * 50)
    
    # Get Hugging Face token
    token = getpass.getpass("Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens): ")
    
    # Write to .env file
    with open(".env", "w") as f:
        f.write(f"HUGGINGFACE_API_KEY={token}\n")
    
    # Update config.yaml
    with open("config.yaml", "r") as f:
        config_content = f.read()
    
    # Replace API key in config
    config_content = config_content.replace('api_key: "your_huggingface_token_here"', f'api_key: "{token}"')
    
    with open("config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Environment setup completed!")
    print("📝 You can now run the application")

if __name__ == "__main__":
    setup_environment()