import os
import requests
import logging
from typing import Dict, Any, List
from utils.helpers import load_config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import dotenv
dotenv.load_dotenv()

class HuggingFaceClient:
    def __init__(self):
        self.api_key = os.getenv('HF_TOKEN')
        if not self.api_key:
            logger.warning("HF_TOKEN environment variable not set.")
            
        # This is the single, correct endpoint for the chat router
        self.base_url = "https://router.huggingface.co/v1/chat/completions"
        
    def chat_completion(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """
        Generate a chat completion using the HF Chat Completions API.
        """
        if not self.api_key:
            logger.error("API key is not available.")
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload matches the format from your first script
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7  # You can add other parameters here
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # The response structure is {"choices": [{"message": {"content": "..."}}]}
            if result.get("choices") and len(result["choices"]) > 0:
                message_content = result["choices"][0].get("message", {}).get("content", "")
                return message_content
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ""

# --- Example Usage ---
if __name__ == "__main__":
    # Make sure to set your HF_TOKEN in your environment
    # export HF_TOKEN='your_hf_token_here'
    
    client = HuggingFaceClient()
    
    if client.api_key:
        messages = [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
        model_name = "meta-llama/Llama-3.1-8B-Instruct:novita"
        
        response_text = client.chat_completion(model_name, messages)
        
        if response_text:
            print(f"Assistant: {response_text}")
        else:
            print("Failed to get a response.")