import os
import requests
import logging
from typing import Dict, Any
from utils.helpers import load_config

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    def __init__(self):
        self.config = load_config()
        self.api_key = self.config['huggingface'].get('api_key') or os.getenv('HUGGINGFACE_API_KEY')
        self.base_url = "https://api-inference.huggingface.co/models"
        
    def generate_text(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Hugging Face Inference API"""
        url = f"{self.base_url}/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ""

    def chat_completion(self, model: str, messages: list, max_tokens: int = 512) -> str:
        """Simulate chat completion using text generation"""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        return self.generate_text(model, prompt, max_tokens)
    
    def _messages_to_prompt(self, messages: list) -> str:
        """Convert chat messages to a single prompt"""
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt