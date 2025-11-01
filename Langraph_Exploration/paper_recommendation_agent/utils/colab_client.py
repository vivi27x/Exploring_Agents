import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ColabClient:
    """Client for free models hosted on Google Colab"""
    
    def __init__(self):
        # These are example endpoints - you'd need to deploy your own
        self.endpoints = {
            "mistral": "https://your-colab-app-12345.ue.r.appspot.com/generate",
            "llama": "https://your-colab-app-12345.ue.r.appspot.com/generate"
        }
    
    def generate_text(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        """Generate text using Colab-deployed model"""
        endpoint = self.endpoints.get(model)
        if not endpoint:
            logger.error(f"No endpoint for model: {model}")
            return ""
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('text', '')
        except Exception as e:
            logger.error(f"Colab API error: {e}")
            return ""