import json
import logging
from typing import Dict
from utils.helpers import load_config
from utils.hf_client import HuggingFaceClient

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self, model_name: str = None):
        self.config = load_config()
        self.model_name = model_name or self.config['models']['planner']
        self.client = HuggingFaceClient()
    
    def plan(self, user_query: str) -> Dict:
        """Create a search plan based on user interests"""
        prompt = f"""
        Analyze this research interest and create a structured search plan:
        "{user_query}"
        
        Output a JSON object with the following structure:
        {{
            "domains": ["list of relevant academic domains e.g., cs.AI, cs.LG"],
            "key_concepts": ["list of key technical concepts and keywords"],
            "recency_preference": "preferred time frame (e.g., last 2 years, all time)",
            "depth": "level of detail needed (e.g., foundational, recent advances, comprehensive)",
            "specific_requirements": ["any specific requirements or constraints"]
        }}
        
        Be precise and extract the main technical concepts.
        Return ONLY the JSON object, no additional text.
        JSON:
        """
        
        try:
            response = self.client.generate_text(self.model_name, prompt, max_tokens=256)
            
            if not response:
                logger.warning("Empty response from HF API, using fallback")
                return self._create_fallback_plan(user_query)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                plan = json.loads(json_str)
                logger.info(f"Generated plan: {plan}")
                return plan
            else:
                logger.warning("No JSON found in response, using fallback")
                return self._create_fallback_plan(user_query)
            
        except Exception as e:
            logger.error(f"Error in planner: {e}")
            return self._create_fallback_plan(user_query)
    
    def _create_fallback_plan(self, user_query: str) -> Dict:
        """Create fallback plan when API fails"""
        # Simple keyword extraction
        words = user_query.lower().split()
        technical_terms = [word for word in words if len(word) > 3][:5]
        
        return {
            "domains": ["cs.AI", "cs.LG", "cs.CL"],
            "key_concepts": technical_terms,
            "recency_preference": "last 3 years",
            "depth": "comprehensive",
            "specific_requirements": []
        }