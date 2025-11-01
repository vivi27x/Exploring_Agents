import ollama
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
    
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
        JSON:
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            plan_text = response['message']['content']
            
            # Extract JSON from response
            json_start = plan_text.find('{')
            json_end = plan_text.rfind('}') + 1
            json_str = plan_text[json_start:json_end]
            
            plan = json.loads(json_str)
            logger.info(f"Generated plan: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"Error in planner: {e}")
            # Fallback plan
            return {
                "domains": ["cs.AI", "cs.LG", "cs.CL"],
                "key_concepts": user_query.split()[:5],
                "recency_preference": "last 3 years",
                "depth": "comprehensive",
                "specific_requirements": []
            }