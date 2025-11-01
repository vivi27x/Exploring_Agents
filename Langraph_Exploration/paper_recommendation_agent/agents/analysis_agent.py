import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict
from utils.helpers import load_config

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, model_path: str):
        self.config = load_config()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            logger.info("Loaded fine-tuned relevance model")
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {e}. Using fallback scoring.")
            self.model = None
    
    def analyze_relevance(self, user_interests: str, paper: Dict) -> Dict:
        """Analyze relevance of paper to user interests"""
        if self.model is None:
            # Fallback: use search score
            return {
                "paper": paper,
                "relevance_score": paper.get('search_score', 0.5),
                "justification": "Using search similarity score (fine-tuned model not available)"
            }
        
        try:
            # Prepare input text
            text = f"Interests: {user_interests} Paper: {paper['title']} {paper['abstract'][:400]}"
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                relevance_score = torch.sigmoid(outputs.logits).item()
            
            # Generate simple justification
            justification = self._generate_justification(user_interests, paper, relevance_score)
            
            return {
                "paper": paper,
                "relevance_score": relevance_score,
                "justification": justification
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {
                "paper": paper,
                "relevance_score": paper.get('search_score', 0.5),
                "justification": "Error in analysis, using fallback score"
            }
    
    def _generate_justification(self, interests: str, paper: Dict, score: float) -> str:
        """Generate a simple justification for the relevance score"""
        if score > 0.7:
            return f"Highly relevant! Paper strongly aligns with your interests in {interests.split()[0]}"
        elif score > 0.5:
            return f"Moderately relevant. Paper touches on aspects of {interests.split()[0]}"
        else:
            return "Limited relevance to your specific interests"