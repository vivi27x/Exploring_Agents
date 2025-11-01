import yaml
import json
import logging
from typing import Dict, Any, List

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('agent.log'),
            logging.StreamHandler()
        ]
    )

def save_recommendations(recommendations: List[Dict], filename: str):
    """Save recommendations to JSON file"""
    with open(filename, 'w') as f:
        json.dump(recommendations, f, indent=2)

def load_recommendations(filename: str) -> List[Dict]:
    """Load recommendations from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)