import json
import random
from typing import List, Dict
from datasets import load_dataset

def generate_training_data() -> List[Dict]:
    """Generate training data for relevance model"""
    # Load some base dataset for academic papers
    try:
        arxiv_dataset = load_dataset("arxiv_dataset", split="train[:1000]")
    except:
        # Fallback: create synthetic data
        return create_synthetic_data()
    
    training_samples = []
    
    # Create synthetic user interests and relevance scores
    research_interests = [
        "machine learning deep learning neural networks",
        "natural language processing transformers BERT",
        "computer vision convolutional networks object detection", 
        "reinforcement learning Q-learning policy gradients",
        "graph neural networks network embedding",
        "time series forecasting ARIMA LSTM",
        "few-shot learning meta learning",
        "self-supervised learning contrastive learning"
    ]
    
    for paper in arxiv_dataset:
        if len(training_samples) >= 500:  # Limit dataset size
            break
            
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        
        # Create multiple training samples per paper with different interests
        for interests in random.sample(research_interests, 2):
            # Simulate relevance based on keyword overlap
            relevance_score = calculate_synthetic_relevance(interests, paper_text)
            
            training_samples.append({
                "user_interests": interests,
                "paper_title": paper.get('title', ''),
                "paper_abstract": paper.get('abstract', ''),
                "paper_categories": paper.get('categories', []),
                "relevance_score": relevance_score,
                "justification": f"This paper is relevant because it covers topics related to {interests.split()[0]}"
            })
    
    return training_samples

def calculate_synthetic_relevance(interests: str, paper_text: str) -> float:
    """Calculate synthetic relevance score based on keyword overlap"""
    interest_words = set(interests.lower().split())
    paper_words = set(paper_text.lower().split())
    
    overlap = len(interest_words.intersection(paper_words))
    max_possible = len(interest_words)
    
    return min(overlap / max_possible if max_possible > 0 else 0, 1.0)

def create_synthetic_data() -> List[Dict]:
    """Create completely synthetic training data"""
    samples = []
    
    paper_templates = [
        {
            "title": "Advanced Methods in Deep Learning for Natural Language Processing",
            "abstract": "We present novel deep learning architectures for NLP tasks including text classification and generation.",
            "categories": ["cs.CL", "cs.LG"]
        },
        {
            "title": "Computer Vision Approaches for Object Detection in Real Time",
            "abstract": "This paper explores efficient convolutional networks for real-time object detection in video streams.",
            "categories": ["cs.CV"]
        }
    ]
    
    interests = ["natural language processing", "computer vision", "deep learning"]
    
    for paper in paper_templates:
        for interest in interests:
            score = 0.8 if any(word in paper["abstract"].lower() for word in interest.split()) else 0.2
            
            samples.append({
                "user_interests": interest,
                "paper_title": paper["title"],
                "paper_abstract": paper["abstract"],
                "paper_categories": paper["categories"],
                "relevance_score": score,
                "justification": f"Paper discusses {interest} techniques" if score > 0.5 else "Limited relevance to interests"
            })
    
    return samples

if __name__ == "__main__":
    training_data = generate_training_data()
    with open("data/training_data/training_samples.json", "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"Generated {len(training_data)} training samples")