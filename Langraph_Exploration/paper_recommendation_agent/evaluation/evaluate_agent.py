import json
import numpy as np
from typing import List, Dict
import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")
from main import PaperRecommendationAgent

def evaluate_agent():
    """Evaluate the recommendation agent"""
    agent = PaperRecommendationAgent()
    
    # Test cases
    test_cases = [
        {
            "query": "few-shot learning natural language processing",
            "expected_domains": ["cs.CL", "cs.LG"],
            "min_expected_papers": 5
        },
        {
            "query": "computer vision object detection", 
            "expected_domains": ["cs.CV"],
            "min_expected_papers": 5
        },
        {
            "query": "reinforcement learning deep Q networks",
            "expected_domains": ["cs.LG", "cs.AI"],
            "min_expected_papers": 5
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"Testing: {test_case['query']}")
        
        result = agent.recommend(test_case['query'], save_output=False)
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            results.append({
                "test_case": test_case['query'],
                "status": "error",
                "error": result['error']
            })
            continue
        
        # Check if we got enough papers
        paper_count = len(result['recommendations'])
        sufficient_papers = paper_count >= test_case['min_expected_papers']
        
        # Check if domains match
        planned_domains = result['plan']['domains']
        domain_match = any(domain in planned_domains for domain in test_case['expected_domains'])
        
        # Calculate average relevance score
        avg_relevance = np.mean([r['relevance_score'] for r in result['recommendations']])
        
        test_result = {
            "test_case": test_case['query'],
            "status": "success",
            "paper_count": paper_count,
            "sufficient_papers": sufficient_papers,
            "domain_match": domain_match,
            "avg_relevance": avg_relevance,
            "planned_domains": planned_domains
        }
        
        results.append(test_result)
        
        print(f"  ✅ Papers: {paper_count}, Avg Relevance: {avg_relevance:.3f}")
        print(f"  📊 Domain match: {domain_match}, Sufficient papers: {sufficient_papers}")
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    successful_tests = [r for r in results if r['status'] == 'success']
    
    if successful_tests:
        avg_papers = np.mean([r['paper_count'] for r in successful_tests])
        avg_relevance = np.mean([r['avg_relevance'] for r in successful_tests])
        domain_match_rate = np.mean([r['domain_match'] for r in successful_tests])
        sufficient_papers_rate = np.mean([r['sufficient_papers'] for r in successful_tests])
        
        print(f"Tests completed: {len(successful_tests)}/{len(test_cases)}")
        print(f"Average papers per query: {avg_papers:.1f}")
        print(f"Average relevance score: {avg_relevance:.3f}")
        print(f"Domain match rate: {domain_match_rate:.1%}")
        print(f"Sufficient papers rate: {sufficient_papers_rate:.1%}")
    else:
        print("No successful tests to evaluate")
    
    # Save detailed results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    evaluate_agent()