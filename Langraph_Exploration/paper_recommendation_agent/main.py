import logging
from agents.planner_agent import PlannerAgent
from agents.search_agent import SearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.justification_agent import JustificationAgent
from utils.helpers import setup_logging, load_config, save_recommendations

class PaperRecommendationAgent:
    def __init__(self):
        self.config = load_config()
        setup_logging()
        
        # Initialize agents
        # self.planner = PlannerAgent(self.config['models']['planner'])
        self.planner = PlannerAgent('llama-3.3-70b-versatile')

        self.searcher = SearchAgent()

        self.analyzer = AnalysisAgent(self.config['models']['analysis'])
        self.justifier = JustificationAgent()
        
        self.logger = logging.getLogger(__name__)
    
    def recommend(self, user_query: str, save_output: bool = True) -> dict:
        """Main recommendation pipeline"""
        self.logger.info(f"Starting recommendation for: {user_query}")
        
        try:
            # Step 1: Plan
            self.logger.info("Planning search...")
            plan = self.planner.plan(user_query)
            
            # Step 2: Search
            self.logger.info("Searching for papers...")
            candidate_papers = self.searcher.search(plan)
            
            if not candidate_papers:
                self.logger.warning("No papers found in search")
                return {"error": "No papers found matching your query"}
            
            # Step 3: Analyze
            self.logger.info("Analyzing paper relevance...")
            analyzed_papers = []
            for paper in candidate_papers:
                analysis = self.analyzer.analyze_relevance(user_query, paper)
                analyzed_papers.append(analysis)
            
            # Step 4: Justify and format
            self.logger.info("Formatting recommendations...")
            recommendations = self.justifier.format_recommendations(user_query, analyzed_papers)
            
            # Prepare result
            result = {
                "query": user_query,
                "plan": plan,
                "recommendations": analyzed_papers[:self.config['agent']['max_recommendations']],
                "formatted_output": recommendations,
                "total_candidates": len(analyzed_papers)
            }
            
            # Save results
            if save_output:
                save_recommendations(result, "recommendations.json")
            
            self.logger.info("Recommendation process completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in recommendation pipeline: {e}")
            return {"error": f"Processing failed: {str(e)}"}

def main():
    """Main function for command line usage"""
    agent = PaperRecommendationAgent()
    
    print("🤖 Academic Paper Recommendation Agent")
    print("=" * 50)
    
    user_query = input("Enter your research interests: ").strip()
    
    if not user_query:
        user_query = "machine learning deep learning neural networks"
        print(f"Using default query: {user_query}")
    
    print("\nProcessing your request...")
    result = agent.recommend(user_query)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\n" + "=" * 50)
        print(result['formatted_output'])
        print(f"\n📊 Found {result['total_candidates']} candidate papers")
        print(f"💾 Results saved to recommendations.json")

if __name__ == "__main__":
    main()