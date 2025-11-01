import logging
from typing import List, Dict
from utils.helpers import load_config
from utils.hf_client import HuggingFaceClient

logger = logging.getLogger(__name__)

class JustificationAgent:
    def __init__(self, model_name: str = None):
        self.config = load_config()
        self.model_name = model_name or self.config['models']['justification']
        self.client = HuggingFaceClient()
    
    def format_recommendations(self, user_query: str, analyzed_papers: List[Dict]) -> str:
        """Format recommendations with detailed justifications"""
        try:
            # Sort by relevance score
            sorted_papers = sorted(analyzed_papers, key=lambda x: x["relevance_score"], reverse=True)
            
            # Take top papers
            top_papers = sorted_papers[:10]
            
            # Generate detailed justifications for top papers
            for i, paper in enumerate(top_papers[:3]):  # Limit to 3 to save API calls
                if paper["relevance_score"] > 0.5:
                    detailed_justification = self._generate_detailed_justification(user_query, paper)
                    paper["detailed_justification"] = detailed_justification
            
            return self._create_output_format(user_query, top_papers)
            
        except Exception as e:
            logger.error(f"Error in justification: {e}")
            return self._create_fallback_output(analyzed_papers)
    
    def _generate_detailed_justification(self, user_query: str, paper: Dict) -> str:
        """Generate detailed justification using HF API"""
        prompt = f"""
        User research interests: "{user_query}"
        
        Paper: {paper['paper']['title']}
        Abstract: {paper['paper']['abstract'][:500]}
        
        Explain specifically why this paper is relevant to the user's research interests.
        Focus on technical connections and practical relevance.
        Keep it concise (2-3 sentences).
        
        Justification:
        """
        
        try:
            response = self.client.generate_text(self.model_name, prompt, max_tokens=150)
            return response.strip() if response else paper["justification"]
        except Exception as e:
            logger.error(f"Error generating detailed justification: {e}")
            return paper["justification"]
    
    def _create_output_format(self, user_query: str, papers: List[Dict]) -> str:
        """Create formatted output string"""
        output = [f"# Paper Recommendations for: {user_query}\n"]
        output.append(f"Found {len(papers)} relevant papers\n")
        
        for i, paper_data in enumerate(papers):
            paper = paper_data['paper']
            output.append(f"## {i+1}. {paper['title']}")
            output.append(f"**Relevance Score:** {paper_data['relevance_score']:.3f}")
            output.append(f"**Categories:** {', '.join(paper['categories'])}")
            output.append(f"**Published:** {paper['published']}")
            
            if 'detailed_justification' in paper_data:
                output.append(f"**Why it's relevant:** {paper_data['detailed_justification']}")
            else:
                output.append(f"**Why it's relevant:** {paper_data['justification']}")
            
            output.append(f"**Abstract preview:** {paper['abstract'][:200]}...")
            output.append("---")
        
        return "\n".join(output)
    
    def _create_fallback_output(self, papers: List[Dict]) -> str:
        """Create fallback output format"""
        sorted_papers = sorted(papers, key=lambda x: x["relevance_score"], reverse=True)[:10]
        
        output = ["# Paper Recommendations\n"]
        for i, paper_data in enumerate(sorted_papers):
            paper = paper_data['paper']
            score = paper_data['relevance_score']
            relevance_level = "Highly relevant" if score > 0.7 else "Moderately relevant" if score > 0.5 else "Somewhat relevant"
            
            output.append(f"{i+1}. **{paper['title']}** ({relevance_level}, Score: {score:.3f})")
            output.append(f"   {paper_data['justification']}")
            output.append("")
        
        return "\n".join(output)