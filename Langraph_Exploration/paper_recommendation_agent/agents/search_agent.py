import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict
from utils.helpers import load_config

logger = logging.getLogger(__name__)

class SearchAgent:
    def __init__(self):
        config = load_config()
        self.embedder = SentenceTransformer(config['models']['embedding'])
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=config['paths']['vector_db'])
        self.collection = self.client.get_collection("arxiv_papers")
        
        self.search_top_k = config['agent']['search_top_k']
    
    def search(self, plan: Dict) -> List[Dict]:
        """Search for papers based on the plan"""
        try:
            # Create search query from key concepts
            search_query = " ".join(plan["key_concepts"])
            
            # Generate embedding for the query
            query_embedding = self.embedder.encode(search_query).tolist()
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.search_top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            papers = []
            for i in range(len(results['ids'][0])):
                paper = {
                    'id': results['ids'][0][i],
                    'title': results['metadatas'][0][i]['title'],
                    'abstract': results['documents'][0][i],
                    'categories': results['metadatas'][0][i]['categories'],
                    'published': results['metadatas'][0][i]['published'],
                    'pdf_url': results['metadatas'][0][i].get('pdf_url', ''),
                    'search_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} candidate papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []