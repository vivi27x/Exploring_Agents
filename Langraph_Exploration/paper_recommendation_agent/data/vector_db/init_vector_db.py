import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

from utils.helpers import load_config
def initialize_vector_db():
    """Initialize ChromaDB with paper data"""
    config = load_config()
    
    # Initialize embedding model
    embedder = SentenceTransformer(config['models']['embedding'])
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=config['paths']['vector_db'])
    collection = client.get_or_create_collection("arxiv_papers")
    
    # Load papers
    papers_file = "data/arxiv_papers.json"
    if not os.path.exists(papers_file):
        print("No papers file found. Please run arxiv_loader.py first.")
        return
    
    with open(papers_file, 'r') as f:
        papers = json.load(f)
    
    # Add papers to vector DB
    documents = []
    metadatas = []
    ids = []
    
    for paper in papers[:1000]:  # Limit for demo
        doc_text = f"{paper['title']} {paper['abstract'][:500]}"
        documents.append(doc_text)
        metadatas.append({
            'title': paper['title'],
            'categories': ', '.join(paper['categories']),  # Join categories into a single string
            'published': paper['published'],
            'pdf_url': paper.get('pdf_url', '')
        })
        ids.append(paper['id'])
    
    # Generate embeddings in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        embeddings = embedder.encode(batch_docs).tolist()
        
        collection.add(
            embeddings=embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        
        print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    print(f"Vector DB initialized with {len(documents)} papers")

if __name__ == "__main__":
    initialize_vector_db()