import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time
from utils.helpers import load_config

def fetch_arxiv_papers(categories: List[str] = None, max_results: int = 1000) -> List[Dict]:
    """Fetch papers from Arxiv API"""
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL"]
    
    papers = []
    for category in categories:
        query = f"cat:{category}"
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'id': entry.find('atom:id', ns).text.split('/')[-1],
                    'title': entry.find('atom:title', ns).text.strip(),
                    'abstract': entry.find('atom:summary', ns).text.strip(),
                    'categories': [cat.get('term') for cat in entry.findall('atom:category', ns)],
                    'published': entry.find('atom:published', ns).text,
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    'pdf_url': None
                }
                
                # Find PDF link
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        paper['pdf_url'] = link.get('href')
                
                papers.append(paper)
            
            time.sleep(1)  # Be nice to Arxiv API
            
        except Exception as e:
            print(f"Error fetching category {category}: {e}")
    
    return papers

def save_papers_to_json(papers: List[Dict], filename: str):
    """Save papers to JSON file"""
    import json
    with open(filename, 'w') as f:
        json.dump(papers, f, indent=2)

if __name__ == "__main__":
    papers = fetch_arxiv_papers()
    save_papers_to_json(papers, "data/arxiv_papers.json")
    print(f"Fetched {len(papers)} papers")