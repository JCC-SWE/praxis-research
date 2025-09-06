import urllib.request
import urllib.parse
import json
import time
import sys
import os
import hashlib
from datetime import datetime as dt

# Add cosmos_util path and import CosmosDB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cosmos_util_path = os.path.join(parent_dir, 'azure_resources')
sys.path.insert(0, cosmos_util_path)
from cosmos_util import CosmosDB 
from keyvault_client import get_secrets

# High-Level AI Topics for Event Registry (15 max)
AI_TOPICS = [
    "large language models",
    "neural networks", 
    "deep learning",
    "computer vision",
    "natural language processing",
    "reinforcement learning",
    "AI safety",
    "AI alignment",
    "transformer models",
    "AI ethics",
    "robotics AI",
    "AI drug discovery",
    "federated learning",
    "multimodal AI",
    "AI benchmarking"
]

def title_to_hash(title):
    """Convert title to SHA256 hash for use as ID"""
    return hashlib.sha256(title.encode('utf-8')).hexdigest()

def fetch_semanticscholar_batch_to_cosmos_multi_topic(topics=None, papers_per_topic=100, 
                                                     year_range=None, container_name="s_scholar_container", 
                                                     delay_seconds=2):
    """
    Fetch papers from Semantic Scholar for multiple topics using API key and insert into CosmosDB.
    
    Args:
        topics (list): List of search topics/queries (defaults to AI_TOPICS)
        papers_per_topic (int): Papers per topic (max 1000)
        year_range (tuple): (start, end) e.g. ('01-2020', '12-2024') or (2020, 2024)
        container_name (str): CosmosDB container name
        delay_seconds (float): Delay between topic queries
    """
    if topics is None:
        topics = AI_TOPICS
    
    # Get API key from secrets
    secrets = get_secrets()
    api_key = secrets.get('s-scholar-key')  # Assuming this is the Semantic Scholar key

    if not api_key:
        print("❌ No API key found in secrets")
        return False
    
    # Initialize CosmosDB connection
    db = CosmosDB(container_name=container_name)
    
    print(f"🚀 Starting multi-topic Semantic Scholar ingest to {container_name}")
    print(f"Topics: {len(topics)} topics")
    print(f"Papers per topic: {papers_per_topic}")
    print(f"Year range: {year_range}")
    print(f"Expected total papers: ~{len(topics) * papers_per_topic}")
    
    total_retrieved = 0
    total_inserted = 0
    
    for topic_idx, topic in enumerate(topics, 1):
        print(f"\n📚 Processing topic {topic_idx}/{len(topics)}: '{topic}'")
        
        try:
            time.sleep(delay_seconds)
            
            # Use bulk search endpoint with API key
            base_url = 'https://api.semanticscholar.org/graph/v1/paper/search/bulk'
            params = {
                'query': topic,
                'offset': 0,
                'limit': min(papers_per_topic, 1000),  # Max 1000 per call
                'fields': 'title,abstract,authors,year,url,venue,externalIds,citationCount,paperId'
            }
            
            # Add year filter if provided
            if year_range:
                if isinstance(year_range[0], str) and '-' in year_range[0]:
                    # Extract years from MM-YYYY format
                    start_year = year_range[0].split('-')[1]
                    end_year = year_range[1].split('-')[1]
                    params['year'] = f"{start_year}-{end_year}"
                else:
                    params['year'] = f"{year_range[0]}-{year_range[1]}"
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            headers = {
                'User-Agent': 'Python-Research-Agent/1.0',
                'x-api-key': api_key
            }
            
            request = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(request, timeout=30)
            json_data = json.loads(response.read().decode('utf-8'))
            
            papers = json_data.get('data', [])
            topic_papers = []
            
            # Convert to CosmosDB format
            for paper in papers:
                title = paper.get('title', '').strip()
                if not title:
                    continue
                    
                authors = [author.get('name') for author in paper.get('authors', []) if author.get('name')]
                
                record = {
                    'id': title_to_hash(title),
                    'paper_id': paper.get('paperId'),
                    'title': title,
                    'abstract': paper.get('abstract', ''),
                    'authors': authors,
                    'year': paper.get('year'),
                    'venue': paper.get('venue'),
                    'citation_count': paper.get('citationCount', 0),
                    'url': paper.get('url'),
                    'external_ids': paper.get('externalIds', {}),
                    'doi': paper.get('externalIds', {}).get('DOI'),
                    'topic': topic,
                    'ingested_at': dt.utcnow().isoformat()
                }
                topic_papers.append(record)
            
            # Insert batch into CosmosDB
            if topic_papers:
                results = db.insert_batch(topic_papers)
                successful_inserts = len([r for r in results if r is not None])
                
                total_retrieved += len(topic_papers)
                total_inserted += successful_inserts
                
                print(f"  📄 Found: {len(topic_papers)} papers | Inserted: {successful_inserts}")
            else:
                print(f"  ⚠️ No papers found for '{topic}'")
                
        except Exception as e:
            print(f"  ❌ Error fetching '{topic}': {e}")
    
    print(f"\n📦 FINAL SUMMARY:")
    print(f"📚 Topics processed: {len(topics)}")
    print(f"📄 Total papers retrieved: {total_retrieved}")
    print(f"💾 Total papers inserted: {total_inserted}")
    
    return total_inserted > 0

if __name__ == "__main__":
    fetch_semanticscholar_batch_to_cosmos_multi_topic(
        topics=AI_TOPICS,  # Use all AI topics, or pass custom list
        papers_per_topic=20000,  # Fetch up to 400 papers per topic
        year_range=('01-2025', '09-2025'),
        container_name="s_scholar_container"
    )