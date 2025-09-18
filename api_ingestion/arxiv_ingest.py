import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import sys
import os
import hashlib
from datetime import datetime

# Add cosmos_util path and import CosmosDB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cosmos_util_path = os.path.join(parent_dir, 'azure_resources')
sys.path.insert(0, cosmos_util_path)
from cosmos_util import CosmosDB
def load_topics_from_file(file_path="ai_topics.txt"):
    """Load topics from a text file, one topic per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
        return topics
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default topics.")
        return ['machine learning', 'artificial intelligence', 'deep learning']
    except Exception as e:
        print(f"Error reading {file_path}: {e}. Using default topics.")
        return ['machine learning', 'artificial intelligence', 'deep learning']
topics = load_topics_from_file()
# Comprehensive AI-related topics
AI_TOPICS = [
    # AI Hardware & Architecture
    "tensor processing units",
    "GPU computing",
    "FPGA machine learning",
    "AI chips",
    "parallel computing",
    "distributed computing",
    "cloud AI",
    
    # Advanced AI Methods
    "ensemble learning",
    "boosting algorithms",
    "kernel methods",
    "manifold learning",
    "sparse coding",
    "dictionary learning",
    "compressed sensing",
    "matrix factorization",
    "topic modeling",
    "latent variable models",
    
    # Time Series & Sequential Data
    "time series forecasting",
    "sequence modeling",
    "temporal data mining",
    "anomaly detection",
    "change point detection",
    "streaming algorithms",
    "online learning",
    
    # AI for Science & Engineering
    "scientific machine learning",
    "physics informed neural networks",
    "computational fluid dynamics AI",
    "AI for drug discovery",
    "molecular machine learning",
    "protein folding",
    "AI for astronomy",
    "geospatial AI",
    "remote sensing",
    
    # Human-AI Interaction
    "human-AI collaboration",
    "AI usability",
    "cognitive computing",
    "affective computing",
    "emotion recognition",
    "social AI",
    "conversational AI",
    "AI personalization",
    
    # AI Theory & Foundations
    "computational learning theory",
    "information theory",
    "statistical learning theory",
    "approximation theory",
    "complexity theory",
    "algorithmic information theory"
]

def build_arxiv_query(topic, start_date=None, end_date=None):
    """
    Build arXiv Lucene-style search query with optional submittedDate range
    """
    topic_encoded = urllib.parse.quote_plus(topic)
    if start_date and end_date:
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")
        date_filter = f"submittedDate:[{start_fmt}+TO+{end_fmt}]"
        return f"({date_filter})+AND+all:{topic_encoded}"
    else:
        return f"all:{topic_encoded}"

def title_to_hash(title):
    """Convert title to SHA256 hash for use as ID"""
    return hashlib.sha256(title.encode('utf-8')).hexdigest()

def fetch_arxiv_batch(query_encoded, start, max_results=300, delay=3):
    """
    Fetch a single arXiv API batch using a fully formed encoded query string
    """
    url = f"http://export.arxiv.org/api/query?search_query={query_encoded}&start={start}&max_results={max_results}"
    print(f"📡 Fetching: {url}")
    time.sleep(delay)

    response = urllib.request.urlopen(url)
    xml_data = response.read()
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    records = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        record = {
            'id': title_to_hash(title),  # Use hash of title as ID
            'arxiv_id': entry.find('atom:id', ns).text,
            'title': title,
            'abstract': entry.find('atom:summary', ns).text.strip(),
            'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
            'published': entry.find('atom:published', ns).text,
            'updated': entry.find('atom:updated', ns).text,
            'primary_category': entry.find('atom:category', ns).attrib.get('term', None),
            'link': next((link.attrib['href'] for link in entry.findall('atom:link', ns)
                          if link.attrib.get('type') == 'text/html'), None),
            'ingested_at': datetime.utcnow().isoformat()
        }
        records.append(record)

    return records

def batch_get_arxiv_to_cosmos_multi_topic(topics=None, papers_per_topic=500, batch_size=100,
                                          container_name="arxiv_container", delay=3,
                                          start_date=None, end_date=None):
    """
    Fetch papers for multiple topics from arXiv and insert into CosmosDB
    """
    if topics is None:
        topics = AI_TOPICS
    
    # Initialize CosmosDB connection
    db = CosmosDB(container_name=container_name)
    
    total_retrieved = 0
    total_inserted = 0
    
    print(f"🚀 Starting multi-topic arXiv ingest to {container_name}")
    print(f"Topics: {len(topics)} topics")
    print(f"Papers per topic: {papers_per_topic}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Expected total papers: ~{len(topics) * papers_per_topic}")

    for topic_idx, topic in enumerate(topics, 1):
        print(f"\n📚 Processing topic {topic_idx}/{len(topics)}: '{topic}'")
        
        query_encoded = build_arxiv_query(topic, start_date, end_date)
        topic_retrieved = 0
        topic_inserted = 0
        
        for start in range(0, papers_per_topic, batch_size):
            batch = fetch_arxiv_batch(query_encoded, start=start, max_results=batch_size, delay=delay)
            if not batch:
                print(f"⚠️ No more results for '{topic}'. Moving to next topic.")
                break

            # Insert batch into CosmosDB
            results = db.insert_batch(batch)
            successful_inserts = len([r for r in results if r is not None])
            
            topic_retrieved += len(batch)
            topic_inserted += successful_inserts
            total_retrieved += len(batch)
            total_inserted += successful_inserts
            
            print(f"  📄 Batch: {len(batch)} papers | Inserted: {successful_inserts} | Topic total: {topic_retrieved}")

            if len(batch) < batch_size:
                break  # End of results for this topic
        
        print(f"✅ Completed '{topic}': {topic_retrieved} papers retrieved, {topic_inserted} inserted")

    print(f"\n📦 FINAL SUMMARY:")
    print(f"📚 Topics processed: {len(topics)}")
    print(f"📄 Total papers retrieved: {total_retrieved}")
    print(f"💾 Total papers inserted: {total_inserted}")

if __name__ == "__main__":
    batch_get_arxiv_to_cosmos_multi_topic(
        topics=topics,  # Use all AI topics, or pass custom list
        papers_per_topic=400,  # Fetch up to 400 papers per topic
        batch_size=100,
        container_name="baseline-papers-23",
        start_date="2023-01-01",
        end_date="2023-09-30"
    )
    