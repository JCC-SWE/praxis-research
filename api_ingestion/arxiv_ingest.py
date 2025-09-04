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

def batch_get_arxiv_to_cosmos(search_query="machine learning", total_limit=5000, batch_size=300,
                              container_name="arxiv_container", delay=3,
                              start_date=None, end_date=None):
    """
    Fetch large batches from arXiv and insert into CosmosDB
    """
    # Initialize CosmosDB connection
    db = CosmosDB(container_name=container_name)
    
    query_encoded = build_arxiv_query(search_query, start_date, end_date)
    total_retrieved = 0
    total_inserted = 0
    
    print(f"🚀 Starting arXiv ingest to {container_name}")
    print(f"Query: {search_query}")
    print(f"Date range: {start_date} to {end_date}")

    for start in range(0, total_limit, batch_size):
        batch = fetch_arxiv_batch(query_encoded, start=start, max_results=batch_size, delay=delay)
        if not batch:
            print("⚠️ No more results returned. Exiting.")
            break

        # Insert batch into CosmosDB
        results = db.insert_batch(batch)
        successful_inserts = len([r for r in results if r is not None])
        
        total_retrieved += len(batch)
        total_inserted += successful_inserts
        
        print(f"📄 Batch: {len(batch)} papers | Inserted: {successful_inserts} | Total: {total_retrieved}")

        if len(batch) < batch_size:
            break  # End of results

    print(f"📦 Finished. Total papers collected: {total_retrieved}")
    print(f"💾 Total papers inserted: {total_inserted}")

if __name__ == "__main__":
    batch_get_arxiv_to_cosmos(
        search_query="machine learning",
        total_limit=1000,
        batch_size=100,
        container_name="s_scholar_container",
        start_date="2025-04-30",
        end_date="2025-07-31"
    )