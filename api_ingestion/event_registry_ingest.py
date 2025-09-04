import subprocess
import sys
import time
import os
import hashlib
from datetime import datetime as dt

# Install eventregistry if not available
try:
    from eventregistry import *
except ImportError:
    print("📦 Installing eventregistry...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "eventregistry"])
    from eventregistry import *
    print("✅ eventregistry installed successfully")

# Add cosmos_util path and import CosmosDB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cosmos_util_path = os.path.join(parent_dir, 'azure_resources')
sys.path.insert(0, cosmos_util_path)
from cosmos_util import CosmosDB
from keyvault_client import get_secrets

def title_to_hash(title):
    """Convert title to SHA256 hash for use as ID"""
    return hashlib.sha256(title.encode('utf-8')).hexdigest()

def get_api_key():
    """Retrieve Event Registry API key from secrets"""
    try:
        secrets = get_secrets()
        api_key = secrets.get('event-registry')
        if not api_key:
            print("❌ 'event-registry' key not found in secrets")
            return None
        return api_key
    except Exception as e:
        print(f"❌ Error retrieving API key: {e}")
        return None

def fetch_eventregistry_to_cosmos(keywords=None, max_items=100, date_range=None, 
                                 container_name="event_registry", delay_seconds=1):
    """
    Fetch articles from Event Registry using official package and insert into CosmosDB.
    
    Args:
        keywords (list): List of keywords to search for
        max_items (int): Maximum number of articles to fetch
        date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format
        container_name (str): CosmosDB container name
        delay_seconds (float): Delay before API call
    """
    # Get API key and initialize Event Registry
    api_key = get_api_key()
    if not api_key:
        return False
    
    er = EventRegistry(api_key)
    
    # Initialize CosmosDB connection
    db = CosmosDB(container_name=container_name)
    
    # Default keywords if none provided
    if keywords is None:
        keywords = ["Artificial Intelligence", "Generative AI", "Machine Learning", "Deep Learning"]
    
    print(f"🔍 Fetching {max_items} articles from Event Registry")
    print(f"💾 Target container: {container_name}")
    print(f"🔎 Keywords: {', '.join(keywords)}")
    
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    
    try:
        # Build query using Event Registry package
        query_params = {
            'keywords': QueryItems.OR(keywords),
            'dataType': ["news", "blog"],
            'lang': ["eng"]
        }
        
        # Add date range if provided
        if date_range:
            query_params['dateStart'] = date_range[0]
            query_params['dateEnd'] = date_range[1]
        
        q = QueryArticlesIter(**query_params)
        
        # Fetch articles
        articles = []
        for art in q.execQuery(er, sortBy="date", maxItems=max_items):
            articles.append(art)
        
        print(f"📄 Retrieved {len(articles)} articles")
        
        if not articles:
            print("⚠️ No articles found")
            return False
        
        # Convert to CosmosDB format
        cosmos_articles = []
        for art in articles:
            title = art.get('title', '').strip()
            if not title:
                continue
            
            record = {
                'id': title_to_hash(title),
                'event_registry_uri': art.get('uri'),
                'title': title,
                'body': art.get('body', ''),
                'summary': art.get('summary', ''),
                'url': art.get('url'),
                'source': art.get('source', {}).get('title', 'Unknown') if art.get('source') else 'Unknown',
                'authors': art.get('authors', []),
                'date': art.get('date'),
                'datetime': art.get('dateTime'),
                'language': art.get('lang'),
                'country': art.get('location', {}).get('country', {}).get('label') if art.get('location') else None,
                'sentiment': art.get('sentiment'),
                'duplicate_group_id': art.get('duplicateGroupId'),
                'keywords': keywords,
                'ingested_at': dt.utcnow().isoformat()
            }
            cosmos_articles.append(record)
        
        # Insert batch into CosmosDB
        print(f"💾 Inserting {len(cosmos_articles)} articles into {container_name}")
        results = db.insert_batch(cosmos_articles)
        successful_inserts = len([r for r in results if r is not None])
        
        print(f"✅ Successfully inserted {successful_inserts} out of {len(cosmos_articles)} articles")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fetch_eventregistry_to_cosmos(
        keywords=["artificial intelligence", "machine learning", "deep learning"],
        max_items=500,
        date_range=('2025-01-01', '2025-07-31'),
        container_name="s_scholar_container"
    )