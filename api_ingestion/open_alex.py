import requests
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

def title_to_hash(title):
    """Convert title to SHA256 hash for use as ID"""
    return hashlib.sha256(title.encode('utf-8')).hexdigest()

def fetch_openalex_batch_to_cosmos(search_terms=None, total_papers=100, batch_size=200, 
                                   year_range=None, container_name="s_scholar_container", delay_seconds=1):
    """
    Fetch papers from OpenAlex API and insert into CosmosDB.
    
    Args:
        search_terms (list): List of search terms (default: common AI terms)
        total_papers (int): Total number of papers to fetch
        batch_size (int): Papers per API call (max 200)
        year_range (tuple): (start, end) e.g. ('01-2020', '12-2024') or (2020, 2024)
        container_name (str): CosmosDB container name
        delay_seconds (float): Delay between API calls
    """
    # Default AI search terms
    if search_terms is None:
        search_terms = [
            "artificial intelligence",
            "machine learning", 
            "deep learning",
            "neural networks",
            "natural language processing"
        ]
    
    # Initialize CosmosDB connection
    db = CosmosDB(container_name=container_name)
    
    print(f"🔍 Fetching {total_papers} papers from OpenAlex")
    print(f"💾 Target container: {container_name}")
    print(f"🔎 Search terms: {', '.join(search_terms)}")
    
    # Create search query
    search_query = " OR ".join([f'"{term}"' for term in search_terms])
    
    all_papers = []
    papers_fetched = 0
    page = 1
    
    while papers_fetched < total_papers:
        remaining = total_papers - papers_fetched
        current_batch_size = min(remaining, batch_size, 200)  # OpenAlex limit is 200
        
        print(f"  📄 Fetching batch {page}: {current_batch_size} papers")
        
        try:
            time.sleep(delay_seconds)
            
            # OpenAlex API call
            base_url = "https://api.openalex.org/works"
            
            # Build filter with search query
            filters = [f'title_and_abstract.search:{search_query}', 'type:article']
            
            # Add date filter if provided
            if year_range:
                if isinstance(year_range[0], str) and '-' in year_range[0]:
                    # Convert MM-YYYY to YYYY-MM-DD format for OpenAlex
                    start_month, start_year = year_range[0].split('-')
                    end_month, end_year = year_range[1].split('-')
                    # Ensure proper zero-padding for months
                    start_month = start_month.zfill(2)
                    end_month = end_month.zfill(2)
                    # Use last day of end month
                    import calendar
                    last_day = calendar.monthrange(int(end_year), int(end_month))[1]
                    start_date = f"{start_year}-{start_month}-01"
                    end_date = f"{end_year}-{end_month}-{last_day:02d}"
                    filters.append(f'from_publication_date:{start_date}')
                    filters.append(f'to_publication_date:{end_date}')
                else:
                    # Simple year range
                    filters.append(f'from_publication_date:{year_range[0]}-01-01')
                    filters.append(f'to_publication_date:{year_range[1]}-12-31')
            
            params = {
                'filter': ','.join(filters),
                'sort': 'cited_by_count:desc',
                'per-page': current_batch_size,
                'page': page,
                'mailto': 'jnatc1@gmail.com'  # For polite pool access
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            works = data.get('results', [])
            
            if not works:
                print("    ⚠️ No more papers available")
                break
            
            # Convert to CosmosDB format
            batch_papers = []
            for work in works:
                title = work.get('title', '').strip()
                if not title:
                    continue
                
                # Extract authors
                authors = []
                if work.get('authorships'):
                    for authorship in work['authorships']:
                        if authorship.get('author') and authorship['author'].get('display_name'):
                            authors.append(authorship['author']['display_name'])
                
                # Reconstruct abstract from inverted index
                abstract_text = "Abstract not available"
                abstract = work.get('abstract_inverted_index')
                if abstract:
                    word_positions = []
                    for word, positions in abstract.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort(key=lambda x: x[0])
                    abstract_text = ' '.join([word for _, word in word_positions])
                
                # Get venue
                venue = "Unknown"
                if work.get('primary_location') and work['primary_location'].get('source'):
                    venue = work['primary_location']['source'].get('display_name', 'Unknown')
                
                record = {
                    'id': title_to_hash(title),
                    'openalex_id': work.get('id'),
                    'title': title,
                    'abstract': abstract_text,
                    'authors': authors,
                    'publication_date': work.get('publication_date'),
                    'publication_year': work.get('publication_year'),
                    'venue': venue,
                    'citation_count': work.get('cited_by_count', 0),
                    'doi': work.get('doi'),
                    'open_access': work.get('open_access', {}).get('is_oa', False),
                    'pdf_url': work.get('open_access', {}).get('oa_url'),
                    'source': 'openalex',
                    'search_terms': search_terms,
                    'ingested_at': dt.utcnow().isoformat()
                }
                batch_papers.append(record)
            
            all_papers.extend(batch_papers)
            papers_fetched += len(batch_papers)
            
            print(f"    ✅ Found {len(batch_papers)} papers (Total: {papers_fetched})")
            
            # If we got fewer papers than requested, we've hit the end
            if len(works) < current_batch_size:
                break
                
            page += 1
            
        except Exception as e:
            print(f"    ❌ Error fetching batch {page}: {e}")
            break
    
    if not all_papers:
        print("⚠️ No papers found")
        return False
    
    # Insert batch into CosmosDB
    print(f"💾 Inserting {len(all_papers)} papers into {container_name}")
    results = db.insert_batch(all_papers)
    successful_inserts = len([r for r in results if r is not None])
    
    print(f"✅ Successfully inserted {successful_inserts} out of {len(all_papers)} papers")
    return True

if __name__ == "__main__":
    fetch_openalex_batch_to_cosmos(
        search_terms=[# Next-Gen AI Architectures
    "multimodal AI",
    "cross-modal learning",
    "vision-language models",
    "AI foundation models",
    "AI model scaling",
    "mixture of experts",
    "AI retrieval augmented generation",
    "AI in-context learning",
    "AI prompt engineering",
    "AI instruction tuning",
    
    # AI Reasoning & Planning
    "AI logical reasoning",
    "AI causal inference",
    "AI symbolic reasoning",
    "AI planning algorithms",
    "AI decision making",
    "AI problem solving",
    "AI common sense reasoning",
    "AI spatial reasoning",
    "AI temporal reasoning",
    
    # Advanced AI Training
    "AI curriculum learning",
    "AI multi-task learning",
    "AI domain adaptation",
    "AI lifelong learning",
    "AI catastrophic forgetting",
    "AI neural plasticity",
    "AI replay methods",
    "AI memory networks",
    
    # AI Evaluation & Benchmarking
    "AI evaluation metrics",
    "AI benchmark datasets",
    "AI performance analysis",
    "AI model comparison",
    "AI stress testing",
    "AI robustness evaluation",
    "AI generalization assessment",
    
    # Specialized AI Applications
    "AI creative writing",
    "AI code generation",
    "AI mathematical reasoning",
    "AI scientific discovery",
    "AI theorem proving",
    "AI data synthesis",
    "AI simulation",
    "AI virtual environments",
    
    # AI Infrastructure & Deployment
    "AI model optimization",
    "AI quantization",
    "AI pruning",
    "AI knowledge distillation",
    "AI edge deployment",
    "AI mobile AI",
    "AI real-time inference",
    "AI latency optimization",
    
    # Human-Centric AI
    "AI accessibility",
    "AI inclusive design",
    "AI cultural sensitivity",
    "AI bias mitigation",
    "AI fairness metrics",
    "AI human feedback",
    "AI user experience",
    "AI trust and reliability",
    
    # Future AI Concepts
    "artificial general intelligence",
    "AI consciousness",
    "AI self-improvement",
    "AI recursive improvement",
    "AI goal alignment",
    "AI value learning",
    "AI cooperative AI",
    "AI multi-agent coordination"
],
        total_papers=50000,
        batch_size=200,
        year_range=('01-2025', '09-2025'),  # MM-YYYY format
        container_name="s_scholar_container"
    )