#Call Open AI API to create Q&A pairs from a given text
#Get data from s_semantic_scholar
#Parse data to get abstract
#Create Questions
#Insert questions into gpt
#Save Q&A pairs to s_qa_pairs.csv---columns: paper_id, question, answer, date 
import os
import sys
from datetime import datetime as dt

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
azure_path = os.path.join(parent_dir, 'azure_resources')
blob_path = os.path.join(parent_dir, 'blob_interface')
sys.path.insert(0, azure_path)
sys.path.insert(0, blob_path)

# Import everything at the top
from cosmos_util import CosmosDB 
from keyvault_client import get_secrets
from upload_to_blob import upload_to_blob
from download_from_blob import download_blob

def get_papers(limit=10):
    """Get papers from Cosmos DB"""
    # Get secrets first like in your working example
    secrets = get_secrets()
    if not secrets:
        print("❌ No secrets retrieved from Key Vault")
        return []
    
    cosmos = CosmosDB(container_name="s_scholar_container")  # Use your container name
    papers = cosmos.read_limited(limit)
    print(papers)
    return papers

def extract_abstracts(papers):
    """Extract abstracts from paper documents"""
    abstracts_data = []
    
    for paper in papers:
        abstract = paper.get('abstract', '')
        if abstract:  # Only include papers with abstracts
            abstracts_data.append({
                'paper_id': paper.get('paper_id'),
                'id': paper.get('id'),
                'title': paper.get('title'),
                'abstract': abstract
            })
    
    print(f"Extracted {len(abstracts_data)} papers with abstracts")
    return abstracts_data

def save_abstracts_to_blob(abstracts_data):
    """Save abstracts to blob storage as abstracts.txt"""
        
    # Format abstracts as text
    content_lines = []
    for item in abstracts_data:
        content_lines.append(f"Paper ID: {item['paper_id']}")
        content_lines.append(f"Title: {item['title']}")
        content_lines.append(f"Abstract: {item['abstract']}")
        content_lines.append("-" * 80)  # Separator
        content_lines.append("")  # Empty line
    
    content = "\n".join(content_lines)
    
    # Upload to blob (overwrites if exists)
    success = upload_to_blob(content, "abstracts.txt")
    
    if success:
        print(f"Saved {len(abstracts_data)} abstracts to abstracts.txt")
    else:
        print("Failed to save abstracts to blob")
    
    return success

def get_abstracts_from_blob():
    """Download abstracts.txt from blob storage for GPT processing"""
        
    content = download_blob("abstracts.txt")
    
    if content:
        print(f"Downloaded abstracts.txt ({len(content)} characters)")
        return content
    else:
        print("Failed to download abstracts.txt or file is empty")
        return ""
    
def process_papers_to_blob(limit=10):
    """
    Get papers from Cosmos DB, extract abstracts, and save to blob storage
    
    Args:
        limit (int): Number of papers to retrieve from Cosmos DB
        
    Returns:
        bool: Success status of the entire process
    """
    try:
        # Step 1: Get papers from Cosmos DB
        papers = get_papers(limit)
        if not papers:
            print("No papers retrieved, stopping process")
            return False
        # Step 2: Extract abstracts
        abstracts = extract_abstracts(papers)
        if not abstracts:
            print("No abstracts found, stopping process")
            return False
        
        # Step 3: Save to blob storage
        success = save_abstracts_to_blob(abstracts)
        
        if success:
            print(f"Successfully processed {len(abstracts)} papers to blob storage")
        
        return success
        
    except Exception as e:
        print(f"Error in process_papers_to_blob: {e}")
        return False