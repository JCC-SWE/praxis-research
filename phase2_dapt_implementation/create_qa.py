#Call Open AI API to create Q&A pairs from a given text
#Get data from s_semantic_scholar
#Parse data to get abstract
#Create Questions
#Insert questions into gpt
#Save Q&A pairs to s_qa_pairs.csv---columns: paper_id, question, answer, date 
import os
import sys
from datetime import datetime as dt

# Add cosmos_util path and import CosmosDB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
cosmos_util_path = os.path.join(parent_dir, 'azure_resources')
sys.path.insert(0, cosmos_util_path)
from azure_resources.cosmos_util import CosmosDB 
from azure_resources.keyvault_client import get_secrets

def get_papers(limit=10):
    """Get papers from Cosmos DB"""
    # Get secrets first like in your working example
    secrets = get_secrets()
    if not secrets:
        print("❌ No secrets retrieved from Key Vault")
        return []
    
    cosmos = CosmosDB(container_name="s_scholar_container")  # Use your container name
    papers = cosmos.read_limited(limit)
    print(f"Retrieved {len(papers)} papers")
    return papers