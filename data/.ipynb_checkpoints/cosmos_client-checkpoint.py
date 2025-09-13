import sys
import os

# Add cosmos_util path and import CosmosDB
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'azure_resources'))
from cosmos_util import CosmosDB

def get_nlp_ready_data(container_name, text_fields=['title', 'abstract', 'body', 'summary']):
    """
    Retrieve all data from CosmosDB container and format for NLP preprocessing.
    
    Args:
        container_name (str): Name of the CosmosDB container
        text_fields (list): List of fields to extract text from
    
    Returns:
        list: List of dictionaries with id and combined text ready for NLP
    """
    try:
        # Initialize CosmosDB client
        db = CosmosDB(container_name=container_name)
        
        # Get all data from container
        print(f"📊 Retrieving all data from {container_name}...")
        raw_data = db.read_all()
        
        if not raw_data:
            print("⚠️ No data found in container")
            return []
        
        print(f"📄 Processing {len(raw_data)} records for NLP...")
        
        # Format for NLP processing
        nlp_ready_data = []
        
        for record in raw_data:
            # Extract text from specified fields
            text_parts = []
            
            for field in text_fields:
                if field in record and record[field]:
                    text_content = str(record[field]).strip()
                    if text_content and text_content != 'None':
                        text_parts.append(text_content)
            
            # Combine all text fields
            combined_text = ' '.join(text_parts)
            
            # Only include records with actual text content
            if combined_text:
                nlp_record = {
                    'id': record.get('id'),
                    'text': combined_text,
                    'title': record.get('title', ''),
                    'authors': record.get('authors', []),
                    'publication_year': record.get('publication_year') or record.get('year'),
                    'venue': record.get('venue', ''),
                    'source': container_name
                }
                nlp_ready_data.append(nlp_record)
        
        print(f"✅ Formatted {len(nlp_ready_data)} records for NLP processing")
        return nlp_ready_data
        
    except Exception as e:
        print(f"❌ Error retrieving data from {container_name}: {e}")
        return []

def get_combined_nlp_data(container_names=['event_registry', 'arxiv_container', 's_scholar_container']):
    """
    Retrieve and combine data from multiple CosmosDB containers for NLP.
    
    Args:
        container_names (list): List of container names to retrieve from
    
    Returns:
        list: Combined list of NLP-ready records from all containers
    """
    all_data = []
    
    for container in container_names:
        print(f"\n🔄 Processing container: {container}")
        container_data = get_nlp_ready_data(container)
        all_data.extend(container_data)
    
    print(f"\n📊 Total records across all containers: {len(all_data)}")
    return all_data

if __name__ == "__main__":
    # Test with single container
    data = get_nlp_ready_data("s_scholar_container")
    if data:
        print(f"\nSample record:")
        print(f"ID: {data[0]['id']}")
        print(f"Title: {data[0]['title']}")
        print(f"Authors: {len(data[0]['authors'])} authors")
        print(f"Publication Year: {data[0]['publication_year']}")
        print(f"Venue: {data[0]['venue']}")
        print(f"Source: {data[0]['source']}")
        print(f"Text preview: {data[0]['text'][:200]}...")