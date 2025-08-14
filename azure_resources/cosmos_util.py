import os
import json
from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosResourceNotFoundError

# Load environment variables from .env file in root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

# Get VAULT_URL from environment variables
vault_url = os.getenv('VAULT_URL')

if not vault_url:
    raise ValueError("VAULT_URL not found in environment variables")

def get_secrets():
    """Retrieve cosmos secrets from Azure Key Vault"""
    try:
        print("Getting credentials...")
        credential = DefaultAzureCredential()
        print("Creating Key Vault client...")
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        print("Retrieving secrets...")
        secrets = {
            'cosmos-uri': client.get_secret('cosmos-uri').value,
            'cosmos-key': client.get_secret('cosmos-key').value
        }
        return secrets
        
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        return {}

class CosmosDB:
    def __init__(self, container_name="event_registry"):
        """Initialize Cosmos DB client"""
        secrets = get_secrets()
        
        if not secrets.get('cosmos-uri') or not secrets.get('cosmos-key'):
            raise ValueError("Cosmos DB credentials not found in Key Vault")
        
        # Initialize Cosmos client
        self.client = CosmosClient(
            url=secrets['cosmos-uri'],
            credential=secrets['cosmos-key']
        )
        
        # Connect to existing database and container
        self.database = self.client.get_database_client("cosmos-database")
        self.container = self.database.get_container_client(container_name)
    
    def read_item(self, item_id, partition_key=None):
        """Read an item by its ID"""
        try:
            if partition_key is None:
                partition_key = item_id
            
            item = self.container.read_item(item=item_id, partition_key=partition_key)
            print(f"Found item: {item_id}")
            return item
            
        except CosmosResourceNotFoundError:
            print(f"Item '{item_id}' not found")
            return None
        except Exception as e:
            print(f"Error reading item: {e}")
            return None
    
    def update_item(self, item_id, updates, partition_key=None):
        """Update an existing item"""
        try:
            if partition_key is None:
                partition_key = item_id
                
            # Read existing item
            existing_item = self.read_item(item_id, partition_key)
            if not existing_item:
                return None
            
            # Apply updates
            existing_item.update(updates)
            
            # Replace the item
            updated_item = self.container.replace_item(
                item=item_id,
                body=existing_item
            )
            print(f"Updated item: {item_id}")
            return updated_item
            
        except Exception as e:
            print(f"Error updating item: {e}")
            return None
    
    def delete_item(self, item_id, partition_key=None):
        """Delete an item by its ID"""
        try:
            if partition_key is None:
                partition_key = item_id
                
            self.container.delete_item(item=item_id, partition_key=partition_key)
            print(f"Deleted item: {item_id}")
            return True
            
        except CosmosResourceNotFoundError:
            print(f"Item '{item_id}' not found")
            return False
        except Exception as e:
            print(f"Error deleting item: {e}")
            return False
        
if __name__ == "__main__":
    db = CosmosDB()
    item = db.read_item("your-item-id")
    if item:
        print(json.dumps(item, indent=2))