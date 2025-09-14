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
    def __init__(self, container_name="s_semantic_scholar"):
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
    
    def insert_batch(self, items):
        """Insert a batch of items into existing database"""
        try:
            results = []
            for item in items:
                try:
                    created_item = self.container.create_item(body=item)
                    results.append(created_item)
                except Exception as e:
                    print(f"Failed to insert item {item.get('id', 'unknown')}: {e}")
                    results.append(None)
            
            successful = [r for r in results if r is not None]
            print(f"Successfully inserted {len(successful)} out of {len(items)} items")
            return results
            
        except Exception as e:
            print(f"Error in batch insert: {e}")
            return []
    
    def read_batch(self, item_ids):
        """Read a batch of items by their IDs"""
        try:
            results = []
            for item_id in item_ids:
                try:
                    item = self.container.read_item(item=item_id, partition_key=item_id)
                    results.append(item)
                except CosmosResourceNotFoundError:
                    print(f"Item '{item_id}' not found")
                    results.append(None)
                except Exception as e:
                    print(f"Error reading item {item_id}: {e}")
                    results.append(None)
            
            successful = [r for r in results if r is not None]
            print(f"Successfully read {len(successful)} out of {len(item_ids)} items")
            return results
            
        except Exception as e:
            print(f"Error in batch read: {e}")
            return []
    
    def read_all(self):
        """Read all objects from the container"""
        try:
            items = list(self.container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True
            ))
            print(f"Read {len(items)} items")
            return items
            
        except Exception as e:
            print(f"Error reading all items: {e}")
            return []
    
    def read_limited(self, limit=100, offset=0):
        """Read a limited number of objects from the container
        
        Args:
            limit (int): Maximum number of records to return (default: 100)
            offset (int): Number of records to skip (default: 0)
        
        Returns:
            list: List of items from the container
        """
        try:
            query = f"SELECT * FROM c OFFSET {offset} LIMIT {limit}"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            print(f"Read {len(items)} items (offset: {offset}, limit: {limit})")
            return items
            
        except Exception as e:
            print(f"Error reading limited items: {e}")
            return []

    def update_batch(self, updates):
        """Update a batch of items. updates should be list of dicts with 'id' and update fields"""
        try:
            results = []
            for update_data in updates:
                item_id = update_data.get('id')
                if not item_id:
                    print("Update data missing 'id' field")
                    results.append(None)
                    continue
                
                try:
                    # Read existing item
                    existing_item = self.container.read_item(item=item_id, partition_key=item_id)
                    # Apply updates
                    existing_item.update(update_data)
                    # Replace the item
                    updated_item = self.container.replace_item(item=item_id, body=existing_item)
                    results.append(updated_item)
                except Exception as e:
                    print(f"Failed to update item {item_id}: {e}")
                    results.append(None)
            
            successful = [r for r in results if r is not None]
            print(f"Successfully updated {len(successful)} out of {len(updates)} items")
            return results
            
        except Exception as e:
            print(f"Error in batch update: {e}")
            return []
    
    def delete_batch(self, item_ids):
        """Delete a batch of items by their IDs"""
        try:
            results = []
            for item_id in item_ids:
                try:
                    self.container.delete_item(item=item_id, partition_key=item_id)
                    results.append(True)
                    print(f"Deleted item: {item_id}")
                except CosmosResourceNotFoundError:
                    print(f"Item '{item_id}' not found")
                    results.append(False)
                except Exception as e:
                    print(f"Error deleting item {item_id}: {e}")
                    results.append(False)
            
            successful = sum(results)
            print(f"Successfully deleted {successful} out of {len(item_ids)} items")
            return results
            
        except Exception as e:
            print(f"Error in batch delete: {e}")
            return []

if __name__ == "__main__":
    try:
        # Verify connection and list containers
        db = CosmosDB()
        print("✓ Connection successful")
        
        # List all containers in the database
        containers = list(db.database.list_containers())
        print(f"✓ Found {len(containers)} containers:")
        for container in containers:
            print(f"  - {container['id']}")
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")