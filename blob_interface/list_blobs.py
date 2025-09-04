"""
test_blob.py - Simple blob connection test
"""

import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'azure_resources'))

from keyvault_client import get_secrets
from azure.storage.blob import BlobServiceClient

def get_blobs():
    print("Getting secrets...")
    secrets = get_secrets()
    
    if not secrets:
        print("❌ Failed to get secrets")
        return
    
    # Build connection string
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
    
    try:
        # Connect to blob storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(secrets['container'])
        
        # List blobs
        blobs = list(container_client.list_blobs())
        blob_names = []
        print(f"✅ Found {len(blobs)} blobs:")
        for i, blob in enumerate(blobs, 1):
            blob_names.append(blob.name)
        return blob_names
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_blobs()