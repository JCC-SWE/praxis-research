"""
download_from_blob.py - Simple blob download functions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'azure_resources'))

from keyvault_client import get_secrets
from azure.storage.blob import BlobServiceClient

def get_blob_client():
    """Get blob client"""
    secrets = get_secrets()
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client, secrets['container']

def list_blobs():
    """List all blobs and return as dictionary"""
    try:
        blob_service_client, container_name = get_blob_client()
        container_client = blob_service_client.get_container_client(container_name)
        
        blobs = {}
        for blob in container_client.list_blobs():
            blobs[blob.name] = {
                'size': blob.size,
                'last_modified': blob.last_modified
            }
        
        return blobs
    except Exception as e:
        print(f"Error: {e}")
        return {}

def download_blob(blob_name):
    """Download a specific blob"""
    try:
        blob_service_client, container_name = get_blob_client()
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        content = blob_client.download_blob().readall().decode('utf-8')
        return content
    except Exception as e:
        print(f"Error downloading {blob_name}: {e}")
        return ""

def download_all_blobs():
    """Download all blobs and return as dictionary"""
    blobs = list_blobs()
    content = {}
    
    for blob_name in blobs.keys():
        print(f"Downloading {blob_name}...")
        content[blob_name] = download_blob(blob_name)
    
    return content

def get_all_blob_contents():
    """Get all blob contents for training"""
    return download_all_blobs()

if __name__ == "__main__":
    # Test
    contents = get_all_blob_contents()
    print(f"Got {len(contents)} blobs with content")