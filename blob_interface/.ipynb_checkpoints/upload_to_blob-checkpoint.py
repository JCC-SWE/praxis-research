"""
upload_to_blob.py - Simple blob upload function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'azure_resources'))

from keyvault_client import get_secrets
from azure.storage.blob import BlobServiceClient

def upload_to_blob(content, blob_filename):
    """
    Upload text content to blob storage
    
    Args:
        content (str): Text content to upload
        blob_filename (str): Name of the blob file
        
    Returns:
        bool: Success status
    """
    try:
        # Get secrets and connect
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=secrets['container'], blob=blob_filename)
        
        # Upload content
        blob_client.upload_blob(content, overwrite=True)
        print(f"✅ Uploaded {blob_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {blob_filename}: {e}")
        return False

def append_to_blob(content, blob_filename):
    """
    Append content to existing blob file
    
    Args:
        content (str): Text content to append
        blob_filename (str): Name of the blob file
        
    Returns:
        bool: Success status
    """
    try:
        # Get secrets and connect
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=secrets['container'], blob=blob_filename)
        
        # Download existing content
        try:
            existing_content = blob_client.download_blob().readall().decode('utf-8')
        except:
            existing_content = ""
        
        # Append new content
        updated_content = existing_content + "\n" + content
        
        # Upload combined content
        blob_client.upload_blob(updated_content, overwrite=True)
        print(f"✅ Appended to {blob_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error appending to {blob_filename}: {e}")
        return False

if __name__ == "__main__":
    pass