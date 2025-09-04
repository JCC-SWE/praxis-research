"""
blob_utils.py - Core Azure Blob Storage utilities
Provides reusable functions for blob operations across multiple API scripts
"""

from azure.storage.blob import BlobServiceClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from azure_resources.keyvault_client import get_secrets

class BlobStorageManager:
    """Manages Azure Blob Storage operations with Key Vault integration"""
    
    def __init__(self):
        self.blob_service_client = None
        self.container_name = None
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize Azure Blob Storage client using Key Vault secrets"""
        try:
            secrets = get_secrets()
            if not secrets:
                raise Exception("Failed to retrieve secrets from Key Vault")
            
            # Build connection string from secrets
            storage_account_name = secrets['storage_account_name']
            storage_account_key = secrets['storage_account_key']
            
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
            
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self.container_name = secrets['container']
            
            print(f"✅ Successfully connected to Azure Storage account: {storage_account_name}")
            print(f"✅ Using container: {self.container_name}")
            
        except Exception as e:
            print(f"❌ Error initializing Azure Storage: {e}")
            self.blob_service_client = None
            self.container_name = None
    
    def is_initialized(self):
        """Check if blob storage is properly initialized"""
        return self.blob_service_client is not None and self.container_name is not None
    
    def download_file(self, blob_name):
        """
        Download existing file from blob storage
        
        Args:
            blob_name (str): Name of the blob file
            
        Returns:
            str: File content or empty string if file doesn't exist
        """
        if not self.is_initialized():
            print("❌ Azure Storage not properly initialized")
            return ""
            
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            content = blob_client.download_blob().readall().decode('utf-8')
            print(f"📥 Downloaded {blob_name} ({len(content)} characters)")
            return content
        except Exception as e:
            print(f"File {blob_name} doesn't exist yet, starting fresh")
            return ""
    
    def upload_file(self, blob_name, content, overwrite=True):
        """
        Upload content to blob storage
        
        Args:
            blob_name (str): Name of the blob file
            content (str): Content to upload
            overwrite (bool): Whether to overwrite existing file
            
        Returns:
            bool: Success status
        """
        if not self.is_initialized():
            print("❌ Azure Storage not properly initialized")
            return False
            
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.upload_blob(content, overwrite=overwrite)
            print(f"✅ Uploaded {blob_name} to blob storage ({len(content)} characters)")
            return True
        except Exception as e:
            print(f"❌ Error uploading {blob_name}: {e}")
            return False
    
    def append_to_file(self, blob_name, new_content, separator="\n"):
        """
        Append content to existing blob file
        
        Args:
            blob_name (str): Name of the blob file
            new_content (str): Content to append
            separator (str): Separator between old and new content
            
        Returns:
            bool: Success status
        """
        if not self.is_initialized():
            print("❌ Azure Storage not properly initialized")
            return False
        
        try:
            # Download existing content
            existing_content = self.download_file(blob_name)
            
            # Combine content
            updated_content = existing_content + separator + new_content
            
            # Upload combined content
            success = self.upload_file(blob_name, updated_content)
            
            if success:
                print(f"📊 Total file size now: {len(updated_content)} characters")
            
            return success
            
        except Exception as e:
            print(f"❌ Error appending to {blob_name}: {e}")
            return False
    
    def list_blobs(self, prefix=None):
        """
        List all blobs in the container
        
        Args:
            prefix (str): Optional prefix to filter blobs
            
        Returns:
            list: List of blob names
        """
        if not self.is_initialized():
            print("❌ Azure Storage not properly initialized")
            return []
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blobs = []
            
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append(blob.name)
            
            print(f"📋 Found {len(blobs)} blobs" + (f" with prefix '{prefix}'" if prefix else ""))
            return blobs
            
        except Exception as e:
            print(f"❌ Error listing blobs: {e}")
            return []
    
    def delete_file(self, blob_name):
        """
        Delete a blob file
        
        Args:
            blob_name (str): Name of the blob file to delete
            
        Returns:
            bool: Success status
        """
        if not self.is_initialized():
            print("❌ Azure Storage not properly initialized")
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.delete_blob()
            print(f"🗑️  Deleted {blob_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting {blob_name}: {e}")
            return False
    
    def file_exists(self, blob_name):
        """
        Check if a blob file exists
        
        Args:
            blob_name (str): Name of the blob file
            
        Returns:
            bool: True if file exists, False otherwise
        """
        if not self.is_initialized():
            return False
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            return blob_client.exists()
            
        except Exception as e:
            print(f"❌ Error checking if {blob_name} exists: {e}")
            return False

# Create a global instance for easy importing
blob_manager = BlobStorageManager()

# Convenience functions for backward compatibility
def download_existing_file(blob_name):
    """Convenience function - download file from blob storage"""
    return blob_manager.download_file(blob_name)

def upload_updated_file(blob_name, content):
    """Convenience function - upload file to blob storage"""
    return blob_manager.upload_file(blob_name, content)

def append_to_blob_file(blob_name, content, separator="\n"):
    """Convenience function - append content to blob file"""
    return blob_manager.append_to_file(blob_name, content, separator)