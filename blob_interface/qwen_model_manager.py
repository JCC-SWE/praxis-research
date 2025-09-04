"""
qwen_model_manager.py - Qwen model specific blob storage functions
Uses Azure CLI authentication for RunPod
"""
import sys
import os
import json
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.identity import AzureCliCredential

def _get_blob_client():
    """Get blob client using keyvault authentication first (working method)"""
    try:
        # Use your existing working keyvault method first
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'azure_resources'))
        from keyvault_client import get_secrets
        
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        return blob_service_client, secrets['container']
        
    except Exception as e:
        print(f"Keyvault auth failed: {e}")
        # Fallback to Azure CLI if keyvault fails
        try:
            from azure.identity import AzureCliCredential
            credential = AzureCliCredential()
            
            storage_account_name = "praxisstorage"
            container_name = "praxisblob"
            
            account_url = f"https://{storage_account_name}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url, credential=credential)
            
            return blob_service_client, container_name
        except Exception as e2:
            raise Exception(f"Both keyvault and Azure CLI auth failed. Keyvault: {e}, CLI: {e2}")

def upload_qwen_model(model_path, model_version="qwen-2.5-3b"):
    """
    Upload Qwen model directory to blob storage with progress tracking
    """
    try:
        blob_service_client, container_name = _get_blob_client()
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"Model path {model_path} does not exist")
            return False
        
        # Get all files first and show what we're uploading
        all_files = [f for f in model_path.rglob('*') if f.is_file()]
        print(f"Found {len(all_files)} files to upload")
        
        # Show file sizes
        for file_path in all_files:
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  {file_path.name}: {size_mb:.1f} MB")
        
        uploaded_files = []
        
        # Upload files with progress tracking
        for i, file_path in enumerate(all_files, 1):
            try:
                # Skip symlinks and follow the actual files
                if file_path.is_symlink():
                    # Resolve symlink to actual file
                    actual_file = file_path.resolve()
                    if not actual_file.exists():
                        print(f"[{i}/{len(all_files)}] Skipping broken symlink: {file_path}")
                        continue
                    file_to_upload = actual_file
                else:
                    file_to_upload = file_path
                
                # Create blob name with qwen prefix
                relative_path = file_path.relative_to(model_path)
                blob_name = f"qwen_models/{model_version}/{relative_path}"
                
                size_mb = file_to_upload.stat().st_size / 1024 / 1024
                print(f"[{i}/{len(all_files)}] Uploading {file_path.name} ({size_mb:.1f} MB)...")
                
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                import time
                start_time = time.time()
                
                with open(file_to_upload, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                elapsed = time.time() - start_time
                speed = size_mb / elapsed if elapsed > 0 else 0
                print(f"    Completed in {elapsed:.1f}s ({speed:.1f} MB/s)")
                
                uploaded_files.append(blob_name)
                
            except Exception as file_error:
                print(f"    ERROR uploading {file_path.name}: {file_error}")
                return False
        
        print(f"Uploaded {len(uploaded_files)} files for Qwen model {model_version}")
        
        # Create a manifest file for this model
        manifest = {
            "model_version": model_version,
            "model_type": "qwen",
            "files": uploaded_files,
            "file_count": len(uploaded_files),
            "upload_timestamp": "timestamp_placeholder"
        }
        
        manifest_blob = f"qwen_models/{model_version}/manifest.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=manifest_blob
        )
        blob_client.upload_blob(json.dumps(manifest, indent=2), overwrite=True)
        print("Manifest uploaded")
        
        return True
        
    except Exception as e:
        print(f"Error uploading Qwen model {model_version}: {e}")
        return False

def download_qwen_model(model_version="qwen-2.5-3b", local_path="/workspace/models"):
    """
    Download Qwen model from blob storage using Azure CLI authentication
    """
    try:
        blob_service_client, container_name = _get_blob_client()
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create local directory
        model_local_path = Path(local_path) / model_version
        model_local_path.mkdir(parents=True, exist_ok=True)
        
        # List all blobs for this Qwen model
        model_prefix = f"qwen_models/{model_version}/"
        blobs = list(container_client.list_blobs(name_starts_with=model_prefix))
        
        if not blobs:
            print(f"No files found for Qwen model {model_version}")
            return None
        
        downloaded_files = []
        for blob in blobs:
            if blob.name.endswith('manifest.json'):
                continue  # Skip manifest for now
                
            # Remove the model prefix to get relative path
            relative_path = blob.name[len(model_prefix):]
            local_file_path = model_local_path / relative_path
            
            # Create subdirectories if needed
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {blob.name.split('/')[-1]} ({blob.size / 1024 / 1024:.1f} MB)...")
            
            # Download the blob
            blob_client = blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob.name
            )
            
            with open(local_file_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            downloaded_files.append(str(local_file_path))
        
        print(f"Downloaded {len(downloaded_files)} files for Qwen model {model_version}")
        return str(model_local_path)
            
    except Exception as e:
        print(f"Error downloading Qwen model {model_version}: {e}")
        return None

def get_qwen_model(model_version="qwen-2.5-3b", cache_dir="/workspace/model_cache"):
    """
    Get Qwen model from local cache or download from blob if not available
    This is the main function your DAPT code should use
    """
    local_model_path = Path(cache_dir) / model_version
    
    # Check if model exists locally and has files
    if local_model_path.exists() and any(local_model_path.iterdir()):
        print(f"Using cached Qwen model at {local_model_path}")
        return str(local_model_path)
    
    # Download from blob
    print(f"Downloading Qwen model {model_version} from Azure Blob...")
    downloaded_path = download_qwen_model(model_version, cache_dir)
    
    if downloaded_path:
        return downloaded_path
    else:
        raise Exception(f"Failed to download Qwen model {model_version}")

def list_qwen_models():
    """
    List all available Qwen models in blob storage using Azure CLI authentication
    """
    try:
        blob_service_client, container_name = _get_blob_client()
        container_client = blob_service_client.get_container_client(container_name)
        
        # List all Qwen model manifests
        manifest_blobs = container_client.list_blobs(name_starts_with="qwen_models/")
        
        models = {}
        for blob in manifest_blobs:
            if blob.name.endswith('manifest.json'):
                # Extract model version
                parts = blob.name.split('/')
                if len(parts) >= 3:
                    model_version = parts[1]
                    models[model_version] = {
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'manifest_blob': blob.name
                    }
        
        return models
        
    except Exception as e:
        print(f"Error listing Qwen models: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    print("Available Qwen models:")
    models = list_qwen_models()
    for name, info in models.items():
        print(f"  {name}: modified {info['last_modified']}")
    
    # Test download
    # model_path = get_qwen_model("qwen-2.5-3b")
    # print(f"Model available at: {model_path}")