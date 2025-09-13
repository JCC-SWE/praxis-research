"""
qwen_model_manager.py - Qwen model specific blob storage functions
Uses existing blob infrastructure without modifying upload_to_blob.py
"""
import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'azure_resources'))
from keyvault_client import get_secrets
from azure.storage.blob import BlobServiceClient

def upload_qwen_model(model_path, model_version="qwen-2.5-3b"):
    """
    Upload Qwen model directory to blob storage using existing infrastructure
    
    Args:
        model_path (str): Local path to the Qwen model directory
        model_version (str): Version identifier for the model
        
    Returns:
        bool: Success status
    """
    try:
        # Get secrets and connect using existing pattern
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model path {model_path} does not exist")
            return False
        
        uploaded_files = []
        
        # Upload all files in the model directory
        for file_path in model_path.rglob('*'):
            if file_path.is_file():
                # Create blob name with qwen prefix
                relative_path = file_path.relative_to(model_path)
                blob_name = f"qwen_models/{model_version}/{relative_path}"
                
                print(f"Uploading {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)...")
                
                blob_client = blob_service_client.get_blob_client(
                    container=secrets['container'], 
                    blob=blob_name
                )
                
                with open(file_path, 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)
                    uploaded_files.append(blob_name)
        
        print(f"✅ Uploaded {len(uploaded_files)} files for Qwen model {model_version}")
        
        # Create a manifest file for this model
        manifest = {
            "model_version": model_version,
            "model_type": "qwen",
            "files": uploaded_files,
            "file_count": len(uploaded_files),
            "upload_timestamp": str(pd.Timestamp.now()) if 'pd' in globals() else "timestamp_unavailable"
        }
        
        manifest_blob = f"qwen_models/{model_version}/manifest.json"
        blob_client = blob_service_client.get_blob_client(
            container=secrets['container'], 
            blob=manifest_blob
        )
        blob_client.upload_blob(json.dumps(manifest, indent=2), overwrite=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading Qwen model {model_version}: {e}")
        return False

def download_qwen_model(model_version="qwen-2.5-3b", local_path="/workspace/praxis-research/base-model"):
    """
    Download Qwen model from blob storage to local directory
    
    Args:
        model_version (str): Version of the model to download
        local_path (str): Local path to download the model to
        
    Returns:
        str: Path to downloaded model, or None if failed
    """
    from pathlib import Path
    
    # Ensure base local path exists
    base_path = Path(local_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists locally
    model_local_path = base_path / model_version
    
    # Check if model directory exists and has files
    if model_local_path.exists() and any(model_local_path.iterdir()):
        print(f"✅ Model {model_version} already exists locally at {model_local_path}")
        return str(model_local_path)
    
    try:
        # Get secrets and connect using existing pattern
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(secrets['container'])
        
        # Create local directory
        model_local_path.mkdir(parents=True, exist_ok=True)
        
        # List all blobs for this Qwen model
        model_prefix = f"qwen_models/{model_version}/"
        blobs = list(container_client.list_blobs(name_starts_with=model_prefix))
        
        if not blobs:
            print(f"❌ No files found for Qwen model {model_version}")
            return None
        
        print(f"📥 Downloading model {model_version} from Azure storage...")
        
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
                container=secrets['container'], 
                blob=blob.name
            )
            
            with open(local_file_path, 'wb') as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            downloaded_files.append(str(local_file_path))
        
        print(f"✅ Downloaded {len(downloaded_files)} files for Qwen model {model_version}")
        return str(model_local_path)
            
    except Exception as e:
        print(f"❌ Error downloading Qwen model {model_version}: {e}")
        return None

def get_qwen_model(model_version="qwen-2.5-3b", cache_dir="/workspace/praxis-research/base-model"):
    """
    Get Qwen model from local cache or download from blob if not available
    This is the main function your DAPT code should use
    
    Args:
        model_version (str): Version of the Qwen model
        cache_dir (str): Local cache directory
        
    Returns:
        tuple: (model, tokenizer) - The loaded Qwen model and tokenizer
    """
    from transformers import AutoModel, AutoTokenizer
    from pathlib import Path
    
    local_model_path = Path(cache_dir) / model_version
    
    # Check if model exists locally and has files
    if local_model_path.exists() and any(local_model_path.iterdir()):
        print(f"✅ Using cached Qwen model at {local_model_path}")
    else:
        # Download from blob
        print(f"📥 Downloading Qwen model {model_version} from Azure Blob...")
        downloaded_path = download_qwen_model(model_version, cache_dir)
        
        if not downloaded_path:
            raise Exception(f"Failed to download Qwen model {model_version}")
        
        local_model_path = Path(downloaded_path)
    
    # Load the model and tokenizer
    print(f"🔄 Loading model and tokenizer from {local_model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
        model = AutoModel.from_pretrained(str(local_model_path))
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Successfully loaded Qwen model {model_version}")
        return model, tokenizer
        
    except Exception as e:
        raise Exception(f"Failed to load model from {local_model_path}: {e}")

def list_qwen_models():
    """
    List all available Qwen models in blob storage
    
    Returns:
        dict: Dictionary of available Qwen models and their info
    """
    try:
        # Get secrets using your existing get_secrets function
        secrets = get_secrets()
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={secrets['storage_account_name']};AccountKey={secrets['storage_account_key']};EndpointSuffix=core.windows.net"
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(secrets['container'])
        
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
        print(f"❌ Error listing Qwen models: {e}")
        print("Make sure you're authenticated with Azure and have access to the storage account")
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