import os
from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Load environment variables from .env file
load_dotenv()

# Get VAULT_URL from environment variables
vault_url = os.getenv('VAULT_URL')

# Verify the URL was loaded
if not vault_url:
    raise ValueError("VAULT_URL not found in environment variables")
def get_secrets():
    """
    Retrieve all secrets from Azure Key Vault
    
    Returns:
        dict: Dictionary containing:
            - subscription_key
            - storage_account_key
            - storage_account_name
    """
        
    try:
        # Authenticate and create client
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        # Retrieve secrets
        secrets = {
            'subscription_key': client.get_secret("subscription-key").value,
            'storage_account_key': client.get_secret("storage-account-key").value,
            'storage_account_name': client.get_secret("storage-account-name").value,
            'account_url': client.get_secret("account-url").value,
            'container': client.get_secret("container").value,
            'connection-string-blob':client.get_secret('connection-string-blob').value,
            'event-registry': client.get_secret('event-registry').value,
            'serp-api-key': client.get_secret('serp-api-key').value,
            'cosmos-conn-string':client.get_secret('cosmos-conn-string').value,
            'cosmos-key': client.get_secret('cosmos-key').value,
            'cosmos-uri': client.get_secret('cosmos-uri').value,
            's-scholar-key': client.get_secret('s-scholar-key').value
        }
        
        return secrets
        
    except Exception as e:
        print(f"Error retrieving secrets: {e}")
        return {}