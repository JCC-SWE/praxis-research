"# Efficiency & Practicality metrics" 
import os
import sys
from datetime import datetime as dt

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
azure_path = os.path.join(parent_dir, 'azure_resources')
blob_path = os.path.join(parent_dir, 'blob_interface')
sys.path.insert(0, azure_path)
sys.path.insert(0, blob_path)

# Import everything at the top
from blob_interface.upload_to_blob import upload_to_blob
from download_from_blob import download_blob
from phase2_dapt_implementation.get_qa_data import get_abstracts_from_blob
from get_qa_texts import pull_qa_texts

data_2023 = pull_qa_texts(data='qa-2023.txt')
data_2025 = pull_qa_texts(data='qa-2025.txt')

