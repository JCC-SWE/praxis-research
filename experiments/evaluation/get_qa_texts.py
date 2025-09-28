"# Efficiency & Practicality metrics" 
import os
import sys
from datetime import datetime as dt
import os, sys

def find_project_root(start, markers=("pyproject.toml", ".git", "README.md")):
    cur = os.path.abspath(start)
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            return None
        cur = nxt

PROJECT_ROOT = find_project_root(__file__)
if PROJECT_ROOT and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
azure_path = os.path.join(parent_dir, 'azure_resources')
blob_path = os.path.join(parent_dir, 'blob_interface')
phase2_path = os.path.join(parent_dir, 'phase2_dapt_implementation')
#qa_data_path = os.path.join(parent_dir, 'qa_data')  # Add this if get_qa_data.py is in qa_data folder
sys.path.insert(0, phase2_path)
sys.path.insert(0, azure_path)
sys.path.insert(0, blob_path)
#sys.path.insert(0, qa_data_path)  # Add this if needed

# Import everything at the top
from phase2_dapt_implementation.get_qa_data import get_abstracts_from_blob  # Ensure get_qa_data.py exists in the correct folder
# from blob_interface.download_from_blob import download_blob
# from phase1_model_instantiation.func_test import *

def pull_qa_texts(data):
    return get_abstracts_from_blob(data=data)

if __name__ == "__main__":
    data = "qa-2023.txt"
    qa_texts = pull_qa_texts(data)
    print(f"Retrieved {len(qa_texts)} QA texts.")


