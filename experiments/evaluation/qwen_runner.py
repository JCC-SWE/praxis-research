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
phase1_path = os.path.join(parent_dir, 'phase1_model_instantiation')
#qa_data_path = os.path.join(parent_dir, 'qa_data')  # Add this if get_qa_data.py is in qa_data folder
sys.path.insert(0, phase1_path)
sys.path.insert(0, azure_path)
sys.path.insert(0, blob_path)
from model_setup import get_qwen_model  # Ensure model_setup.py exists in the correct folder

