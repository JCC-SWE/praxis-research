#Call Open AI API to create Q&A pairs from a given text
#Get data from s_semantic_scholar
#Parse data to get abstract
#Create Questions
#Insert questions into gpt
#Save Q&A pairs to s_qa_pairs.csv---columns: paper_id, question, answer, date 
import os
import sys
from datetime import datetime as dt, timezone

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
blob_path = os.path.join(parent_dir, 'blob_interface')
sys.path.insert(0, blob_path)

from upload_to_blob import upload_to_blob
from openai_client import GPT 
from get_qa_data import get_abstracts_from_blob

def parse_abstracts_text(abstracts_text):
    """Parse the abstracts.txt content into individual papers"""
    papers = []
    sections = abstracts_text.split("-" * 80)
    
    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) >= 3:
            paper_data = {}
            for line in lines:
                if line.startswith("Paper ID:"):
                    paper_data['paper_id'] = line.replace("Paper ID:", "").strip()
                elif line.startswith("Title:"):
                    paper_data['title'] = line.replace("Title:", "").strip()
                elif line.startswith("Abstract:"):
                    paper_data['abstract'] = line.replace("Abstract:", "").strip()
            
            if all(key in paper_data for key in ['paper_id', 'title', 'abstract']):
                papers.append(paper_data)
    
    return papers


def generate_qa_pairs(papers):
    """Generate Q&A pairs from papers using GPT"""
    gpt = GPT()
    qa_pairs = []
    
    for paper in papers:
        prompt = f"""Based on this research paper abstract, generate one insightful question and answer pair.

        Title: {paper['title']}
        Abstract: {paper['abstract']}

        Create a question about ONE of these aspects:
        - A specific methodology and how it was applied
        - A key finding and its implications
        - A novel contribution and why it matters
        - Results or performance metrics achieved

        Your answer should:
        - Be 3-5 sentences explaining the concept in depth
        - Provide context and significance, not just facts
        - Avoid bullet points or simple lists
        - Explain the "why" or "how" behind the approach

        Format:
        Question: In the paper {paper['title']}, [specific question]
        Answer: [detailed, explanatory answer]"""

        try:
            response = gpt.ask(prompt, max_tokens=1000, temperature=0.5)
            
            # Parse the response to extract question and answer
            question = ""
            answer = ""
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
            
            qa_pairs.append({
                'question': question,
                'answer': answer
            })
            
            print(f"Generated Q&A pair {len(qa_pairs)}")
            
        except Exception as e:
            print(f"Error generating Q&A: {e}")
    
    return qa_pairs

import json

def save_qa_to_blob(qa_pairs):
    """Save Q&A pairs to qa.txt in blob storage"""
    content = json.dumps(qa_pairs, indent=2)
    
    success = upload_to_blob(content, "qa-2025.txt")
    
    if success:
        print(f"Saved {len(qa_pairs)} Q&A pairs to qa-2025.txt")
    else:
        print("Failed to save Q&A pairs to blob")
    
    return success

def create_qa_pipeline():
    """Main function to create Q&A pairs from abstracts"""
    # Get abstracts from blob using existing function
    abstracts_text = get_abstracts_from_blob()
    if not abstracts_text:
        return False
    
    # Parse abstracts into individual papers
    papers = parse_abstracts_text(abstracts_text)
    print(f"Parsed {len(papers)} papers from abstracts")
    
    # Generate Q&A pairs
    qa_pairs = generate_qa_pairs(papers)
    
    # Save to blob
    if qa_pairs:
        return save_qa_to_blob(qa_pairs)
    else:
        print("No Q&A pairs generated")
        return False

if __name__ == "__main__":
    create_qa_pipeline()