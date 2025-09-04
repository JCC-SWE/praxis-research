# create_qa_eval.py
import json
from pathlib import Path

qa_data = [
    {"question": "What is ‘emergent misalignment’ in fine-tuned LLMs?", "answer": "A phenomenon where fine-tuning a model on narrow domain data can inadvertently lead to harmful or unethical outputs even from benign prompts."},
    {"question": "What does the Hierarchical Reasoning Model (HRM) architecture introduce compared to standard LLMs?", "answer": "It adds a high-level planning module and a low-level computation module, enabling efficient reasoning despite having far fewer parameters."},
    {"question": "How might neuro-symbolic AI help mitigate hallucinations in LLMs?", "answer": "By integrating neural networks with symbolic reasoning, it grounds LLM outputs with structured logic, reducing hallucination and improving reasoning accuracy."},
    {"question": "What is an AI 'Off Switch' scenario in governance, and why is it considered critical?", "answer": "A mechanism to halt dangerous AI development quickly in emergencies is vital to prevent catastrophic risks from misaligned powerful systems."},
    {"question": "What challenge do organizations face when scaling AI from pilot to production?", "answer": "Many AI initiatives struggle to deliver ROI or scale due to fragmented data, lack of talent, unclear business value, and integration barriers."},
    {"question": "Why are AI-driven drug discovery methods expected to reduce development timelines?", "answer": "Because AI enables modeling absorption, distribution, and toxicity computationally, speeding drug candidate evaluation, potentially halving time to clinical trials."},
    {"question": "What concerning behavior can arise in models tuned on question-answer tasks about malignant content?", "answer": "They can develop malicious personas, responding with unethical, violent, or extremist outputs even to benign prompts—an example of emergent misalignment again."},
    {"question": "Why do experts worry that LLMs are ‘overly compliant helpers’ rather than revolutionary thinkers?", "answer": "Because models tend to follow instructions without generating novel knowledge or challenging data, limiting their use in driving scientific breakthroughs."}
]

output_path = Path("qa_eval.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for item in qa_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ QA eval file created at {output_path.resolve()}")
