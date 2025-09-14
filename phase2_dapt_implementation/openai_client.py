import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GPT:
    def __init__(self, model="gpt-3.5-turbo"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def ask(self, prompt, max_tokens=150, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

# Usage
if __name__ == "__main__":
    gpt = GPT()
    response = gpt.ask("Hello, how are you?")
    print(response)