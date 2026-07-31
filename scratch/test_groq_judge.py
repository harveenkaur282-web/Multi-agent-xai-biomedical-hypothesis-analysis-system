import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print(f"Loaded GROQ_API_KEY: {groq_api_key[:10] if groq_api_key else 'None'}...")

prompt = """
You are an expert biomedical diagnostics validator acting as an LLM-as-a-Judge.
Return ONLY a valid JSON object matching this structure:
{
  "rotterdam_accuracy": { "score": 90, "justification": "test" },
  "clinical_correlation": { "score": 85, "justification": "test" },
  "differential_logic": { "score": 80, "justification": "test" },
  "faithfulness": { "score": 95, "justification": "test" },
  "context_relevance": { "score": 90, "justification": "test" },
  "answer_relevance": { "score": 85, "justification": "test" }
}
"""

if not groq_api_key:
    print("Error: No GROQ_API_KEY found in env!")
else:
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    print("Sending request to Groq...")
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        print(f"Status Code: {res.status_code}")
        print("Response headers:", res.headers)
        if res.status_code == 200:
            print("Response text:")
            print(res.json()["choices"][0]["message"]["content"])
        else:
            print(f"Failed: {res.text}")
    except Exception as e:
        print(f"Exception during request: {e}")
