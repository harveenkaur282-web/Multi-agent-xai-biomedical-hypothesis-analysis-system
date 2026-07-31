import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

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

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
model_name = "qwen2.5:7b"

print(f"Connecting to {base_url}/api/generate with model {model_name}...")
try:
    res = requests.post(
        f"{base_url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
        timeout=90
    )
    print(f"Status Code: {res.status_code}")
    if res.status_code == 200:
        response_text = res.json()["response"]
        print("Response received:")
        print(response_text)
        try:
            parsed = json.loads(response_text)
            print("Successfully parsed JSON!")
        except Exception as parse_err:
            print(f"Failed to parse JSON: {parse_err}")
    else:
        print(f"Error response: {res.text}")
except Exception as e:
    print(f"Request failed: {e}")
