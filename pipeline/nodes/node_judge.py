import os
import json
import requests
from pipeline.state import PCOSState

def node_judge_fn(state: PCOSState) -> PCOSState:
    """
    Node 2.5: LLM-as-a-Judge and RAG Evaluation Node.
    Evaluates Rotterdam Criteria Accuracy, Clinical Correlation, Differential Rule-out Logic,
    Faithfulness, Context Relevance, and Answer Relevance.
    Generates structured scores and formats them into a Markdown table for the report.
    """
    print("\n" + "=" * 60)
    print("[JUDGE NODE] COMPUTING QUANTITATIVE CLINICAL & RAG METRICS...")
    print("=" * 60)

    # 1. Retrieve data from state
    raw_patient = state.get("raw_input", {}) or {}
    hypothesis_data = state.get("clinical_hypothesis", {}) or {}
    chunks = state.get("retrieved_chunks", []) or []

    # Safe extraction of patient values
    lh_fsh_val = float(raw_patient.get("lh_fsh_ratio", 2.1))
    insulin_val = float(raw_patient.get("fasting_insulin", 14.2))
    bmi_val = float(raw_patient.get("bmi", 24.5))
    amh_val = float(raw_patient.get("amh_levels", 4.2))
    testo_val = float(raw_patient.get("free_testosterone", 55.0))
    remarks = raw_patient.get("clinical_remarks", "")

    patient_context_str = (
        f"Patient Profile:\n"
        f"- Age: {raw_patient.get('age', 27)} years old\n"
        f"- BMI: {bmi_val} kg/m²\n"
        f"- LH/FSH Ratio: {lh_fsh_val}\n"
        f"- Fasting Insulin: {insulin_val} uIU/mL\n"
        f"- AMH Levels: {amh_val} ng/mL\n"
        f"- Free Testosterone: {testo_val} ng/dL\n"
        f"- Remarks: {remarks}"
    )

    hypothesis_str = (
        f"Generated Diagnosis & Hypothesis:\n"
        f"- Phenotype Assessment: {hypothesis_data.get('phenotype_assessment', 'N/A')}\n"
        f"- Clinical Hypothesis: {hypothesis_data.get('clinical_hypothesis', 'N/A')}\n"
        f"- Primary Risk Factor: {hypothesis_data.get('primary_risk_factor', 'N/A')}\n"
        f"- Confidence Level: {hypothesis_data.get('agent_confidence_level', 'N/A')}\n"
        f"- Differential Diagnoses: {hypothesis_data.get('differential_diagnoses', [])}\n"
        f"- Recommended Biomarkers: {hypothesis_data.get('recommended_biomarkers', [])}"
    )

    literature_list = []
    for c in chunks:
        if isinstance(c, dict) and c.get("is_paper") is True:
            literature_list.append(f"Chunk ID: {c.get('id', 'N/A')}\nTitle: {c.get('title', 'N/A')}\nText: {c.get('text', 'N/A')}")
    
    literature_context_str = "\n\n".join(literature_list) if literature_list else "No literature retrieved."

    # 2. Setup LLM Backend Call
    llm_choice = raw_patient.get("llm_choice", "Ollama Local")
    groq_api_key = raw_patient.get("groq_api_key") or os.getenv("GROQ_API_KEY")

    prompt = f"""
You are an expert biomedical diagnostics validator acting as an LLM-as-a-Judge and RAG evaluator.
Evaluate the generated diagnosis against the patient profile and retrieved literature chunks.

{patient_context_str}

{hypothesis_str}

Retrieved Literature Chunks:
{literature_context_str}

You must evaluate and score the following six metrics (from 0 to 100) and provide a concise clinical justification for each:
1. **rotterdam_accuracy**: Rotterdam Criteria Accuracy. Measures if the hypothesis correctly verifies Oligomenorrhea, Hyperandrogenism, and Polycystic Ovaries morphology as described in the clinical remarks and patient serology.
2. **clinical_correlation**: Clinical Correlation. Measures how well the hypothesis aligns with the patient's laboratory metrics (LH/FSH ratio, Fasting Insulin, AMH, Testosterone, and BMI).
3. **differential_logic**: Differential Rule-out Logic. Checks completeness of missing biomarker identification (17-OH progesterone, DHEA-S, prolactin, TSH, free T4) to rule out PCOS mimics.
4. **faithfulness**: Faithfulness (Groundedness). Measures if the generated hypothesis is strictly grounded in the retrieved chunks (hallucination check).
5. **context_relevance**: Context Relevance. Measures how relevant the retrieved chunks are to the user's initial clinical case.
6. **answer_relevance**: Answer Relevance. Measures if the hypothesis directly addresses the clinical query and case presentation.

Return ONLY a valid JSON object matching this structure:
{{
  "rotterdam_accuracy": {{ "score": 90, "justification": "brief explanation" }},
  "clinical_correlation": {{ "score": 85, "justification": "brief explanation" }},
  "differential_logic": {{ "score": 80, "justification": "brief explanation" }},
  "faithfulness": {{ "score": 95, "justification": "brief explanation" }},
  "context_relevance": {{ "score": 90, "justification": "brief explanation" }},
  "answer_relevance": {{ "score": 85, "justification": "brief explanation" }}
}}
"""

    response_json = {}
    try:
        if llm_choice == "Groq API" and groq_api_key:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                response_json = json.loads(res.json()["choices"][0]["message"]["content"])
        else:
            # Fallback to Ollama Local
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            raw_model = os.getenv("LLM_MODEL", "ollama/llama3.2:3b")
            model_name = raw_model.split("/")[-1] if "/" in raw_model else raw_model
            
            res = requests.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
                timeout=45
            )
            if res.status_code == 200:
                response_json = json.loads(res.json()["response"])
    except Exception as e:
        print(f"[JUDGE NODE WARNING] LLM Judge call failed: {e}. Falling back to rule-based fallback evaluations.")

    # Rule-based fallback if LLM response is empty or failed
    if not response_json or "rotterdam_accuracy" not in response_json:
        # Fallback values based on clinical parameters
        has_all_rotterdam = (lh_fsh_val > 2.0 or testo_val > 45) and "polycystic" in remarks.lower() and "oligomenorrhea" in remarks.lower()
        rotterdam_score = 95 if has_all_rotterdam else 75
        correlation_score = 90 if (insulin_val > 14.0 and bmi_val > 25.0 and "classic metabolic" in str(hypothesis_data).lower()) else 80
        
        missing_markers = hypothesis_data.get("recommended_biomarkers", [])
        diff_score = 100 - (len(missing_markers) * 5)
        
        response_json = {
            "rotterdam_accuracy": { "score": rotterdam_score, "justification": "Calculated via fallback clinical heuristic engine." },
            "clinical_correlation": { "score": correlation_score, "justification": "Calculated via fallback correlation mapping." },
            "differential_logic": { "score": max(50, diff_score), "justification": "Calculated via checking missing rule-out biomarker counts." },
            "faithfulness": { "score": None, "justification": "Unavailable: Groundedness metrics require an active LLM-as-a-Judge API connection." },
            "context_relevance": { "score": None, "justification": "Unavailable: Relevance metrics require an active LLM-as-a-Judge API connection." },
            "answer_relevance": { "score": None, "justification": "Unavailable: Answer alignment metrics require an active LLM-as-a-Judge API connection." }
        }

    # Helper function to format score safely
    def format_score(score_val):
        if score_val is None:
            return "Unavailable"
        return f"{score_val}%"

    # 3. Create Markdown Table
    table_lines = [
        "| Evaluation Dimension | Score | Clinical Justification / Reasoning |",
        "| :--- | :--- | :--- |",
        f"| **Rotterdam Criteria Accuracy** | {format_score(response_json['rotterdam_accuracy']['score'])} | {response_json['rotterdam_accuracy']['justification']} |",
        f"| **Clinical Correlation** | {format_score(response_json['clinical_correlation']['score'])} | {response_json['clinical_correlation']['justification']} |",
        f"| **Differential Rule-out Logic** | {format_score(response_json['differential_logic']['score'])} | {response_json['differential_logic']['justification']} |",
        f"| **Faithfulness (Groundedness)** | {format_score(response_json['faithfulness']['score'])} | {response_json['faithfulness']['justification']} |",
        f"| **Context Relevance** | {format_score(response_json['context_relevance']['score'])} | {response_json['context_relevance']['justification']} |",
        f"| **Answer Relevance** | {format_score(response_json['answer_relevance']['score'])} | {response_json['answer_relevance']['justification']} |"
    ]
    table_markdown = "\n".join(table_lines)

    # 4. Save metrics into state
    state["llm_judge_evaluation"] = {
        "scores": {k: v["score"] for k, v in response_json.items()},
        "justifications": {k: v["justification"] for k, v in response_json.items()},
        "table_markdown": table_markdown
    }

    state["rag_eval_metrics"] = {
        "faithfulness": response_json["faithfulness"]["score"],
        "context_relevance": response_json["context_relevance"]["score"],
        "answer_relevance": response_json["answer_relevance"]["score"]
    }

    print("[JUDGE NODE] Evaluation metrics calculated successfully and stored in state.")
    print("=" * 60 + "\n")

    return state
