import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class ConsensusHypothesisModel(BaseModel):
    phenotype_assessment: str = Field(..., description="Clinician-style phenotype classification.")
    clinical_hypothesis: str = Field(..., description="Formal biomedical hypothesis statement explaining findings.")
    primary_risk_factor: str = Field(..., description="Main endocrine or metabolic concern identified.")
    agent_confidence_level: str = Field(..., description="Confidence assessment: High, Medium, Low.")
    recommended_biomarkers: List[str] = Field(..., description="Biomarkers from the rule-out list missing from the patient record.")
    pcos_diagnosis_likely: bool = Field(..., description="True if Rotterdam criteria are met, False otherwise.")
    differential_diagnoses: List[str] = Field(..., description="Differential rule-outs based on missing biomarkers.")

PHENOTYPE_LOCK = [
    "lean hyperandrogenic PCOS phenotype, with possible neuroendocrine dominance",
    "classic metabolic PCOS phenotype",
    "non-PCOS endocrine pattern"
]

MISSING_BIOMARKERS_LOCK = [
    "17-OH progesterone",
    "DHEA-S",
    "prolactin",
    "TSH",
    "free T4"
]

def _extract_value(pattern: str, text: str, default: float) -> float:
    try:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default
    except Exception:
        return default

def _summarize_context(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:max_chars]

def run_pcos_debate(graph_context: str, literature_context: str, patient_data: str) -> dict:
    bmi = _extract_value(r"BMI:\s*([\d\.]+)", patient_data, 24.5)
    insulin = _extract_value(r"Fasting Insulin:\s*([\d\.]+)", patient_data, 14.2)
    lh_fsh = _extract_value(r"LH/FSH Ratio:\s*([\d\.]+)", patient_data, 2.1)
    testosterone = _extract_value(r"Testosterone:\s*([\d\.]+)", patient_data, 55.0)
    amh = _extract_value(r"AMH:\s*([\d\.]+)", patient_data, 4.2)

    try:
        clinical_remarks = re.search(r"Clinical Remarks:\s*([^\n]+)", patient_data).group(1)
    except AttributeError:
        clinical_remarks = "Oligomenorrhea, acne, polycystic ovary morphology on ultrasound"

    graph_excerpt = _summarize_context(graph_context, max_chars=400)
    lit_excerpt = _summarize_context(literature_context, max_chars=400)

    local_llama = LLM(
        model="ollama/llama3.2:3b", 
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout=300
    )

    endocrinology_agent = Agent(
        role="Reproductive Endocrinologist",
        goal="Classify the dynamic patient presentation using Rotterdam criteria.",
        backstory="Expert in mapping cycle history, androgen signs, and morphology to specific reproductive phenotypes.",
        llm=local_llama,
        verbose=True
    )

    metabolic_agent = Agent(
        role="Metabolic & Endocrine Pathway Specialist",
        goal="Analyze insulin sensitivity, BMI metrics, and metabolic risk pathways.",
        backstory="Meticulous bio-statistician focused on identifying hidden insulin resistance, hyperinsulinemia, and metabolic features of endocrine disorders.",
        llm=local_llama,
        verbose=True
    )

    differential_agent = Agent(
        role="Clinical Differential Diagnosis Expert",
        goal="Determine missing rule-out tests by comparing current patient metrics against baseline requirements.",
        backstory="Diagnostic coordinator verifying mimic panels to confirm clinical rule-outs cleanly.",
        llm=local_llama,
        verbose=True
    )

    consensus_agent = Agent(
        role="Consensus Harmonizer",
        goal="Synthesize structured observations into an exact target schema validation format.",
        backstory="Compiles independent expert views into a clean final data payload with zero omissions.",
        llm=local_llama,
        verbose=True
    )

    # --- TASK DEFINITIONS ---

    task1_reproductive = Task(
        description=(
            f"Patient Context Metrics:\n- BMI: {bmi}\n- LH/FSH ratio: {lh_fsh}\n- Testosterone: {testosterone}\n- AMH: {amh}\n- Remarks: {clinical_remarks}\n\n"
            f"Task Instruction:\nSelect the true matching classification out of these exact options: {PHENOTYPE_LOCK}.\n"
            f"Formulate exactly one clean summary hypothesis sentence evaluating the reproductive picture."
        ),
        expected_output="A phenotype categorization string and one supporting hypothesis sentence.",
        agent=endocrinology_agent
    )

    task2_metabolic = Task(
        description=(
            f"Evaluate the patient's metabolic risk vector using:\n- BMI: {bmi}\n- Fasting Insulin: {insulin} uIU/mL\n\n"
            f"Task Instruction:\nDetermine the primary risk factor or primary threat vector (e.g. 'Insulin Signaling Dysfunction', 'Androgen Excess Dominance')."
        ),
        expected_output="A single primary risk factor title/phrase.",
        agent=metabolic_agent
    )

    task3_differential = Task(
        description=(
            f"Labs already tested: LH/FSH, Fasting Insulin, AMH, Testosterone\n"
            f"Baseline Target Directory: {MISSING_BIOMARKERS_LOCK}\n\n"
            f"Task Instruction:\nIdentify which items from the directory are completely missing from labs already tested. Return ONLY those absent tests."
        ),
        expected_output="An array of untested missing biomarkers.",
        agent=differential_agent
    )

    task4_consensus = Task(
        description=(
            "Collect the outputs from the Reproductive task, the Metabolic task, and the Differential task. "
            "Map them cleanly to the properties requested by the target JSON structure. Ensure NO fields are left blank or empty."
        ),
        expected_output="Raw valid JSON matching the ConsensusHypothesisModel schema.",
        agent=consensus_agent,
        context=[task1_reproductive, task2_metabolic, task3_differential],
        output_json=ConsensusHypothesisModel
    )

    crew = Crew(
        agents=[endocrinology_agent, metabolic_agent, differential_agent, consensus_agent],
        tasks=[task1_reproductive, task2_metabolic, task3_differential, task4_consensus],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    if hasattr(result, "json_dict") and result.json_dict:
        return result.json_dict
    if hasattr(result, "pydantic") and result.pydantic:
        return result.pydantic.model_dump()
        
    cleaned_raw = str(result).strip()
    if "```json" in cleaned_raw:
        cleaned_raw = cleaned_raw.split("```json", 1)[1].split("```", 1)[0].strip()
    return json.loads(cleaned_raw)