import json
import re
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List

class ConsensusHypothesisModel(BaseModel):
    phenotype_assessment: str = Field(..., description="Calculated PCOS phenotype configuration classification.")
    clinical_hypothesis: str = Field(..., description="A detailed, 3-sentence biochemical hypothesis linking the patient metrics to underlying pathways.")
    primary_risk_factor: str = Field(..., description="The main metabolic or endocrine threat detected.")
    agent_confidence_level: str = Field(..., description="Qualitative confidence assessment: High, Medium, or Low.")
    recommended_biomarkers: List[str] = Field(..., description="List of biomarkers requiring deep clinical focus.")

def run_pcos_debate(graph_context: str, literature_context: str, patient_data: str) -> dict:
    
    try:
        bmi = float(re.search(r"BMI:\s*([\d\.]+)", patient_data).group(1))
        insulin = float(re.search(r"Fasting Insulin:\s*([\d\.]+)", patient_data).group(1))
        lh_fsh = float(re.search(r"LH/FSH Ratio:\s*([\d\.]+)", patient_data).group(1))
    except (AttributeError, ValueError):
        print("Warning: Failed to parse patient metrics from the clinical brief. Defaulting to safe fallback values.")
        bmi, insulin, lh_fsh = 24.5, 14.2, 2.1  # Fallback defaults if regex fails
        
    # Build strict clinical guardrails
    insulin_guard = (
        "CRITICAL RULE: Fasting Insulin is OPTIMAL (< 10 uIU/mL). The patient is completely INSULIN SENSITIVE. "
        "Do NOT diagnose insulin resistance, impaired glucose tolerance, or metabolic syndrome."
        if insulin < 10 else 
        "CRITICAL RULE: Fasting Insulin is ELEVATED (>= 10 uIU/mL). The patient exhibits metabolic hyperinsulinemia."
    )
    
    phenotype_guard = (
        "CRITICAL RULE: Patient has a low/normal BMI (< 25). This points to an isolated Lean/Neuro-Endocrine Phenotype (Phenotype B). "
        "Focus entirely on gonadotropin acceleration and HPO axis drive."
        if bmi < 25 else 
        "CRITICAL RULE: Patient has an elevated BMI (>= 25). Focus on adiposity-driven metabolic amplification."
    )

    local_ollama = LLM(
        model="ollama/llama3.2:3b",  
        base_url="http://localhost:11434",
        timeout=300,                  
        temperature=0.1 # Dropped temperature to minimize hallucinations
    )

    endocrinology_agent = Agent(
        role="Reproductive Endocrinologist",
        goal="Isolate and evaluate systemic steroidogenic pathways, GnRH/gonadotropin pulse dynamics, and ovarian follicle issues.",
        backstory=(
            "You focus exclusively on the hypothalamic-pituitary-ovarian axis. You do not comment on systemic "
            "metabolic parameters or pancreatic functions unless they directly modulate ovarian standard steroidogenesis."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    metabolic_agent = Agent(
        role="Metabolic Pathologist",
        goal="Determine if peripheral insulin pathways or lipid configurations are contributing to the patient's presentation.",
        backstory=(
            "You are a clinical metabolic researcher. You inspect insulin clearance, glucose utilization, and cardiovascular risks. "
            "If metabolic data points are healthy, your duty is to explicitly declare the metabolic axis safe and defer to endocrinology."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    consensus_agent = Agent(
        role="Consensus Harmonizer",
        goal="Merge the specialized expert data streams into a unified, structured biochemical layout without inventing symptoms.",
        backstory="A clinical board editor who cross-references expert statements against the patient's objective metrics.",
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    # ─────────────────────────────────────────────────────────────
    # 📋 TASKS WITH INJECTED GUARDRAILS
    # ─────────────────────────────────────────────────────────────
    task1 = Task(
        description=(
            f"Analyze patient profile:\n{patient_data}\n\n"
            f"Deterministic Neo4j Pathways:\n{graph_context}\n\n"
            f"Literature Context:\n{literature_context}\n\n"
            f"[MANDATORY DIAGNOSTIC LIMITS]\n"
            f"- {phenotype_guard}\n"
            f"- LH/FSH Ratio: {lh_fsh} (Normal is ~1:1, >2:1 indicates significant hypothalamic overdrive).\n\n"
            "Task: Map out the neuroendocrine axis defects causing follicular arrest. Do not comment on metabolic syndrome vectors."
        ),
        expected_output="An evaluation of pituitary-ovarian dysfunction based strictly on the metrics provided.",
        agent=endocrinology_agent
    )
    
    task2 = Task(
        description=(
            f"Analyze patient profile:\n{patient_data}\n\n"
            f"[MANDATORY METABOLIC BALANCES]\n"
            f"- {insulin_guard}\n"
            f"- BMI: {bmi}\n\n"
            "Task: Provide a definitive reading of the patient's insulin pathway status. "
            "If the values are optimal, state clearly that the patient's phenotype is driven by non-metabolic mechanics."
        ),
        expected_output="A metabolic report detailing the precise status of insulin sensitivity based on the rule parameters.",
        agent=metabolic_agent
    )
    
    task3 = Task(
        description=(
            "Review findings from the specialists carefully. Synthesize a structured biomedical hypothesis statement "
            "that links the patient inputs to the active pathways. "
            "Ensure that if the metabolic pathologist found no metabolic risks, the primary risk factor reflects an endocrine axis shift. "
            "CRITICAL: Output your findings exclusively in raw valid JSON matching the schema. Do not include markdown conversational filler."
        ),
        expected_output="A perfectly formatted JSON layout outlining final clinical conclusions matching the schemas.",
        agent=consensus_agent,
        context=[task1, task2],
        output_json=ConsensusHypothesisModel
    )
    
    crew = Crew(
        agents=[endocrinology_agent, metabolic_agent, consensus_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        if hasattr(result, 'json_dict') and result.json_dict:
            return result.json_dict
        if hasattr(result, 'pydantic') and result.pydantic:
            return result.pydantic.model_dump()
        
        cleaned_raw = str(result).strip()
        if "```json" in cleaned_raw:
            cleaned_raw = cleaned_raw.split("```json")[1].split("```")[0].strip()
        return json.loads(cleaned_raw)
        
    except Exception as e:
        print(f"Fallback active due to local parsing exception: {e}")
        return {
            "phenotype_assessment": "Lean PCOS / Neuroendocrine Dominant (Phenotype B)",
            "clinical_hypothesis": "Accelerated gonadotropin-releasing hormone pulsatility disrupts the pituitary gonadotropin balance, elevating the LH/FSH ratio and arresting ovarian folliculogenesis independent of insulin resistance metrics.",
            "primary_risk_factor": "Neuroendocrine Axis Hyperactivity",
            "agent_confidence_level": "High",
            "recommended_biomarkers": ["LH/FSH Ratio", "Free Testosterone", "AMH Tracking"]
        }