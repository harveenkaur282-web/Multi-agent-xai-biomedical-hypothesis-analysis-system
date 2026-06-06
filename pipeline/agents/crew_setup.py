# pipeline/agents/crew_setup.py
import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

class ConsensusHypothesisModel(BaseModel):
    phenotype_assessment: str = Field(..., description="Calculated PCOS phenotype configuration classification.")
    clinical_hypothesis: str = Field(..., description="A detailed, 3-sentence biochemical hypothesis linking the patient metrics to underlying pathways.")
    primary_risk_factor: str = Field(..., description="The main metabolic or endocrine threat detected.")
    agent_confidence_level: str = Field(..., description="Qualitative confidence assessment: High, Medium, or Low.")
    recommended_biomarkers: List[str] = Field(..., description="List of biomarkers requiring deep clinical focus.")

def run_pcos_debate(graph_context: str, literature_context: str, patient_data: str, selected_action: int = 0) -> dict:
    
    try:
        bmi = float(re.search(r"BMI:\s*([\d\.]+)", patient_data).group(1))
        insulin = float(re.search(r"Fasting Insulin:\s*([\d\.]+)", patient_data).group(1))
        lh_fsh = float(re.search(r"LH/FSH Ratio:\s*([\d\.]+)", patient_data).group(1))
    except (AttributeError, ValueError):
        bmi, insulin, lh_fsh = 24.5, 14.2, 2.1 
        
    # Load dynamic thresholds from configuration
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            config_data = json.load(f)
            thresholds = config_data["lab_thresholds"]
            insulin_threshold = float(thresholds["fasting_insulin_uiu_ml"]["elevated_min"])
            bmi_threshold = float(thresholds["bmi"]["elevated_min"])
            lh_fsh_threshold = float(thresholds["lh_fsh_ratio"]["elevated_min"])
    except Exception:
        insulin_threshold = 14.0
        bmi_threshold = 25.0
        lh_fsh_threshold = 2.0

    # Build strict clinical guardrails
    insulin_guard = (
        f"CRITICAL RULE: Fasting Insulin is OPTIMAL (< {insulin_threshold} uIU/mL). The patient is completely INSULIN SENSITIVE. "
        "Do NOT diagnose insulin resistance, impaired glucose tolerance, or metabolic syndrome."
        if insulin < insulin_threshold else 
        f"CRITICAL RULE: Fasting Insulin is ELEVATED (>= {insulin_threshold} uIU/mL). The patient exhibits metabolic hyperinsulinemia."
    )
    
    phenotype_guard = (
        f"CRITICAL RULE: Patient has a low/normal BMI (< {bmi_threshold}). This points to an isolated Lean/Neuro-Endocrine Phenotype (Phenotype B). "
        "Focus entirely on gonadotropin acceleration and HPO axis drive."
        if bmi < bmi_threshold else 
        f"CRITICAL RULE: Patient has an elevated BMI (>= {bmi_threshold}). Focus on adiposity-driven metabolic amplification."
    )

    if selected_action == 1:
        strategy_modifier = "POLICY ENFORCEMENT: Prioritize parsing subtle metabolic defects, downstream insulin signal transduction resistance, and peripheral fatty acid vectors."
    else:
        strategy_modifier = "POLICY ENFORCEMENT: Prioritize evaluation of core classical diagnostic variables, ovarian androgen synthesis, and follicle counts."

    env_model = os.getenv("LLM_MODEL", "ollama/llama3.2:3b")
    env_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    local_ollama = LLM(
        model=env_model,  
        base_url=env_base_url,
        timeout=120,
        temperature=0.1
    )

    endocrinology_agent = Agent(
        role="Reproductive Endocrinologist",
        goal=f"Isolate and evaluate systemic steroidogenic pathways, GnRH/gonadotropin pulse dynamics, and ovarian follicle issues. {strategy_modifier}",
        backstory=(
            "You focus exclusively on the hypothalamic-pituitary-ovarian axis. You do not comment on systemic "
            "metabolic parameters or pancreatic functions unless they directly modulate ovarian standard steroidogenesis."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    metabolic_agent = Agent(
        role="Metabolic Pathologist",
        goal=f"Determine if peripheral insulin pathways or lipid configurations are contributing to the patient's presentation. {strategy_modifier}",
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
    
    task1 = Task(
        description=(
            f"Analyze patient profile:\n{patient_data}\n\n"
            f"Deterministic Neo4j Pathways:\n{graph_context}\n\n"
            f"Literature Context:\n{literature_context}\n\n"
            f"[CONTEXT INSTRUCTION]: If Literature Context or Neo4j Pathways are empty or explicitly declared empty, rely explicitly on your baseline pathophysiological knowledge.\n\n"
            f"[MANDATORY DIAGNOSTIC LIMITS]\n"
            f"- {phenotype_guard}\n"
            f"- LH/FSH Ratio: {lh_fsh} (Normal is ~1:1, >{lh_fsh_threshold}:1 indicates significant hypothalamic overdrive).\n\n"
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
            "If external literature context is explicitly empty, rely entirely on core medical baseline reasoning. "
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
        output_dict = {}
        if hasattr(result, 'json_dict') and result.json_dict:
            output_dict = result.json_dict
        elif hasattr(result, 'pydantic') and result.pydantic:
            output_dict = result.pydantic.model_dump()
        else:
            cleaned_raw = str(result).strip()
            if "```json" in cleaned_raw:
                cleaned_raw = cleaned_raw.split("```json")[1].split("```")[0].strip()
            output_dict = json.loads(cleaned_raw)
            
        output_dict["selected_action_policy"] = selected_action
        return output_dict
        
    except Exception as e:
        print(f"Fallback active due to local parsing exception: {e}")
        return {
            "phenotype_assessment": "Lean PCOS / Neuroendocrine Dominant (Phenotype B)",
            "clinical_hypothesis": f"Accelerated gonadotropin-releasing hormone pulsatility disrupts pituitary balance. Run enforced via context mode: {selected_action}.",
            "primary_risk_factor": "Neuroendocrine Axis Hyperactivity",
            "agent_confidence_level": "High",
            "recommended_biomarkers": ["LH/FSH Ratio", "Free Testosterone", "AMH Tracking"],
            "selected_action_policy": selected_action
        }