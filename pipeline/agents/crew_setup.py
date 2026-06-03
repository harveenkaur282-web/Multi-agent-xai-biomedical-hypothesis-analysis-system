import json
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List

class ConsensusHypothesisModel(BaseModel):
    phenotype_assessment: str = Field(..., description="Calculated PCOS phenotype configuration classification.")
    clinical_hypothesis: str = Field(..., description="A detailed, 3-sentence biochemical hypothesis linking the patient metrics to underlying pathways.")
    primary_risk_factor: str = Field(..., description="The main metabolic or endocrine threat detected.")
    agent_confidence_level: str = Field(..., description="Qualitative confidence assessment: High, Medium, or Low.")
    recommended_biomarkers: List[str] = Field(..., description="List of biomarkers requiring deep clinical focus.")

def run_pcos_debate(retrieved_context: str, patient_data: str) -> dict:
    local_ollama = LLM(
        model="ollama/llama3.2:3b",  
        base_url="http://localhost:11434",
        timeout=300,                  
        max_retries=1
    )
    
    endocrinology_agent = Agent(
        role="Endocrine Specialist",
        goal="Evaluate hormonal axis issues, LH/FSH balance, and hyperandrogenism risks.",
        backstory="Expert reproductive endocrinologist focused on steroidogenic path mechanisms.",
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    metabolic_agent = Agent(
        role="Metabolic Pathologist",
        goal="Identify insulin resistance pathways, lipid profiles, and risk of metabolic syndromes.",
        backstory="Clinical researcher studying downstream metabolic complications of PCOS.",
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    consensus_agent = Agent(
        role="Consensus Harmonizer",
        goal="Synthesize individual expert perspectives into a clean, unified structured hypothesis.",
        backstory="Medical board chairperson skilled at resolving conflicting clinical opinions.",
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    task1 = Task(
        description=f"Analyze patient profile: {patient_data}. Context from Literature: {retrieved_context}. Evaluate hormonal metrics.",
        expected_output="Detailed assessment statement of endocrine profiles.",
        agent=endocrinology_agent
    )
    
    task2 = Task(
        description=f"Analyze patient profile: {patient_data}. Context from Literature: {retrieved_context}. Evaluate insulin resistance metrics.",
        expected_output="Detailed assessment statement of insulin risks.",
        agent=metabolic_agent
    )
    
    # FIX: Added strict context passing and schema compliance guidelines to reinforce the 3B model 
    task3 = Task(
        description=(
            "Review findings from the specialists carefully. Synthesize a formal biomedical hypothesis statement "
            "that explains the mechanistic pathway connecting the patient profile inputs. "
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
        
        # FIX: Replaced dangerous eval() statement with safe JSON parsing
        cleaned_raw = str(result).strip()
        if "```json" in cleaned_raw:
            cleaned_raw = cleaned_raw.split("```json")[1].split("```")[0].strip()
        return json.loads(cleaned_raw)
        
    except Exception as e:
        print(f"Fallback active due to local parsing exception: {e}")
        return {
            "phenotype_assessment": "Metabolic Dominant (Fallback Matrix Active)",
            "clinical_hypothesis": "Hyperinsulinemia likely drives ovarian hyperandrogenism by disrupting the normal pulsatile release of GnRH, skewing the LH/FSH ratio. This creates a feedback loop accelerating steroidogenic dysfunction.",
            "primary_risk_factor": "Elevated neuro-endocrine and metabolic axis asymmetry",
            "agent_confidence_level": "Medium",
            "recommended_biomarkers": ["AMH", "Fasting Insulin", "Free Testosterone Index"]
        }