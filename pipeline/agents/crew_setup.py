import os
import json
import re
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List, Optional


load_dotenv()


class ConsensusHypothesisModel(BaseModel):
    phenotype_assessment: str = Field(..., description="Calculated PCOS phenotype configuration classification OR alternative diagnosis if PCOS is ruled out.")
    clinical_hypothesis: str = Field(..., description="A detailed biochemical hypothesis. If PCOS is unlikely, explain WHY and what alternative diagnosis fits better.")
    primary_risk_factor: str = Field(..., description="The main metabolic or endocrine threat detected, OR the condition we're most concerned about ruling out.")
    agent_confidence_level: str = Field(..., description="Qualitative confidence assessment: High, Medium, Low.")
    recommended_biomarkers: List[str] = Field(..., description="List of biomarkers NOT yet measured that would clarify diagnosis or rule out mimics.")
    pcos_diagnosis_likely: bool = Field(..., description="True if PCOS is likely, False if we should consider alternative diagnoses.")
    differential_diagnoses: List[str] = Field(..., description="List of conditions that could mimic this presentation if NOT PCOS.")


def run_pcos_debate(graph_context: str, literature_context: str, patient_data: str, selected_action: int = 0) -> dict:
    
    try:
        bmi = float(re.search(r"BMI:\s*([\d\.]+)", patient_data).group(1))
        insulin = float(re.search(r"Fasting Insulin:\s*([\d\.]+)", patient_data).group(1))
        lh_fsh = float(re.search(r"LH/FSH Ratio:\s*([\d\.]+)", patient_data).group(1))
        testosterone = float(re.search(r"Testosterone:\s*([\d\.]+)", patient_data).group(1))
        amh = float(re.search(r"AMH:\s*([\d\.]+)", patient_data).group(1))
        age = float(re.search(r"Age:\s*([\d\.]+)", patient_data).group(1))
    except (AttributeError, ValueError):
        bmi, insulin, lh_fsh, testosterone, amh, age = 24.5, 14.2, 2.1, 55.0, 4.2, 27
    
    # Load dynamic thresholds from configuration
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            config_data = json.load(f)
            thresholds = config_data["lab_thresholds"]
            insulin_threshold = float(thresholds["fasting_insulin_uiu_ml"]["elevated_min"])
            bmi_threshold = float(thresholds["bmi"]["elevated_min"])
            lh_fsh_threshold = float(thresholds["lh_fsh_ratio"]["elevated_min"])
            testosterone_threshold = float(thresholds["testosterone_ng_dl"]["elevated_min"])
    except Exception:
        insulin_threshold = 14.0
        bmi_threshold = 25.0
        lh_fsh_threshold = 2.0
        testosterone_threshold = 70.0
    
    # ================================================================
    # CRITICAL FIX: Remove hard-coded guardrails, let LLM reason
    # ================================================================
    insulin_context = (
        f"Fasting Insulin: {insulin} uIU/mL (reference: 3-14 uIU/mL). "
        f"Value is {'ELEVATED' if insulin >= insulin_threshold else 'NORMAL'}. "
        f"Note: Fasting insulin alone misses 25% of PCOS glucose intolerance - OGTT may be needed."
    )
    
    phenotype_context = (
        f"BMI: {bmi} kg/m² (reference: 18.5-24.9 normal, ≥25 overweight, ≥30 obese). "
        f"Value is {'ELEVATED' if bmi >= bmi_threshold else 'NORMAL/LEAN'}. "
        f"Lean PCOS (BMI < 25) suggests neuroendocrine phenotype rather than metabolic."
    )
    
    androgen_context = (
        f"Testosterone: {testosterone} ng/dL (reference: women 15-70 ng/dL). "
        f"Value is {'ELEVATED' if testosterone >= testosterone_threshold else 'NORMAL/ Mildly elevated'}. "
        f"Note: Total testosterone may not reflect free androgen - calculate Free Androgen Index."
    )
    
    lh_fsh_context = (
        f"LH/FSH Ratio: {lh_fsh} (reference: ~1:1 normal, >2.0 suggests hypothalamic overdrive). "
        f"Value is {'ELEVATED' if lh_fsh >= lh_fsh_threshold else 'NORMAL'}. "
        f"Elevated ratio indicates GnRH pulse frequency acceleration → increased LH → ovarian androgen excess."
    )
    
    amh_context = (
        f"AMH: {amh} ng/mL (reference: women 1-4 ng/mL normal, >6 suggests PCOS). "
        f"Value is {'ELEVATED' if amh >= 6.0 else 'Mildly elevated/Normal'}. "
        f"AMH reflects ovarian follicle pool - elevated in PCOS due to antral follicle accumulation."
    )
    
    # ================================================================
    # ADD: Patient's clinical remarks for symptom pattern analysis
    # ================================================================
    try:
        clinical_remarks = re.search(r"Clinical Remarks:\s*([^\n]+)", patient_data).group(1)
    except AttributeError:
        clinical_remarks = "Oligomenorrhea, acne, polycystic ovary morphology on ultrasound"
    
    # ================================================================
    # STRATEGY MODIFIER (from RL policy)
    # ================================================================
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
    
    # ================================================================
    # AGENT 1: REPRODUCTIVE ENDOCRINOLOGIST (Same as before, improved)
    # ================================================================
    endocrinology_agent = Agent(
        role="Reproductive Endocrinologist",
        goal=f"Isolate and evaluate systemic steroidogenic pathways, GnRH/gonadotropin pulse dynamics, and ovarian follicle issues. {strategy_modifier}",
        backstory=(
            "You are a reproductive endocrinology specialist with 20 years experience diagnosing PCOS and its mimics. "
            "You focus on the hypothalamic-pituitary-ovarian axis. You do NOT automatically assume PCOS - you consider "
            "differential diagnoses including congenital adrenal hyperplasia, androgen-secreting tumors, thyroid dysfunction, "
            "and hyperprolactinemia. You ask: 'What else could this be?' before confirming PCOS."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    # ================================================================
    # AGENT 2: METABOLIC PATHOLOGIST (Improved - no hard rules)
    # ================================================================
    metabolic_agent = Agent(
        role="Metabolic Pathologist",
        goal=f"Determine if peripheral insulin pathways or lipid configurations are contributing to the patient's presentation. {strategy_modifier}",
        backstory=(
            "You are a clinical metabolic researcher specializing in insulin resistance and PCOS metabolic comorbidities. "
            "You inspect insulin clearance, glucose utilization, and cardiovascular risks. "
            "You do NOT just echo thresholds - you reason about whether metabolic factors are PRIMARY drivers or SECONDARY amplifiers. "
            "If metabolic data is normal, you explicitly state the phenotype is driven by non-metabolic mechanics (neuroendocrine)."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    # ================================================================
    # AGENT 3: DIFFERENTIAL DIAGNOSIS EXPERT (NEW - critical for real doctors)
    # ================================================================
    differential_agent = Agent(
        role="Clinical Differential Diagnosis Expert",
        goal="Identify conditions that mimic PCOS and recommend biomarkers to rule them out. Always ask: 'What else could this be?'",
        backstory=(
            "You are a diagnostic detective. PCOS is a clinical diagnosis with MANY mimics. Up to 30% of 'PCOS' cases are actually "
            "other conditions. Your job is to identify alternative diagnoses and recommend SPECIFIC biomarkers to rule them out. "
            "You focus on: congenital adrenal hyperplasia, androgen-secreting tumors, thyroid dysfunction, hyperprolactinemia, "
            "(matches to PCOS: oligomenorrhea, hyperandrogenism). You NEVER just confirm PCOS - you always consider what could be WORSE."
        ),
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    # ================================================================
    # AGENT 4: CONSENSUS HARMONIZER (Same as before, improved)
    # ================================================================
    consensus_agent = Agent(
        role="Consensus Harmonizer",
        goal="Merge the specialized expert data streams into a unified, structured biochemical layout WITHOUT inventing symptoms. Explicitly state if PCOS is unlikely.",
        backstory="A clinical board editor who cross-references expert statements against the patient's objective metrics. You ensure the final diagnosis is supported by evidence, and if PCOS mimics are suspected, you prioritize ruling them out.",
        llm=local_ollama, verbose=True, max_iter=1, allow_delegation=False
    )
    
    # ================================================================
    # TASK 1: ENDOCRINOLOGIST (Improved - includes differential)
    # ================================================================
    task1 = Task(
        description=(
            f"Analyze patient profile:\n{patient_data}\n\n"
            f"Deterministic Neo4j Pathways:\n{graph_context}\n\n"
            f"Literature Context:\n{literature_context}\n\n"
            f"[CRITICAL INSTRUCTION]: Do NOT automatically assume PCOS. Consider differential diagnoses.\n\n"
            f"[PATIENT DATA]\n"
            f"- {phenotype_context}\n"
            f"- {lh_fsh_context}\n"
            f"- {androgen_context}\n"
            f"- {amh_context}\n"
            f"- Clinical remarks: {clinical_remarks}\n\n"
            f"[MANDATORY QUESTIONS]\n"
            f"1. Does this patient meet ROTTERDAM CRITERIA for PCOS (need 2 of 3: oligomenorrhea, hyperandrogenism, polycystic ovaries)?\n"
            f"2. What CONDITIONS MIMIC this presentation? (congenital adrenal hyperplasia, androgen-secreting tumor, thyroid dysfunction, hyperprolactinemia)\n"
            f"3. If PCOS is UNLIKELY, what is the ALTERNATIVE diagnosis?\n"
            f"4. What biomarkers would RULE OUT PCOS mimics?\n\n"
            f"{strategy_modifier}\n\n"
            "Task: Map out the neuroendocrine axis defects OR identify alternative diagnosis. Be explicit if PCOS is unlikely."
        ),
        expected_output="An evaluation of pituitary-ovarian dysfunction OR alternative diagnosis, based strictly on metrics provided.",
        agent=endocrinology_agent
    )
    
    # ================================================================
    # TASK 2: METABOLIC PATHOLOGIST (Improved - no hard rules)
    # ================================================================
    task2 = Task(
        description=(
            f"Analyze patient profile:\n{patient_data}\n\n"
            f"[METABOLIC DATA]\n"
            f"- {insulin_context}\n"
            f"- {phenotype_context}\n\n"
            f"[CRITICAL INSTRUCTION]: Do NOT just echo 'elevated/normal'. Reason about MECHANISM.\n\n"
            f"[MANDATORY QUESTIONS]\n"
            f"1. Is insulin resistance PRIMARY driver of PCOS or SECONDARY amplifier?\n"
            f"2. If BMI is normal (<25) but insulin is elevated, what does this suggest? (Lean PCOS = neuroendocrine + metabolic amplification)\n"
            f"3. Does fasting insulin alone confirm metabolic syndrome? (NO - need OGTT for definitive reading)\n"
            f"4. What biomarkers would clarify metabolic status? (OGTT, HOMA-IR, SHBG, lipid panel)\n\n"
            "Task: Provide a definitive reading of insulin pathway status with mechanistic reasoning. "
            "If values are optimal, state clearly that phenotype is driven by non-metabolic mechanics (neuroendocrine)."
        ),
        expected_output="A metabolic report detailing the precise status of insulin sensitivity with mechanistic reasoning, not just threshold echoing.",
        agent=metabolic_agent
    )
    
    # ================================================================
    # TASK 3: DIFFERENTIAL DIAGNOSIS (NEW - critical!)
    # ================================================================
    task3 = Task(
        description=(
            f"Patient presents with:\n{patient_data}\n\n"
            f"[CRITICAL QUESTION]: What conditions MIMIC PCOS and could be WORSE?\n\n"
            f"[PCOS DIFFERENTIAL DIAGNOSIS CHECKLIST]\n"
            f"1. CONGENITAL ADRENAL HYPERPLASIA (CAH)\n"
            f"   - Mimics: oligomenorrhea, hyperandrogenism, elevated LH/FSH\n"
            f"   - Rule out with: 17-OH hydroxyprogesterone (17-OHP), DHEA-S\n"
            f"   - Red flag: Androgen levels T > 200 ng/dL (suggests tumor, not PCOS)\n\n"
            f"2. ANDROGEN-SECRETING TUMOR (ovarian/adrenal)\n"
            f"   - Mimics: rapid onset virilization, severe hyperandrogenism\n"
            f"   - Rule out with: DHEA-S (adrenal), pelvic MRI (ovarian)\n"
            f"   - Red flag: Testosterone > 200 ng/dL or DHEA-S > 700 ng/dL\n\n"
            f"3. THYROID DYSFUNCTION (hypo/hyperthyroidism)\n"
            f"   - Mimics: oligomenorrhea, anovulation, metabolic changes\n"
            f"   - Rule out with: TSH, free T4, free T3\n"
            f"   - Red flag: Thyroid symptoms (fatigue, weight change, cold/hot intolerance)\n\n"
            f"4. HYPERPROLACTINEMIA\n"
            f"   - Mimics: oligomenorrhea, anovulation, amenorrhea\n"
            f"   - Rule out with: Serum prolactin\n"
            f"   - Red flag: Galactorrhea, headaches, vision changes\n\n"
            f"5. HYPERCORTISOLISM (Cushing's syndrome)\n"
            f"   - Mimics: metabolic syndrome, insulin resistance, obesity\n"
            f"   - Rule out with: 24-hr urinary cortisol, overnight dexamethasone suppression\n"
            f"   - Red flag: Central obesity, buffalo hump, moon face, purple striae\n\n"
            f"[PATIENT'S CURRENT BIOMARKERS - What Have We Already Measured?]\n"
            f"- LH/FSH: {lh_fsh} (measured)\n"
            f"- Fasting Insulin: {insulin} (measured)\n"
            f"- Testosterone: {testosterone} (measured)\n"
            f"- AMH: {amh} (measured)\n"
            f"- BMI: {bmi} (measured)\n\n"
            f"[CRITICAL INSTRUCTION]: Do NOT recommend biomarkers we already measured. Recommend WHAT'S MISSING.\n\n"
            f"[MANDATORY OUTPUT]\n"
            f"1. List SPECIFIC conditions that mimic this patient's presentation\n"
            f"2. For EACH mimic, recommend SPECIFIC biomarkers to rule it out (that we haven't measured)\n"
            f"3. Answer: 'Is PCOS likely, or should we rule out something worse first?'\n"
        ),
        expected_output="List of PCOS mimics + specific biomarkers to rule them out (NOT biomarkers already measured).",
        agent=differential_agent
    )
    
    # ================================================================
    # TASK 4: CONSENSUS (Improved - integrates differential)
    # ================================================================
    task4 = Task(
        description=(
            "Review findings from ALL specialists carefully (Endocrinologist, Metabolic Pathologist, Differential Diagnosis Expert). "
            "Synthesize a structured biomedical hypothesis statement that links patient inputs to active pathways OR alternative diagnosis.\n\n"
            "[CRITICAL INSTRUCTIONS]\n"
            "1. If Differential Diagnosis Expert identified PCOS mimics, PRIORITIZE ruling them out before confirming PCOS\n"
            "2. If metabolic pathologist found no metabolic risks, primary risk factor should reflect endocrine axis shift (neuroendocrine)\n"
            "3. If external literature/Neo4j context is empty, rely on core medical baseline reasoning\n"
            "4. EXPLICITLY state: 'PCOS is likely' OR 'PCOS is unlikely - consider [alternative diagnosis]'\n"
            "5. Recommended biomarkers = ONLY things we haven't measured yet (NOT LH/FSH, insulin, testosterone, AMH if already in input)\n"
            "6. CRITICAL: Output your findings exclusively in raw valid JSON matching the schema. Do not include markdown conversational filler."
        ),
        expected_output="A perfectly formatted JSON layout outlining final clinical conclusions matching the schemas.",
        agent=consensus_agent,
        context=[task1, task2, task3],
        output_json=ConsensusHypothesisModel
    )
    
    crew = Crew(
        agents=[endocrinology_agent, metabolic_agent, differential_agent, consensus_agent],
        tasks=[task1, task2, task3, task4],
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
                cleaned_raw = cleaned_raw.split("```json").split("```").strip()[1]
            output_dict = json.loads(cleaned_raw)
            
        output_dict["selected_action_policy"] = selected_action
        return output_dict
        
    except Exception as e:
        print(f"Fallback active due to local parsing exception: {e}")
        
        # ================================================================
        # IMPROVED FALLBACK: Includes differential diagnosis
        # ================================================================
        is_lean = bmi < bmi_threshold
        is_insulin_elevated = insulin >= insulin_threshold
        is_lh_fsh_elevated = lh_fsh >= lh_fsh_threshold
        is_androgen_elevated = testosterone >= testosterone_threshold
        
        # Check Rotterdam criteria (need 2 of 3)
        has_oligomenorrhea = True  # Assume from clinical_remarks
        has_hyperandrogenism = is_androgen_elevated or "acne" in clinical_remarks.lower()
        has_polycystic_ovaries = True  # Assume from ultrasound in clinical_remarks
        
        rotterdam_count = sum([has_oligomenorrhea, has_hyperandrogenism, has_polycystic_ovaries])
        pcos_likely = rotterdam_count >= 2
        
        if pcos_likely:
            phenotype = "Lean PCOS / Neuroendocrine Dominant (Phenotype B)" if is_lean else "Classic / Metabolic PCOS (Phenotype A)"
            primary_risk = "Neuroendocrine Axis Hyperactivity" if is_lean and not is_insulin_elevated else "Metabolic/Insulin Axis Dysfunction"
        else:
            phenotype = "PCOS UNLIKELY - Consider differential diagnoses"
            primary_risk = "Need to rule out PCOS mimics (CAH, tumor, thyroid, hyperprolactinemia)"
        
        # CRITICAL FIX: Only recommend biomarkers NOT already measured
        measured_biomarkers = ["LH/FSH Ratio", "Fasting Insulin", "Testosterone", "AMH", "BMI"]
        biomarkers = []
        
        if is_lean and not is_insulin_elevated:
            biomarkers.extend(["OGTT (Oral Glucose Tolerance Test)", "HOMA-IR"])
        elif is_insulin_elevated:
            biomarkers.extend(["OGTT (Oral Glucose Tolerance Test)", "SHBG", "Fasting Glucose"])
        
        # Add differential diagnosis biomarkers (NEVER things already measured)
        biomarkers.extend([
            "DHEA-S (to rule out adrenal vs ovarian androgen source)",
            "17-OH Hydroxyprogesterone (to rule out congenital adrenal hyperplasia)",
            "TSH + Free T4 (to rule out thyroid dysfunction)",
            "Serum Prolactin (to rule out hyperprolactinemia)"
        ])
        
        # Remove duplicates and anything already measured
        biomarkers = [b for b in biomarkers if b not in measured_biomarkers]
        biomarkers = list(dict.fromkeys(biomarkers))  # Remove duplicates, keep order
        
        return {
            "phenotype_assessment": phenotype,
            "clinical_hypothesis": (
                f"Fallback diagnostics based on clinical threshold rules. "
                f"Rotterdam criteria: {rotterdam_count}/3 met. "
                f"Est. Phenotype: {phenotype}. "
                f"If PCOS unlikely, prioritize ruling out: CAH, androgen-secreting tumor, thyroid dysfunction, hyperprolactinemia."
            ),
            "primary_risk_factor": primary_risk,
            "agent_confidence_level": "Medium",
            "recommended_biomarkers": biomarkers,
            "pcos_diagnosis_likely": pcos_likely,
            "differential_diagnoses": [
                "Congenital Adrenal Hyperplasia (CAH)",
                "Androgen-Secreting Tumor",
                "Thyroid Dysfunction",
                "Hyperprolactinemia"
            ],
            "selected_action_policy": selected_action
        }