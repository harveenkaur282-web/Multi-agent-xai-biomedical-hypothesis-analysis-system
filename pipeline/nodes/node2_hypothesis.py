# pipeline/nodes/node2_hypothesis.py
import os
import json
import random
from typing import Dict, Any
from pipeline.state import PCOSState
from pipeline.agents.crew_setup import run_pcos_debate

POLICY_FILE = "utils/rl_policy_matrix.json"

def read_rl_action_policy(lh_fsh: float, insulin: float) -> int:
    """Reads the persistent matrix and picks the optimal agent action path (0 or 1)."""
    if not os.path.exists(POLICY_FILE):
        return 0 
        
    with open(POLICY_FILE, "r") as f:
        policy_memory = json.load(f)
        
    q_table = policy_memory["Q_table"]
    
    estimated_state = "LOW_ENTROPY_LOW_VARIANCE"
    if lh_fsh > 2.5 or insulin > 15.0:
        estimated_state = "HIGH_ENTROPY_HIGH_VARIANCE"
        
    state_q_values = q_table.get(estimated_state, [0.5, 0.5])
    
    if random.random() < 0.2:
        print("[RL Engine] 🎲 Action Exploration triggered randomly.")
        return random.choice([0, 1])
    else:
        chosen_act = int(state_q_values.index(max(state_q_values)))
        print(f"[RL Engine] 🧠 Action Exploitation chosen: Path {chosen_act}")
        return chosen_act

def node2_hypothesis_fn(state: PCOSState) -> dict:
    """
    Node 2: Multi-Agent Consensus Layer.
    Implements strict structural length limits to stabilize local inference runtime parameters.
    """
    print("\n" + "="*60)
    print("[NODE 2] INITIALIZING MULTI-AGENT DEBATE TRACKING...")
    print("="*60)

    chunks = state.get("retrieved_chunks", [])
    literature_papers = [c for c in chunks if isinstance(c, dict) and c.get("is_paper") is True]
    
    literature_context_list = []
    for idx, paper in enumerate(literature_papers):
        # Guardrail: Ensure individual abstracts are trimmed before addition 🌟
        abstract_text = paper.get('text', '')
        trimmed_abstract = " ".join(abstract_text.split()[:250]) # Cap abstracts to 250 words
        literature_context_list.append(
            f"[{idx+1}] Title: {paper.get('title')}\nAbstract: {trimmed_abstract}\n"
        )
        
    # Cap total literature feedback strings at the first 4 papers maximum
    literature_context_str = "\n".join(literature_context_list[:4]) if literature_context_list else "No literature abstracts retrieved."

    graph_knowledge = state.get("graph_knowledge", [])
    graph_context_list = []
    for edge in graph_knowledge:
        graph_context_list.append(
            f"Edge: ({edge.get('source')}) --[{edge.get('type')}]--> ({edge.get('target')})"
        )
    # Cap graph paths length to top 15 connection vectors maximum 🌟
    graph_context_str = "\n".join(graph_context_list[:15]) if graph_context_list else "No explicit graph pathways retrieved."

    raw_patient = state.get("raw_input", {})
    lh_fsh_val = float(raw_patient.get("lh_fsh_ratio", 2.1))
    insulin_val = float(raw_patient.get("fasting_insulin", 14.2))

    selected_action = read_rl_action_policy(lh_fsh_val, insulin_val)

    patient_data = (
        f"Patient ID: {raw_patient.get('patient_id', 'TEST_CASE_001')}\n"
        f"- Age: {raw_patient.get('age')} years old\n"
        f"- BMI: {raw_patient.get('bmi')}\n"
        f"- LH/FSH Ratio: {lh_fsh_val}\n"
        f"- Fasting Insulin: {insulin_val} uIU/mL\n"
        f"- AMH Levels: {raw_patient.get('amh_levels')} ng/mL\n"
        f"- Free Testosterone: {raw_patient.get('free_testosterone')} ng/dL\n"
        f"- Narrative Clinical Remarks: {raw_patient.get('clinical_remarks')}"
    )

    # Trigger debate with compact, bounded payload structures
    consensus_out = run_pcos_debate(
        graph_context=graph_context_str,
        literature_context=literature_context_str,
        patient_data=patient_data,
        selected_action=selected_action
    )

    print(f"[NODE 2] Execution finished using Policy Action {selected_action}.")
    print("="*60 + "\n")
    return {"clinical_hypothesis": consensus_out}