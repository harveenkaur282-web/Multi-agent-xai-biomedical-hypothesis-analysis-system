from pipeline.state import PCOSState
from pipeline.agents.crew_setup import run_pcos_debate

def node2_hypothesis_fn(state: PCOSState) -> dict:
    chunks = state["retrieved_chunks"]
    patient_data = str(state["raw_input"])
    context_str = " ".join([f"[{c['title']}: {c['text']}]" for c in chunks])
    
    consensus_out = run_pcos_debate(context_str, patient_data)
    return {"consensus_hypothesis": consensus_out}