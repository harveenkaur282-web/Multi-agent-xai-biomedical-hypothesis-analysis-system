import json
import os
from datetime import datetime

AUDIT_DIR = "data/audit_trail"

def log_run(state: dict) -> str:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    patient_id = state.get("raw_input", {}).get("patient_id", "unknown")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{patient_id}_{timestamp}.json"
    filepath = os.path.join(AUDIT_DIR, filename)
    
    # Only serialize what exists in state at time of call
    payload = {
        "run_metadata": {
            "patient_id": patient_id,
            "timestamp_utc": timestamp,
            "pipeline_version": "1.0.0"
        },
        "input_payload":    state.get("raw_input", {}),
        "node1_chunks":     state.get("retrieved_chunks", []),
        "node2_consensus":  state.get("consensus_hypothesis", {}),
        "node3_classical":  state.get("classical_scores", {}),
        "node3_quantum":    state.get("quantum_scores", {}),
        "node4_fused":      state.get("fused_result", {})
    }
    
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    
    print(f"[Audit] Run logged → {filepath}")
    return filepath