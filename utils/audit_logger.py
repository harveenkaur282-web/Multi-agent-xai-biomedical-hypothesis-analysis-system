import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_DIR = Path("data") / "audit_trail"

def log_run(state: dict) -> str:

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    
    patient_id = state.get("raw_input", {}).get("patient_id", "unknown")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    filename = f"{patient_id}_{timestamp}.json"
    filepath = AUDIT_DIR / filename

    payload = {
        "run_metadata": {
            "patient_id": patient_id,
            "timestamp_utc": timestamp,
            "pipeline_version": "1.0.0"
        },
        "input_payload":    state.get("raw_input", {}),
        "node1_chunks":     state.get("retrieved_chunks", []),
        "node2_consensus":  state.get("clinical_hypothesis", {}),
        "node3_classical":  state.get("classical_scores", {}),
        "node3_quantum":    state.get("quantum_scores", {}),
        "node3_ici":        state.get("ici_metrics", {}),
        "node5_xai_metrics":state.get("xai_metrics", {}),
        "node5_xai_report": state.get("xai_report", "")
    }
    
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    filepath_str = str(filepath)
    print(f"[Audit] Run logged → {filepath_str}")
    return filepath_str