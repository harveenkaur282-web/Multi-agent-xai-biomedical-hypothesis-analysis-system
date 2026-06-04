import json
from datetime import datetime, timezone
from pathlib import Path

# Use Path objects instead of hardcoded string slashes
AUDIT_DIR = Path("data") / "audit_trail"

def log_run(state: dict) -> str:
    # Safely handles directory creation cross-platform
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    
    patient_id = state.get("raw_input", {}).get("patient_id", "unknown")
    
    # CRITICAL FIX FOR 2026: datetime.utcnow() is deprecated! 
    # Use timezone-aware UTC datetime to keep your codebase production-grade.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    filename = f"{patient_id}_{timestamp}.json"
    
    # Securely joins paths with correct system slashes (/ for UNIX, \ for Windows)
    filepath = AUDIT_DIR / filename
    
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
    
    # Open the PosixPath or WindowsPath object directly
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    
    # Convert back to a string for your log print and return
    filepath_str = str(filepath)
    print(f"[Audit] Run logged → {filepath_str}")
    return filepath_str