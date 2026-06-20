import json
import re
from datetime import datetime, timezone
from pathlib import Path

AUDIT_DIR = Path("data") / "audit_trail"


def _sanitize_filename(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value or "unknown"


def log_run(state: dict) -> str:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    patient_id = state.get("raw_input", {}).get("patient_id", "unknown")
    safe_patient_id = _sanitize_filename(patient_id)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    iso_timestamp = now.isoformat()

    filename = f"{safe_patient_id}_{timestamp}.json"
    filepath = AUDIT_DIR / filename

    payload = {
        "run_metadata": {
            "patient_id": patient_id,
            "timestamp_utc": iso_timestamp,
            "pipeline_version": "1.0.0",
        },
        "input_payload": state.get("raw_input", {}),
        "node1_chunks": state.get("retrieved_chunks", []),
        "node2_consensus": state.get("clinical_hypothesis", {}),
        "node3_classical": state.get("classical_scores", {}),
        "node3_quantum": state.get("quantum_scores", {}),
        "node3_ici": state.get("ici_metrics", {}),
        "node5_xai_metrics": state.get("xai_metrics", {}),
        "node5_xai_report": state.get("xai_report", ""),
    }

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str, ensure_ascii=False)

    filepath_str = str(filepath)
    print(f"[Audit] Run logged → {filepath_str}")
    return filepath_str