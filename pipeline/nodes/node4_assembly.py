from typing import Dict, Any
from pipeline.state import PCOSState


def node4_assembly_fn(state: PCOSState) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("[NODE 4] EXECUTION RUN: SCHEMA LOCK & PAYLOAD VALIDATION")
    print("=" * 60)

    raw_input = state.get("raw_input", {}) or {}
    retrieved_chunks = state.get("retrieved_chunks", []) or []
    graph_knowledge = state.get("graph_knowledge", []) or []
    clinical_hypothesis = state.get("clinical_hypothesis", {}) or {}
    classical_scores = state.get("classical_scores", {}) or {}
    quantum_scores = state.get("quantum_scores", {}) or {}
    ici_metrics = state.get("ici_metrics", {}) or {}
    node3_summary = state.get("node3_summary", {}) or {}

    raw_input_keys = list(raw_input.keys()) if isinstance(raw_input, dict) else []
    hyp_keys = list(clinical_hypothesis.keys()) if isinstance(clinical_hypothesis, dict) else []
    classical_keys = list(classical_scores.keys()) if isinstance(classical_scores, dict) else []
    quantum_keys = list(quantum_scores.keys()) if isinstance(quantum_scores, dict) else []
    node3_keys = list(node3_summary.keys()) if isinstance(node3_summary, dict) else []

    print(f"[Node4 Trace] Raw Input Keys: {raw_input_keys}")
    print(f"[Node4 Trace] Retrieved Chunks: {len(retrieved_chunks)}")
    print(f"[Node4 Trace] Graph Evidence Items: {len(graph_knowledge)}")
    print(f"[Node4 Trace] Clinical Hypothesis Keys: {hyp_keys}")
    print(f"[Node4 Trace] Classical Score Keys: {classical_keys}")
    print(f"[Node4 Trace] Quantum Score Keys: {quantum_keys}")
    print(f"[Node4 Trace] Node3 Summary Keys: {node3_keys}")
    print(f"[Node4 Trace] ICI Score: {ici_metrics.get('integrated_clinical_index', 'N/A')}")

    required_state_keys = [
        "raw_input",
        "retrieved_chunks",
        "graph_knowledge",
        "clinical_hypothesis",
        "classical_scores",
        "quantum_scores",
        "ici_metrics",
        "node3_summary",
    ]
    missing_state_keys = [k for k in required_state_keys if k not in state]
    if missing_state_keys:
        print(f"[LINEAGE WARNING] Missing upstream state keys: {missing_state_keys}")
    else:
        print("[SCHEMA VERIFIED] Upstream payload contract is complete.")

    required_classical = ["bayesian_credibility_score", "confidence_interval_bounds", "interpretation"]
    required_quantum = ["quantum_interaction_score", "raw_counts", "von_neumann_entropy", "qubit_activation", "top_states"]
    missing_classical = [k for k in required_classical if k not in classical_keys]
    missing_quantum = [k for k in required_quantum if k not in quantum_keys]

    if missing_classical:
        print(f"[LINEAGE WARNING] Classical score fields missing: {missing_classical}")
    if missing_quantum:
        print(f"[LINEAGE WARNING] Quantum score fields missing: {missing_quantum}")

    state["xai_metrics"] = state.get("xai_metrics", {}) or {}
    state["xai_report"] = state.get("xai_report", "") or ""

    state["node4_contract"] = {
        "ready_for_xai": len(missing_state_keys) == 0,
        "missing_state_keys": missing_state_keys,
        "missing_classical_fields": missing_classical,
        "missing_quantum_fields": missing_quantum,
    }

    print("=" * 60 + "\n")
    return state