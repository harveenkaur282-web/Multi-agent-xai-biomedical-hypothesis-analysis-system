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
    debate_history = state.get("debate_history", {}) or {}
    llm_judge_evaluation = state.get("llm_judge_evaluation", {}) or {}
    rag_eval_metrics = state.get("rag_eval_metrics", {}) or {}

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
    print(f"[Node4 Trace] Debate History Agents: {list(debate_history.keys())}")
    print(f"[Node4 Trace] LLM Judge Scores: {list(llm_judge_evaluation.get('scores', {}).keys())}")
    print(f"[Node4 Trace] RAG Eval Metrics: {rag_eval_metrics}")

    required_state_keys = [
        "raw_input",
        "retrieved_chunks",
        "graph_knowledge",
        "clinical_hypothesis",
        "classical_scores",
        "quantum_scores",
        "ici_metrics",
        "node3_summary",
        "debate_history",
        "llm_judge_evaluation",
        "rag_eval_metrics",
    ]
    missing_state_keys = [k for k in required_state_keys if k not in state]
    if missing_state_keys:
        print(f"[LINEAGE WARNING] Missing upstream state keys: {missing_state_keys}")
    else:
        print("[SCHEMA VERIFIED] Upstream payload contract is complete.")

    required_classical = ["bayesian_credibility_score", "confidence_interval_bounds", "interpretation"]
    required_quantum = ["quantum_interaction_score", "raw_counts", "von_neumann_entropy", "qubit_activation", "top_states"]
    required_debate = ["reproductive", "metabolic", "differential"]
    missing_classical = [k for k in required_classical if k not in classical_keys]
    missing_quantum = [k for k in required_quantum if k not in quantum_keys]
    missing_debate = [k for k in required_debate if k not in debate_history]

    if missing_classical:
        print(f"[LINEAGE WARNING] Classical score fields missing: {missing_classical}")
    if missing_quantum:
        print(f"[LINEAGE WARNING] Quantum score fields missing: {missing_quantum}")
    if missing_debate:
        print(f"[LINEAGE WARNING] Debate history agents missing: {missing_debate}")

    state["node4_contract"] = {
        "ready_for_xai": len(missing_state_keys) == 0,
        "missing_state_keys": missing_state_keys,
        "missing_classical_fields": missing_classical,
        "missing_quantum_fields": missing_quantum,
        "missing_debate_agents": missing_debate,
        "debate_agents_captured": list(debate_history.keys()),
        "judge_scores_available": bool(llm_judge_evaluation.get("scores")),
        "rag_metrics_available": all(v is not None for v in rag_eval_metrics.values()) if rag_eval_metrics else False,
    }

    print("=" * 60 + "\n")
    return state