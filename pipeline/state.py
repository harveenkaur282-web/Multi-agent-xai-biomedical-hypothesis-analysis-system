from typing import TypedDict, List, Dict, Any


class PCOSState(TypedDict, total=False):
    raw_input: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    graph_knowledge: List[Dict[str, Any]]
    clinical_hypothesis: Dict[str, Any]
    hypotheses: List[Dict[str, Any]]
    classical_scores: Dict[str, Any]
    quantum_scores: Dict[str, Any]
    ici_metrics: Dict[str, Any]
    node3_summary: Dict[str, Any]
    node4_contract: Dict[str, Any]
    xai_metrics: Dict[str, Any]
    xai_report: str