from typing import TypedDict, List, Dict, Any

class PCOSState(TypedDict):
    raw_input: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    clinical_hypothesis: str
    classical_scores: Dict[str, Any]
    quantum_scores: Dict[str, Any]