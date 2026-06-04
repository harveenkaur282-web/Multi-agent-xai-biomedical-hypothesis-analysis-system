from typing import TypedDict, List, Dict, Any, Optional

class PCOSState(TypedDict):
    # Input
    raw_input:           Dict[str, Any]

    # Node 1
    retrieved_chunks:    List[Dict[str, Any]]
    graph_knowledge:     List[Dict[str, Any]]   # Neo4j edges

    # Node 2
    clinical_hypothesis: Dict[str, Any]         # consensus agent output
    hypotheses:          List[Dict[str, Any]]   # per-agent outputs for Layer 0

    # Node 3
    classical_scores:    Dict[str, Any]
    quantum_scores:      Dict[str, Any]
    ici_metrics:         Dict[str, Any]

    # Node 5
    xai_metrics:         Dict[str, Any]         # raw numbers for Streamlit charts
    xai_report:          str                    # markdown string for download button