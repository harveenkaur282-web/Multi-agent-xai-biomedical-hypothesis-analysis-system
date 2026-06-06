from typing import TypedDict, List, Dict, Any, Optional

class PCOSState(TypedDict):
    raw_input:           Dict[str, Any]
    retrieved_chunks:    List[Dict[str, Any]]
    graph_knowledge:     List[Dict[str, Any]]   
    clinical_hypothesis: Dict[str, Any]         
    hypotheses:          List[Dict[str, Any]]   
    classical_scores:    Dict[str, Any]
    quantum_scores:      Dict[str, Any]
    ici_metrics:         Dict[str, Any]
    xai_metrics:         Dict[str, Any]        
    xai_report:          str                  
    rl_policy_metadata:  Dict[str, Any]