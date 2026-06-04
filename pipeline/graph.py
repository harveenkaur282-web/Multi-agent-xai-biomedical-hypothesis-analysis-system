from langgraph.graph import StateGraph, END
from pipeline.state import PCOSState
from pipeline.nodes.node1_ingestion import node1_ingestion_fn
from pipeline.nodes.node2_hypothesis import node2_hypothesis_fn
from pipeline.nodes.node3_dual_analysis import node3_dual_analysis_fn
from pipeline.nodes.node4_assembly import node_4_assembly_fn  
from utils.audit_logger import log_run

def build_pcos_pipeline():
    workflow = StateGraph(PCOSState)
    
    # Map computational execution graph steps
    workflow.add_node("IngestionNode", node1_ingestion_fn)
    workflow.add_node("HypothesisNode", node2_hypothesis_fn)
    workflow.add_node("DualAnalysisNode", node3_dual_analysis_fn)
    workflow.add_node("AssemblyNode", node_4_assembly_fn)  # 👈 Add Node 4
    
    # Set entry configuration point
    workflow.set_entry_point("IngestionNode")
    
    # Direct chronological edge tracking layout vectors
    workflow.add_edge("IngestionNode", "HypothesisNode")
    workflow.add_edge("HypothesisNode", "DualAnalysisNode")
    workflow.add_edge("DualAnalysisNode", "AssemblyNode")  # 👈 Route to Node 4
    
    # Intercept output state directly via a clean intermediate step function
    def run_audit_interceptor(state: PCOSState):
        log_run(state) # Saves full system run logs into data/audit_trail/
        return state
        
    workflow.add_node("AuditInterceptorNode", run_audit_interceptor)
    workflow.add_edge("AssemblyNode", "AuditInterceptorNode")  # 👈 Link Node 4 to Interceptor
    workflow.add_edge("AuditInterceptorNode", END)
    
    return workflow.compile()