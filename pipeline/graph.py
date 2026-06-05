# pipeline/graph.py
from langgraph.graph import StateGraph, END
from pipeline.state import PCOSState
from pipeline.nodes.node1_ingestion import node1_ingestion_fn
from pipeline.nodes.node2_hypothesis import node2_hypothesis_fn
from pipeline.nodes.node3_dual_analysis import node3_dual_analysis_fn
from pipeline.nodes.node4_assembly import node_4_assembly_fn  
from pipeline.nodes.node5_xai import node5_xai_fn # Your Node 5 execution function
from pipeline.nodes.node6_rl_ranking import node6_rl_ranking_fn, rl_policy_router_fn
from utils.audit_logger import log_run

def build_pcos_pipeline():
    workflow = StateGraph(PCOSState)
    
    # 1. Map computational execution graph steps
    workflow.add_node("IngestionNode", node1_ingestion_fn)
    workflow.add_node("HypothesisNode", node2_hypothesis_fn)
    workflow.add_node("DualAnalysisNode", node3_dual_analysis_fn)
    workflow.add_node("AssemblyNode", node_4_assembly_fn)
    workflow.add_node("ExplainableAINode", node5_xai_fn)
    workflow.add_node("RLPolicyRankingNode", node6_rl_ranking_fn)
    
    # 2. Sequential structural edges
    workflow.set_entry_point("IngestionNode")
    workflow.add_edge("IngestionNode", "HypothesisNode")
    workflow.add_edge("HypothesisNode", "DualAnalysisNode")
    workflow.add_edge("DualAnalysisNode", "AssemblyNode")
    workflow.add_edge("AssemblyNode", "ExplainableAINode")
    workflow.add_edge("ExplainableAINode", "RLPolicyRankingNode")
    
    # 3. Add dynamic audit tracking before conditional routing
    def run_audit_interceptor(state: PCOSState):
        log_run(state)
        return state
        
    workflow.add_node("AuditInterceptorNode", run_audit_interceptor)
    
    # 4. Bind the Reinforcement Feedback Loop condition
    workflow.add_conditional_edges(
        "RLPolicyRankingNode",
        rl_policy_router_fn,
        {
            "node2_hypothesis": "HypothesisNode",   # Loop back to tune execution prompts
            "__end__": "AuditInterceptorNode"      # Graduate forward if policy is successful
        }
    )
    
    workflow.add_edge("AuditInterceptorNode", END)
    return workflow.compile()