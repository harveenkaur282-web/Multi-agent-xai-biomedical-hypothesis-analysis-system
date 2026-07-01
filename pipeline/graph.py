from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pipeline.state import PCOSState
from pipeline.nodes.node1_ingestion import node1_ingestion_fn
from pipeline.nodes.node2_hypothesis import node2_hypothesis_fn
from pipeline.nodes.node_judge import node_judge_fn
from pipeline.nodes.node3_dual_analysis import node3_dual_analysis_fn
from pipeline.nodes.node4_assembly import node4_assembly_fn
from pipeline.nodes.node5_xai import node5_xai_fn
from utils.audit_logger import log_run


def build_pcos_pipeline():
    workflow = StateGraph(PCOSState)

    workflow.add_node("IngestionNode", node1_ingestion_fn)
    workflow.add_node("HypothesisNode", node2_hypothesis_fn)
    workflow.add_node("JudgeNode", node_judge_fn)
    workflow.add_node("DualAnalysisNode", node3_dual_analysis_fn)
    workflow.add_node("AssemblyNode", node4_assembly_fn)
    workflow.add_node("ExplainableAINode", node5_xai_fn)

    workflow.set_entry_point("IngestionNode")
    workflow.add_edge("IngestionNode", "HypothesisNode")
    workflow.add_edge("HypothesisNode", "JudgeNode")

    def decide_post_judge(state: PCOSState):
        if state.get("human_approved") is True:
            return "DualAnalysisNode"
        return END

    workflow.add_conditional_edges(
        "JudgeNode",
        decide_post_judge,
        {
            "DualAnalysisNode": "DualAnalysisNode",
            END: END
        }
    )
    workflow.add_edge("DualAnalysisNode", "AssemblyNode")
    workflow.add_edge("AssemblyNode", "ExplainableAINode")

    def run_audit_interceptor(state: PCOSState):
        log_run(state)
        return state

    workflow.add_node("AuditInterceptorNode", run_audit_interceptor)
    workflow.add_edge("ExplainableAINode", "AuditInterceptorNode")
    workflow.add_edge("AuditInterceptorNode", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_after=["JudgeNode"])