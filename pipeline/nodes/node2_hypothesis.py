from pipeline.state import PCOSState
from pipeline.agents.crew_setup import run_pcos_debate

def node2_hypothesis_fn(state: PCOSState) -> dict:
    """
    Node 2: Extracts isolated literature and graph streams from state,
    formats a clean clinical narrative brief, and triggers the multi-agent
    CrewAI consensus debate loop.
    """
    chunks = state.get("retrieved_chunks", [])
    literature_papers = [c for c in chunks if isinstance(c, dict) and c.get("is_paper") is True]
    
    literature_context_list = []
    for idx, paper in enumerate(literature_papers):
        literature_context_list.append(
            f"[{idx+1}] Title: {paper.get('title')}\nAbstract: {paper.get('text')}\n"
        )
    literature_context_str = "\n".join(literature_context_list) if literature_context_list else "No literature abstracts retrieved."

    graph_knowledge = state.get("graph_knowledge", [])
    graph_context_list = []
    for edge in graph_knowledge:
        graph_context_list.append(
            f"Edge: ({edge.get('source')}) --[{edge.get('type')}]--> ({edge.get('target')})"
        )
    graph_context_str = "\n".join(graph_context_list) if graph_context_list else "No explicit graph pathways retrieved."

    raw_patient = state.get("raw_input", {})
    patient_data = (
        f"Patient ID: {raw_patient.get('patient_id', 'TEST_CASE_001')}\n"
        f"- Age: {raw_patient.get('age')} years old\n"
        f"- BMI: {raw_patient.get('bmi')}\n"
        f"- LH/FSH Ratio: {raw_patient.get('lh_fsh_ratio')}\n"
        f"- Fasting Insulin: {raw_patient.get('fasting_insulin')} uIU/mL\n"
        f"- AMH Levels: {raw_patient.get('amh_levels')} ng/mL\n"
        f"- Free Testosterone: {raw_patient.get('free_testosterone')} ng/dL\n"
        f"- Narrative Clinical Remarks: {raw_patient.get('clinical_remarks')}"
    )
    

    consensus_out = run_pcos_debate(
        graph_context=graph_context_str,
        literature_context=literature_context_str,
        patient_data=patient_data
    )

    return {"clinical_hypothesis": consensus_out}