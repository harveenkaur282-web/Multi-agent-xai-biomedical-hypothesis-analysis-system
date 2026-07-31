import streamlit as st
import graphviz
import pandas as pd
import requests
import os
try:
    from langgraph.errors import GraphInterrupt
except Exception:
    class GraphInterrupt(Exception):
        """Fallback for LangGraph versions without GraphInterrupt."""
        pass
from pipeline.graph import build_pcos_pipeline
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="PCOS Multi-Agent XAI Dashboard", layout="wide")

st.title("PCOS Multi-Agent XAI Diagnostics Framework")
st.markdown("---")

st.sidebar.header("Patient Clinical Intake Panel")

with st.sidebar.expander("Patient Demographics & History", expanded=True):
    age = st.sidebar.slider("Age", 18, 45, 27)
    bmi = st.sidebar.number_input("Calculated BMI", value=24.5, step=0.1)
    family_history = st.sidebar.checkbox("Family History of PCOS / Type-2 Diabetes")

with st.sidebar.expander("Endocrine & Metabolic Serology", expanded=True):
    lh_fsh_ratio = st.sidebar.slider("LH / FSH Ratio", 0.5, 4.5, 2.1)
    fasting_insulin = st.sidebar.slider("Fasting Insulin (uIU/mL)", 2.0, 30.0, 14.2)
    amh_levels = st.sidebar.number_input("Anti-Müllerian Hormone (AMH) ng/mL", value=4.2, step=0.1)
    free_testosterone = st.sidebar.number_input("Total/Free Testosterone (ng/dL)", value=55.0, step=1.0)

st.sidebar.markdown("**Unstructured Narrative Medical Notes**")
clinical_remarks = st.sidebar.text_area(
    "Enter physical presentation or ultrasound observations:",
    value="Patient presents with severe, treatment-resistant inflammatory acne along the jawline and neck. Reports a history of profound oligomenorrhea, experiencing only 3 irregular menstrual periods in the past 12 months. Transvaginal pelvic ultrasound reveals marked bilateral polycystic ovary morphology with antral follicle counts consistent with a classic string-of-pearls arrangement."
)

# Ingestion configurations
use_live_search = st.sidebar.checkbox("Use Live PubMed Search", value=False)

with st.sidebar.expander("LLM Engine Configuration", expanded=True):
    llm_choice = st.selectbox("CrewAI LLM Backend", ["Ollama Local", "Groq API"], index=1 if os.getenv("GROQ_API_KEY") else 0)
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if llm_choice == "Groq API" and not groq_api_key:
        groq_api_key = st.text_input("Enter Groq API Key", type="password")

# FIXED: Explicitly normalized map schemas to eliminate upstream node-to-state key tracking drops
user_input_case = {
    "age": age,
    "bmi": bmi,
    "family_history": int(family_history),
    "lh_fsh_ratio": lh_fsh_ratio,
    "fasting_insulin": fasting_insulin,
    "fasting_insulin_uiu_ml": fasting_insulin,  # Backwards compatibility map for node3 parser
    "amh_levels": amh_levels,
    "amh_ng_ml": amh_levels,                    # Backwards compatibility map for node3 parser
    "free_testosterone": free_testosterone,
    "testosterone_ng_dl": free_testosterone,    # Lineage link map for node3 threshold lookups
    "testosterone": free_testosterone,          # Fallback backup link for node5 metrics calculation
    "clinical_remarks": clinical_remarks,
    "llm_choice": llm_choice,
    "groq_api_key": groq_api_key,
    "use_live_search": use_live_search,
}

@st.cache_resource
def get_pipeline():
    return build_pcos_pipeline()

pipeline_executor = get_pipeline()
config = {"configurable": {"thread_id": "pcos_default_thread"}}

# Allow clearing cached pipeline when backend code changes
if st.sidebar.button("🔄 Reset Pipeline Cache"):
    st.cache_resource.clear()
    if "pcos_output_state" in st.session_state:
        del st.session_state["pcos_output_state"]
    st.rerun()

with st.sidebar.expander("View Patient Clinical Intake Data JSON Vector", expanded=False):
    # Filter out sensitive/internal keys before displaying to the user
    _sensitive_keys = {"groq_api_key", "llm_choice"}
    display_case = {k: v for k, v in user_input_case.items() if k not in _sensitive_keys}
    st.json(display_case)

if st.button("Trigger Advanced Execution Graph", type="primary"):
    if llm_choice == "Groq API":
        active_groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not active_groq_key:
            st.error(" CRITICAL ERROR: GROQ_API_KEY is missing. Please provide it in your .env or sidebar.")
            st.stop()
        # Pre-flight handshake check to confirm Groq cloud connectivity
        try:
            ping_url = "https://api.groq.com/openai/v1/models"
            headers = {"Authorization": f"Bearer {active_groq_key}"}
            req = requests.get(ping_url, headers=headers, timeout=5)
            if req.status_code != 200:
                st.error(f" Groq API Authorization Failed (HTTP {req.status_code}). Verify your Groq API key.")
                st.stop()
        except requests.exceptions.RequestException:
            st.warning(" Connection Error: Unable to access Groq API gateway. Check internet connectivity.")
            st.stop()
    else:
        # Pre-flight handshake for Ollama Local
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            req = requests.get(ollama_url, timeout=3)
            if req.status_code != 200:
                st.error(f" Ollama Local Server responded with code {req.status_code} at {ollama_url}. Make sure Ollama is running.")
                st.stop()
        except requests.exceptions.RequestException:
            st.error(f" Connection Error: Unable to reach Ollama Local Server at {ollama_url}. Make sure Ollama is started.")
            st.stop()

    # Execute downstream architecture graph once context pathing validates cleanly
    with st.spinner("Processing multi-agent consensus loops and quantum parameters..."):
        try:
            st.session_state["pcos_output_state"] = None
            st.session_state["pcos_approval_pending"] = False
            output_payload = pipeline_executor.invoke(
                {
                    "raw_input": user_input_case, 
                    "human_approved": None, 
                    "human_feedback": None
                }, 
                config=config
            )
            st.session_state["pcos_output_state"] = output_payload
            state_info = pipeline_executor.get_state(config)
            needs_approval = (
                isinstance(output_payload, dict)
                and output_payload.get("human_approved") is None
                and bool(output_payload.get("llm_judge_evaluation"))
            )
            st.session_state["pcos_approval_pending"] = needs_approval
            if needs_approval:
                st.info("ℹ Pipeline reached the Judge checkpoint. Please review the hypothesis below and approve to continue to Node 3, 4, and 5.")
            else:
                st.success(" Execution Complete!")
            st.rerun()
        except GraphInterrupt:
            state_info = pipeline_executor.get_state(config)
            st.session_state["pcos_output_state"] = state_info.values or {}
            st.session_state["pcos_approval_pending"] = True
            st.info("ℹ Pipeline paused at the Judge checkpoint. Please review the hypothesis below and approve to continue.")
            st.rerun()
        except Exception as e:
            st.error(f" Pipeline Execution Interrupted: {str(e)}")

# Get current state from memory saver checkpointer
state_info = pipeline_executor.get_state(config)
output_state = st.session_state.get("pcos_output_state")
needs_approval = bool(st.session_state.get("pcos_approval_pending", False)) or (
    isinstance(output_state, dict)
    and output_state.get("human_approved") is None
    and bool(output_state.get("llm_judge_evaluation"))
)

if needs_approval:
    st.markdown("### Clinical Verification Panel (Human-in-the-Loop)")
    st.warning("**Awaiting Clinical Review**: The pipeline has reached the verification checkpoint. Please evaluate the agent consensus hypothesis below before authorizing statistical/quantum modeling.")
    
    current_vals = state_info.values or {}
    hypothesis_data = current_vals.get("clinical_hypothesis") or {}
    
    h_col1, h_col2 = st.columns([1, 2])
    with h_col1:
        st.markdown("#### Phenotypic Attributes")
        st.markdown(f"**Classification:** `{hypothesis_data.get('phenotype_assessment', 'N/A')}`")
        st.markdown(f"**Primary Pathway:** `{hypothesis_data.get('primary_risk_factor', 'N/A')}`")
        st.markdown(f"**Agent Confidence:** `{hypothesis_data.get('agent_confidence_level', 'Medium')}`")
    with h_col2:
        st.markdown("#### Formal Biomedical Hypothesis Statement")
        st.info(hypothesis_data.get("clinical_hypothesis", "No hypothesis generated."))
        st.markdown("**Recommended Exploratory Biomarkers (Rule-outs):**")
        biomarkers = hypothesis_data.get("recommended_biomarkers", [])
        if isinstance(biomarkers, list):
            st.markdown(", ".join([f"`{b}`" for b in biomarkers]))
                
    st.markdown("#### LLM-as-a-Judge Evaluation & RAG Faithfulness Metrics")
    judge_eval = current_vals.get("llm_judge_evaluation") or {}
    table_md = judge_eval.get("table_markdown", "")
    if table_md:
        st.markdown(table_md)
    st.markdown("---")
    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button("Approve and Continue Pipeline", type="primary", use_container_width=True):
            with st.spinner("Resuming graph execution..."):
                pipeline_executor.update_state(config, {"human_approved": True}, as_node="JudgeNode")
                output_payload = pipeline_executor.invoke(None, config=config)
                st.session_state["pcos_output_state"] = output_payload
                st.session_state["pcos_approval_pending"] = False
                st.success("Pipeline Resumed and Completed!")
                st.rerun()
    with reject_col:
        rejection_feedback = st.text_area("Provide feedback for rejection:", placeholder="Explain why this hypothesis is inaccurate or what needs adjustment...")
        if st.button("Reject and Terminate", type="secondary", use_container_width=True):
            if not rejection_feedback.strip():
                st.error("Please enter feedback before rejecting.")
            else:
                with st.spinner("Terminating pipeline..."):
                    pipeline_executor.update_state(config, {
                        "human_approved": False,
                        "human_feedback": rejection_feedback
                    }, as_node="JudgeNode")
                    output_payload = pipeline_executor.invoke(None, config=config)
                    st.session_state["pcos_output_state"] = output_payload
                    st.session_state["pcos_approval_pending"] = False
                    st.error("Pipeline Terminated: Feedback Saved.")
                    st.rerun()

st.markdown("---")

paused_at_judge = bool(state_info.next)
completed_state_ready = "pcos_output_state" in st.session_state and st.session_state.get("pcos_output_state") is not None
output_state = st.session_state.get("pcos_output_state") if completed_state_ready else None
approval_pending = bool(st.session_state.get("pcos_approval_pending", False)) or (
    isinstance(output_state, dict)
    and output_state.get("human_approved") is None
    and bool(output_state.get("llm_judge_evaluation"))
)

if approval_pending:
    st.info(" Pipeline is paused at the Judge verification step. Approve or reject the hypothesis above to continue to Node 3 and beyond.")

elif not completed_state_ready:
    st.info("Adjust patient biomarkers on the sidebar and click 'Trigger Advanced Execution Graph' to begin. After the hypothesis is generated, approve to continue to Node 3, 4, and 5.")

else:
    if output_state.get("human_approved") is False:
        st.error(" **Pipeline Terminated**: The clinician rejected the hypothesis.")
        st.info(f"**Clinician Feedback Recorded:** *{output_state.get('human_feedback')}*")
        st.warning("Pipeline terminated. Clinician feedback has been recorded and will inform future iterations.")
    else:
        t1, t2, t3, t4, t5 = st.tabs([
            "Node 1: Hybrid Context",
            "Node 2: Multi-Agent Consensus",
            "Node 3: Dual Analysis",
            "Node 4: Payload Assembly",
            "Node 5: Explainable AI Engine",
        ])

        with t1:
            kg_relations = output_state.get("graph_knowledge") or []
            if not kg_relations:
                kg_relations = output_state.get("neo4j_subgraph") or output_state.get("graph_context") or []

            retrieved_payload = output_state.get("retrieved_chunks") or []
            text_papers = [x for x in retrieved_payload if isinstance(x, dict) and x.get("is_paper") is True]

            st.markdown("### Live Neo4j Knowledge Graph Pathway Extractor")

            if not kg_relations:
                st.info("No matching structural database pathways linked to current parameters.")
            else:
                dot = graphviz.Digraph(comment='Live Production Medical Matrix')
                dot.attr(bgcolor='#0E1117', rankdir='LR')

                for rel in kg_relations:
                    src = rel.get("source", "Unknown")
                    tgt = rel.get("target", "Unknown")
                    edge_type = rel.get("type", "ASSOCIATED_WITH")

                    dot.node(src, f"🟢 {src}", color='#00F0FF', fontcolor='white')
                    dot.node(tgt, f"🟡 {tgt}", color='#FFD700', fontcolor='white')
                    dot.edge(src, tgt, label=edge_type.lower(), color='#555555', fontcolor='#AAAAAA')

                st.graphviz_chart(dot)

            st.markdown("### Retrieved Literary Grounding Context Matrices")
            if not text_papers:
                st.caption("No literature abstracts appended to active context layer.")
            else:
                for idx, paper in enumerate(text_papers):
                    with st.expander(f"[{idx+1}] {paper.get('title', 'Untitled Abstract')}"):
                        st.write(paper.get('text', 'No text content available.'))

        with t2:
            st.markdown("### Synthesized Clinical Diagnostics Summary")
            agent_data = output_state.get("clinical_hypothesis") or {}

            if not agent_data:
                st.warning("No hypothesis data returned from the agent assembly.")
            else:
                c1_agent, c2_agent = st.columns([1, 2])
                with c1_agent:
                    st.markdown("#### Phenotype Classification")
                    st.success(f"**{agent_data.get('phenotype_assessment', 'N/A')}**")
                    st.markdown(f"**Primary Risk Factor:** `{agent_data.get('primary_risk_factor', 'N/A')}`")
                    st.markdown(f"**Agent Confidence:** `{agent_data.get('agent_confidence_level', 'Medium')}`")

                with c2_agent:
                    st.markdown("#### Formal Clinical Hypothesis")
                    st.info(agent_data.get("clinical_hypothesis", "No hypothesis generated."))

                    st.markdown("**Recommended Exploratory Biomarkers (Rule-outs):**")
                    biomarkers = agent_data.get("recommended_biomarkers", [])
                    if isinstance(biomarkers, list):
                        st.markdown(", ".join([f"`{b}`" for b in biomarkers]))
                    else:
                        st.write(biomarkers)

                st.markdown("---")
                st.markdown("### LLM-as-a-Judge & RAG Quantitative Evaluation Metrics")
                judge_eval = output_state.get("llm_judge_evaluation") or {}
                table_md = judge_eval.get("table_markdown", "")
                if table_md:
                    st.markdown(table_md)
                else:
                    st.info("No judge evaluations computed yet.")

        with t3:
            st.markdown("### Algorithmic Evaluation Processing Engines")
            classical = output_state.get("classical_scores") or {}
            quantum = output_state.get("quantum_scores") or {}
            ici_data = output_state.get("ici_metrics") or {}
            node3_data = output_state.get("node3_summary") or {}

            st.markdown("---")
            ici_score_raw = ici_data.get('integrated_clinical_index', 0.0)
            bayes_score = classical.get('bayesian_credibility_score', 0.0)
            q_score = quantum.get('quantum_interaction_score', 0.0)
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(
                    label="Integrated Clinical Index (ICI Score)",
                    value=f"{ici_score_raw:.4f}" if isinstance(ici_score_raw, float) else str(ici_score_raw),
                    delta="Integrated Core Index"
                )
            with m_col2:
                st.metric(
                    label="Bayesian Credibility", 
                    value=f"{bayes_score * 100:.2f}%" if isinstance(bayes_score, float) else str(bayes_score),
                    delta="Classical Support"
                )
            with m_col3:
                st.metric(
                    label="Quantum Interaction Score", 
                    value=f"{q_score:.4f}" if isinstance(q_score, float) else str(q_score),
                    delta="Non-linear Coupling"
                )
                
            st.info(f"💡 **Integrated System Recommendation:** {ici_data.get('interpretation', 'N/A')}")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Classical Statistical Domain")
                st.markdown(f"**Bayesian Posterior Credibility:** `{bayes_score * 100:.2f}%`")
                st.markdown("**Posterior Credibility Interval Bounds:**")
                st.code(str(classical.get('confidence_interval_bounds', [])))
                st.success(f"**Classical Model Interpretation:** {classical.get('interpretation', 'N/A')}")

            with c2:
                st.markdown("#### Quantum Machine Learning Domain")
                st.markdown(f"**Von Neumann Entropy:** `{quantum.get('von_neumann_entropy', 0.0):.4f}`")
                st.markdown(f"**Dominant State Frequency:** `{quantum.get('dominant_state_frequency', 0.0):.4f}`")
                st.success(f"**Quantum State Evaluation:** {quantum.get('interpretation', 'N/A')}")

            st.markdown("---")
            st.markdown("#### Live Quantum State Amplitude Spectrum")
            raw_counts = quantum.get('raw_counts', {})
            if raw_counts:
                df_quantum = pd.DataFrame({
                    "Quantum Computational Basis State (|ψ⟩)": [f"|{k}⟩" for k in raw_counts.keys()],
                    "Measurement Shot Frequency": list(raw_counts.values())
                })
                st.bar_chart(
                    data=df_quantum,
                    x="Quantum Computational Basis State (|ψ⟩)",
                    y="Measurement Shot Frequency",
                    use_container_width=True
                )
            else:
                st.info("No quantum state count distribution available.")

            with st.expander("🛠️ View QML Simulation Details & Raw Payload State"):
                st.markdown("**Node 3 Summary:**")
                st.json(node3_data)
                st.markdown("**Raw Circuit State counts:**")
                st.json(quantum.get('raw_counts', {}))

        with t4:
            st.markdown("### Unified Production Schema Verification")
            st.markdown("_This node confirms alignment and safe variable typing before processing the Node 5 Argumentation XAI engine._")

            contract = output_state.get("node4_contract") or {}
            
            ready = contract.get("ready_for_xai", False)
            if ready:
                st.success("Upstream payload contract is fully complete. All data streams validated.")
            else:
                st.warning("Payload contract is missing some optional or non-critical diagnostic components.")
                
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.markdown("**State Checkpoints:**")
                status_keys = "[OK]" if not contract.get('missing_state_keys') else "[FAIL]"
                st.markdown(f"{status_keys} State Keys completeness")
                status_judge = "[OK]" if contract.get('judge_scores_available') else "[FAIL]"
                st.markdown(f"{status_judge} LLM Judge Evaluations")
                status_rag = "[OK]" if contract.get('rag_metrics_available') else "[FAIL]"
                st.markdown(f"{status_rag} RAG Metrics availability")
            with c_col2:
                st.markdown("**Domain Metrics:**")
                status_classical = "[OK]" if not contract.get('missing_classical_fields') else "[FAIL]"
                st.markdown(f"{status_classical} Classical Statistical Score vectors")
                status_quantum = "[OK]" if not contract.get('missing_quantum_fields') else "[FAIL]"
                st.markdown(f"{status_quantum} QML Simulation metrics")
                status_debate = "[OK]" if not contract.get('missing_debate_agents') else "[FAIL]"
                st.markdown(f"{status_debate} Multi-Agent Debate transcripts")

            with st.expander("View Raw Schema Contract Verification payload"):
                st.json(contract if contract else {"status": "missing"})

                st.markdown("#### Complete Downstream Execution Payload Data Vector")
                st.json({
                    "raw_input_shape": list((output_state.get("raw_input") or {}).keys()),
                    "clinical_hypothesis_keys": list((output_state.get("clinical_hypothesis") or {}).keys()),
                    "classical_scores_keys": list((output_state.get("classical_scores") or {}).keys()),
                    "quantum_scores_keys": list((output_state.get("quantum_scores") or {}).keys()),
                    "ici_metrics_keys": list((output_state.get("ici_metrics") or {}).keys()),
                    "node3_summary_keys": list((output_state.get("node3_summary") or {}).keys()),
                    "debate_history_agents": list((output_state.get("debate_history") or {}).keys()),
                    "judge_evaluation_keys": list((output_state.get("llm_judge_evaluation") or {}).keys()),
                })

        with t5:
            st.markdown("### Explainable AI Engine (ArgMed-Agents Framework)")
            xai_metrics = output_state.get("xai_metrics") or {}
            xai_report = output_state.get("xai_report") or ""

            if not xai_metrics:
                st.warning("Explainability interpretation data missing from pipeline state context.")
            else:
                col_graph, col_citations = st.columns([3, 2])
                
                with col_graph:
                    st.markdown("#### Clinical Argumentation Graph")
                    st.caption("ArgMed-Agents: Abstract Argumentation Scheme (Green: Accepted, Red: Defeated, Gray: Out)")
                    
                    arg_data = xai_metrics.get("arguments", {})
                    arg_edges = xai_metrics.get("graph_edges", [])

                    if arg_data:
                        arg_dot = graphviz.Digraph(comment="Clinical Argumentation Graph")
                        arg_dot.attr(bgcolor="#0E1117", rankdir="TB", fontname="Helvetica")
                        arg_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", fontsize="11")

                        status_colors = {
                            "ACCEPTED": {"fillcolor": "#d4edda", "fontcolor": "#155724", "color": "#28a745"},
                            "DEFEATED": {"fillcolor": "#f8d7da", "fontcolor": "#721c24", "color": "#dc3545"},
                            "OUT": {"fillcolor": "#e2e3e5", "fontcolor": "#383d41", "color": "#6c757d"},
                        }

                        for arg_id, arg in arg_data.items():
                            status = arg.get("status", "OUT")
                            colors = status_colors.get(status, status_colors["OUT"])
                            label = f"{arg_id}: {arg.get('title', '')}\n[{status}]"
                            arg_dot.node(arg_id, label, **colors)

                        for edge in arg_edges:
                            src = edge.get("source", "")
                            tgt = edge.get("target", "")
                            rel = edge.get("relation", "SUPPORTS")
                            if rel == "ATTACKS":
                                arg_dot.edge(src, tgt, label="ATTACKS", color="#dc3545", fontcolor="#dc3545", style="dashed", penwidth="2.0")
                            elif rel == "RESOLVES":
                                arg_dot.edge(src, tgt, label="RESOLVES", color="#007bff", fontcolor="#007bff", style="bold", penwidth="2.0")
                            else:
                                arg_dot.edge(src, tgt, label=rel, color="#6c757d", fontcolor="#AAAAAA")

                        st.graphviz_chart(arg_dot)
                    else:
                        st.info("No argumentation data available.")

                with col_citations:
                    st.markdown("#### Evidence Grounding Matrix")
                    st.caption("Frequently referenced literary chunks in consensus reasoning")
                    citation_map = xai_metrics.get("citations", {})
                    if citation_map:
                        df_citations = pd.DataFrame({
                            "Chunk Citation ID": list(citation_map.keys()),
                            "Reference Count": list(citation_map.values())
                        }).sort_values(by="Reference Count", ascending=False)
                        st.dataframe(df_citations, use_container_width=True)
                    else:
                        st.caption("No literature citations were referenced during the debate.")

                st.markdown("---")
                
                col_details, col_biomarkers = st.columns([1, 1])
                
                with col_details:
                    st.markdown("#### Argument Evaluation Details")
                    if arg_data:
                        for arg_id, arg in arg_data.items():
                            status = arg.get("status", "OUT")
                            icon = "[ACCEPTED]" if status == "ACCEPTED" else ("[DEFEATED]" if status == "DEFEATED" else "[OUT]")
                            with st.expander(f"{icon} {arg_id}: {arg.get('title', 'Unknown')} — [{status}]"):
                                st.markdown(f"**Claim:** {arg.get('claim', 'N/A')}")
                                st.markdown(f"**Evidence:** {arg.get('evidence', 'N/A')}")
                
                with col_biomarkers:
                    st.markdown("#### Patient Biomarker Input Vector")
                    st.caption("Visual representation of biomedical feature magnitudes")
                    shap_data = xai_metrics.get("shap_importance_vectors", {})
                    if shap_data:
                        df_bio = pd.DataFrame({
                            "Biomedical Axis": list(shap_data.keys()),
                            "Patient Value": list(shap_data.values())
                        }).sort_values(by="Patient Value", ascending=True)

                        st.bar_chart(
                            data=df_bio,
                            x="Biomedical Axis",
                            y="Patient Value",
                            use_container_width=True
                        )

                st.markdown("---")

                with st.expander("View Detailed Multi-Agent Debate Transcripts", expanded=False):
                    debate_history = output_state.get("debate_history") or {}
                    if debate_history:
                        rep_trans = debate_history.get("reproductive", "No reproductive output generated.")
                        met_trans = debate_history.get("metabolic", "No metabolic output generated.")
                        diff_trans = debate_history.get("differential", "No differential output generated.")
                        
                        sub_t1, sub_t2, sub_t3 = st.tabs(["Reproductive Agent", "Metabolic Agent", "Differential Agent"])
                        with sub_t1:
                            st.text(rep_trans)
                        with sub_t2:
                            st.text(met_trans)
                        with sub_t3:
                            st.text(diff_trans)
                    else:
                        st.info("No debate transcripts captured from the multi-agent consensus.")

                st.markdown("---")

                if xai_report:
                    st.markdown("#### Final Diagnostic & Explainability Dossier")
                    st.markdown(xai_report)
                    st.download_button(
                        label="Download Clinical Argumentation & Explainability Report (.md)",
                        data=xai_report,
                        file_name="PCOS_ArgMed_XAI_Dossier.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.caption("No markdown report string detected inside state context.")

@st.cache_data
def get_graph_image():
    return pipeline_executor.get_graph().draw_mermaid_png()

with st.sidebar.expander("View LangGraph Operational Topology", expanded=False):
    try:
        png_bytes = get_graph_image()
        st.image(png_bytes, caption="Active Operational Pipeline")
    except Exception:
        st.caption("Displaying fallback architectural node activation list:")
        for node_name in pipeline_executor.get_graph().nodes:
            st.code(f"Active Node Layer: {node_name}")