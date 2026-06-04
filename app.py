# app.py
import streamlit as st
import graphviz
from pipeline.graph import build_pcos_pipeline

st.set_page_config(page_title="PCOS Multi-Agent XAI Dashboard", layout="wide")

st.title("🧬 PCOS Multi-Agent XAI Diagnostics Framework")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# 🔬 EXPANDED PATIENT CLINICAL INTAKE PANEL (SIDEBAR)
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔬 Patient Clinical Intake Panel")

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
    value="Patient presents with severe, treatment-resistant inflammatory acne along the jawline and neck. Reports a history of profound oligomenorrhea, experiencing only 3 irregular menstrual periods in the past 12 months. Transvaginal pelvic ultrasound reveals marked bilateral polycystic ovary morphology with an antral follicle count of 22 on the left and 26 on the right, presenting a classic string-of-pearls arrangement."
)

# Comprehensive data vector matching state models
user_input_case = {
    "age": age,
    "bmi": bmi,
    "family_history": int(family_history),
    "lh_fsh_ratio": lh_fsh_ratio,
    "fasting_insulin": fasting_insulin,
    "amh_levels": amh_levels,
    "free_testosterone": free_testosterone,
    "clinical_remarks": clinical_remarks
}

# Instantiate the compilation engine once at startup
pipeline_executor = build_pcos_pipeline()

# ─────────────────────────────────────────────────────────────────
# 📊 CENTRAL DASHBOARD WORKSPACE
# ─────────────────────────────────────────────────────────────────
st.subheader("Patient Clinical Profile Data Vector")
st.json(user_input_case)

# Trigger Button updates the Session State
if st.button("Trigger Advanced Execution Graph", type="primary"):
    with st.spinner("Processing local multi-agent consensus loops and quantum parameters..."):
        # Invoke the LangGraph pipeline
        output_payload = pipeline_executor.invoke({"raw_input": user_input_case})
        # Stash the payload safely in persistent state cache
        st.session_state["pcos_output_state"] = output_payload
        st.success("Execution Complete!")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# 📑 MULTI-NODE ANALYSIS TABS
# ─────────────────────────────────────────────────────────────
if "pcos_output_state" in st.session_state:
    output_state = st.session_state["pcos_output_state"]
    
    t1, t2, t3, t4 = st.tabs([
        "Node 1: Hybrid Context", 
        "Node 2: Multi-Agent Consensus", 
        "Node 3: Dual Analysis", 
        "Node 4: Payload Assembly"
    ])
    
    # ─────────────────────────────────────────────────────────────
    # TAB 1: KNOWLEDGE GRAPH + LITERATURE ACCUMULATION
    # ─────────────────────────────────────────────────────────────
    with t1:
        kg_relations = output_state.get("graph_knowledge", [])
        if not kg_relations:
            kg_relations = output_state.get("neo4j_subgraph", []) or output_state.get("graph_context", [])
            
        retrieved_payload = output_state.get("retrieved_chunks", [])
        text_papers = [x for x in retrieved_payload if isinstance(x, dict) and x.get("is_paper") is True]
        
        st.markdown("### 🧬 Live Neo4j Knowledge Graph Pathway Extractor")
        
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
            
        st.markdown("### 📚 Retracted Literary Grounding Context Matrices")
        if not text_papers:
            st.caption("No literature abstracts appended to active context layer.")
        else:
            for idx, paper in enumerate(text_papers):
                with st.expander(f"[{idx+1}] {paper.get('title', 'Untitled Abstract')}"):
                    st.write(paper.get('text', 'No text content available.'))

    # ─────────────────────────────────────────────────────────────
    # TAB 2: NODE 2 MULTI-AGENT REASONING OUTPUTS
    # ─────────────────────────────────────────────────────────────
    with t2:
        st.markdown("### Synthesized Clinical Diagnostics Summary")
        agent_data = output_state.get("clinical_hypothesis", {})
        
        if not agent_data:
            st.warning("No hypothesis data returned from the agent assembly.")
        else:
            c1_agent, c2_agent = st.columns([1, 2])
            with c1_agent:
                st.markdown(f"**Phenotype Classification:**\n{agent_data.get('phenotype_assessment', 'N/A')}")
                st.markdown(f"**Primary Threat Vector:**\n{agent_data.get('primary_risk_factor', 'N/A')}")
                st.markdown(f"**Agent Confidence Level:**\n{agent_data.get('agent_confidence_level', 'Medium')}")
                
            with c2_agent:
                st.info("#### Formal Biomedical Hypothesis Statement")
                st.write(agent_data.get("clinical_hypothesis", "No hypothesis generated."))
                
                st.markdown("**Recommended Exploratory Biomarkers:**")
                biomarkers = agent_data.get("recommended_biomarkers", [])
                if isinstance(biomarkers, list):
                    for bio in biomarkers:
                        st.markdown(f"* `{bio}`")
                else:
                    st.write(biomarkers)
        
    # ─────────────────────────────────────────────────────────────
    # TAB 3: NODE 3 ADVANCED SCORING MATRIX
    # ─────────────────────────────────────────────────────────────
    with t3:
        st.markdown("### Algorithmic Evaluation Processing Engines")
        
        classical = output_state.get('classical_scores', {})
        quantum = output_state.get('quantum_scores', {})
        ici_data = output_state.get('ici_metrics', {})
        
        st.markdown("---")
        st.metric(
            label="Integrated Clinical Index (ICI Score)", 
            value=f"{ici_data.get('integrated_clinical_index', 0.0)}",
            delta="Hybrid Ensemble Metric Space Active"
        )
        st.success(f"**Integrated System Recommendation:** {ici_data.get('interpretation', 'N/A')}")
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Classical Statistical Domain")
            st.metric("Bayesian Update Score", f"{classical.get('bayesian_credibility_score', 0.0) * 100:.2f}%")
            st.markdown("**Posterior Credibility Interval Bounds (Beta Distribution):**")
            st.code(str(classical.get('confidence_interval_bounds', [])))
            
            # Access interpretation matching Node 3 dictionary mapping
            st.info(f"**Mathematical Integrity Evaluation:** {classical.get('interpretation', 'N/A')}")
            
        with c2:
            st.markdown("#### Quantum Machine Learning Domain")
            st.metric("Quantum Interaction Score (Simulated)", f"{quantum.get('quantum_interaction_score', 0.0)}")
            st.markdown("**Dominant Sub-circuit Basis Pattern Density:**")
            st.code(str(quantum.get('dominant_state_frequency', 0.0)))
            
            # Access interpretation matching Node 3 dictionary mapping
            st.info(f"**Quantum State Evaluation:** {quantum.get('interpretation', 'N/A')}")
            
        with st.expander("View Raw Circuit Simulated State Dictionary"):
            st.markdown("_Basis string states output generated from Qiskit Aer Statevector execution simulator loops (1024 shots)_")
            st.json(quantum.get('raw_counts', {}))

    # ─────────────────────────────────────────────────────────────
    # TAB 4: NODE 4 PAYLOAD ASSEMBLY & STRUCTURAL INTEGRITY
    # ─────────────────────────────────────────────────────────────
    with t4:
        st.markdown("### 🖥️ Unified Production Schema Verification")
        st.markdown("_This node confirms alignment and safe variable typing before processing the Node 5 XAI algorithms._")
        
        # Verify component presence using dashboard status alerts
        c1_schema, c2_schema, c3_schema = st.columns(3)
        with c1_schema:
            if output_state.get("clinical_hypothesis"):
                st.success("✅ Node 2 Text Payload Linked")
            else:
                st.error("❌ Node 2 Hypothesis Missing")
                
        with c2_schema:
            if output_state.get("classical_scores") and output_state.get("quantum_scores"):
                st.success("✅ Node 3 Numeric Matrices Linked")
            else:
                st.error("❌ Node 3 Matrices Missing")
                
        with c3_schema:
            if output_state.get("ici_metrics"):
                st.success("✅ System Index Record Synchronized")
            else:
                st.warning("⚠️ ICI Metrics Record Unpopulated")
                
        st.markdown("#### Complete Downstream Execution Payload Data Vector")
        # Renders the exact payload that Node 5 will ingest
        st.json({
            "raw_input_shape": list(output_state.get("raw_input", {}).keys()),
            "clinical_hypothesis_keys": list(output_state.get("clinical_hypothesis", {}).keys()),
            "classical_scores_keys": list(output_state.get("classical_scores", {}).keys()),
            "quantum_scores_keys": list(output_state.get("quantum_scores", {}).keys()),
            "ici_metrics_keys": list(output_state.get("ici_metrics", {}).keys())
        })
else:
    st.info("Adjust patient biomarkers on the sidebar and click 'Trigger Advanced Execution Graph' to begin.")

with st.sidebar.expander("View LangGraph Operational Topology", expanded=False):
    try:
        png_bytes = pipeline_executor.get_graph().draw_mermaid_png()
        st.image(png_bytes, caption="Active Operational Pipeline")
    except Exception:
        st.caption("Displaying fallback architectural node activation list:")
        for node_name in pipeline_executor.get_graph().nodes:
            st.code(f"Active Node Layer: {node_name}")