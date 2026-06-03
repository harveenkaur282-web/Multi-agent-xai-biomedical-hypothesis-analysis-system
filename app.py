import streamlit as st
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
    value="Patient presents with persistent acne and mild hirsutism. Pelvic ultrasound reveals string-of-pearls follicle presentation on right ovary."
)

# Comprehensive data vector matching our updated Node 3 keys
user_input_case = {
    "age": age,
    "bmi": bmi,
    "family_history": int(family_history), # Convert boolean to numeric indicator flag
    "lh_fsh_ratio": lh_fsh_ratio,
    "fasting_insulin": fasting_insulin,
    "amh_levels": amh_levels,
    "free_testosterone": free_testosterone,
    "clinical_remarks": clinical_remarks
}

# ─────────────────────────────────────────────────────────────────
# 📊 CENTRAL DASHBOARD WORKSPACE
# ─────────────────────────────────────────────────────────────────
st.subheader("Patient Clinical Profile Data Vector")
st.json(user_input_case)

if st.button("Trigger Advanced Execution Graph", type="primary"):
    with st.spinner("Processing local multi-agent consensus loops and quantum parameters..."):
        # Invoke the LangGraph pipeline
        pipeline_executor = build_pcos_pipeline()
        output_state = pipeline_executor.invoke({"raw_input": user_input_case})
        
        st.success("Execution Complete!")
        st.markdown("---")
        
        # ─────────────────────────────────────────────────────────────
        # 📑 MULTI-NODE ANALYSIS TABS
        # ─────────────────────────────────────────────────────────────
        t1, t2, t3 = st.tabs(["Node 1: Hybrid Context", "Node 2: Multi-Agent Consensus", "Node 3: Dual Analysis"])
        
        # TAB 1: NODE 1 LIT RETRIEVAL (HYBRID AGENTIC RAG)
        with t1:
            st.markdown("### 📚 Top Retained Reference Abstracts (RAG Engine)")
            retrieved_chunks = output_state.get("retrieved_chunks", [])
            
            if not retrieved_chunks:
                st.warning("No context fragments retrieved. Verify database stream or query parameters.")
            else:
                for idx, item in enumerate(retrieved_chunks):
                    with st.expander(f"**[{idx+1}] {item.get('title', 'PubMed Study')}** (Hybrid Score: {item.get('hybrid_score', 0.0)})"):
                        st.caption(item.get('text', 'No abstract content available.'))
                        
        # TAB 2: NODE 2 AGENT SYNTHESIS & HYPOTHESIS
        with t2:
            st.markdown("### 🧠 Synthesized Clinical Diagnostics Summary")
            agent_data = output_state.get("consensus_hypothesis", {})
            
            if not agent_data:
                st.warning("No hypothesis data returned from the agent assembly.")
            else:
                c1_agent, c2_agent = st.columns([1, 2])
                with c1_agent:
                    st.metric("Phenotype Classification", agent_data.get("phenotype_assessment", "N/A"))
                    st.metric("Primary Threat Vector", agent_data.get("primary_risk_factor", "N/A"))
                    st.metric("Agent Confidence Level", f"{agent_data.get('confidence_score', 0.0) * 100:.1f}%")
                with c2_agent:
                    st.info("#### 📋 Formal Biomedical Hypothesis Statement")
                    st.write(agent_data.get("clinical_hypothesis", "No hypothesis generated."))
                    
                    st.markdown("**🔬 Recommended Exploratory Biomarkers:**")
                    biomarkers = agent_data.get("recommended_biomarkers", [])
                    if isinstance(biomarkers, list):
                        for bio in biomarkers:
                            st.markdown(f"* `{bio}`")
                    else:
                        st.write(biomarkers)
            
        # TAB 3: NODE 3 MATHEMATICAL VALIDATION ENGINES
        with t3:
            st.markdown("### 📊 Algorithmic Evaluation Processing Engines")
            
            classical = output_state.get('classical_scores', {})
            quantum = output_state.get('quantum_scores', {})
            ici_data = output_state.get('ici_metrics', {})
            
            # ──── CRITICAL ADDITION: OVERARCHING ENSEMBLE INDEX HEADER ────
            st.markdown("---")
            st.metric(
                label="🌐 Integrated Clinical Index (ICI Score)", 
                value=f"{ici_data.get('integrated_clinical_index', 0.0)}",
                delta="Hybrid Ensemble Metric Space Active"
            )
            st.success(f"**Integrated System Recommendation:** {ici_data.get('interpretation', 'N/A')}")
            st.markdown("---")
            # ──────────────────────────────────────────────────────────────
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("#### Classical Statistical Domain")
                st.metric("Bayesian Prior Update Score", f"{classical.get('bayesian_credibility_score', 0.0) * 100:.2f}%")
                st.markdown("**Posterior Credibility Interval Bounds (Beta Distribution):**")
                st.code(str(classical.get('confidence_interval_bounds', [])))
                st.info(f"**Mathematical Integrity Evaluation:** {classical.get('interpretation', 'N/A')}")
                
            with c2:
                st.markdown("Quantum Machine Learning Domain")
                st.metric("Quantum Interaction Score (Simulated)", f"{quantum.get('quantum_interaction_score', 0.0)}")
                st.markdown("**Dominant Sub-circuit Basis Pattern Density:**")
                st.code(str(quantum.get('dominant_state_frequency', 0.0)))
                st.info(f"**Quantum State Evaluation:** {quantum.get('interpretation', 'N/A')}")
                
            with st.expander("View Raw Circuit Simulated State Dictionary"):
                st.markdown("_Basis string states output generated from Qiskit Aer Statevector execution simulator loops (1024 shots)_")
                st.json(quantum.get('raw_counts', {}))

            # Inside app.py

            with st.sidebar.expander("📍 View LangGraph Execution Nodes", expanded=False):
                pipeline_graph = build_pcos_pipeline()
    
            try:
                # Generate raw PNG bytes using the graphviz backend built into LangGraph
                png_bytes = pipeline_graph.get_graph().draw_mermaid_png()
                st.image(png_bytes, caption="Active LangGraph Operational Topology Matrix")
            except Exception as e:
                st.caption("Graphviz system binaries missing. Displaying structural node list fallback instead:")
            # Fallback list showing nodes and sequential links clearly
            for node_name in pipeline_graph.get_graph().nodes:
                st.code(f"Node Layer Active: {node_name}")