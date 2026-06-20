import numpy as np
if not hasattr(np, "int"):
    np.int = int

from typing import Dict, Any, List
import shap
from sklearn.linear_model import LogisticRegression
from pipeline.state import PCOSState

_RNG = np.random.default_rng(42)
_N_SYNTH = 250

def _build_threshold_surrogate() -> LogisticRegression:
    X_healthy = np.column_stack([
        _RNG.uniform(0.5, 2.0, _N_SYNTH),
        _RNG.uniform(2.0, 13.9, _N_SYNTH),
        _RNG.uniform(10.0, 44.0, _N_SYNTH), # Aligned to updated free testosterone threshold
        _RNG.uniform(16.0, 24.9, _N_SYNTH),
    ])
    X_pcos = np.column_stack([
        _RNG.uniform(2.0, 4.5, _N_SYNTH),
        _RNG.uniform(14.0, 30.0, _N_SYNTH),
        _RNG.uniform(45.0, 120.0, _N_SYNTH), # Aligned to updated free testosterone threshold
        _RNG.uniform(25.0, 45.0, _N_SYNTH),
    ])
    X_train = np.vstack([X_healthy, X_pcos])
    y_train = np.array([0] * _N_SYNTH + [1] * _N_SYNTH)
    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

_SHAP_SURROGATE = _build_threshold_surrogate()

_X_BACKGROUND = np.column_stack([
    np.clip(_RNG.normal(1.0, 0.25, 50), 0.5, 2.0),
    np.clip(_RNG.normal(5.5, 2.0, 50), 2.0, 13.9),
    np.clip(_RNG.normal(30.0, 8.0, 50), 10.0, 44.0),
    np.clip(_RNG.normal(21.5, 1.5, 50), 16.0, 24.9),
])

_FEATURE_NAMES = [
    "Gonadotropin Axis (LH/FSH)",
    "Insulin Pathway (Fasting Insulin)",
    "Hyperandrogenism (Testosterone)",
    "Adipose Mass (BMI)",
]

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def _get_patient_vector(raw_input: dict) -> np.ndarray:
    # FIXED: Direct matching alignment with Streamlit input keys
    lh_fsh = _safe_float(raw_input.get("lh_fsh_ratio", 1.0))
    insulin = _safe_float(raw_input.get("fasting_insulin", 5.0))
    testosterone = _safe_float(raw_input.get("free_testosterone") or raw_input.get("testosterone") or raw_input.get("testosterone_ng_dl") or 20.0)
    bmi = _safe_float(raw_input.get("bmi", 21.5))
    return np.array([[lh_fsh, insulin, testosterone, bmi]], dtype=float)

def _local_shap_attribution(x_patient: np.ndarray) -> dict:
    def pipeline_score_wrapper(x_matrix: np.ndarray) -> np.ndarray:
        return _SHAP_SURROGATE.predict_proba(x_matrix)[:, 1]

    explainer = shap.KernelExplainer(pipeline_score_wrapper, _X_BACKGROUND, link="identity")
    
    # FIXED: Explainer evaluation handling to handle both legacy arrays and modern SHAP Explanation models safely
    shap_results = explainer.shap_values(x_patient, nsamples="auto")
    
    if isinstance(shap_results, list):
        # If classifier yields list of arrays per output class, grab the positive class values
        flat_vals = shap_results[1] if len(shap_results) > 1 else shap_results[0]
    elif hasattr(shap_results, "values"):
        flat_vals = shap_results.values
    else:
        flat_vals = shap_results

    # Ensure shape is completely flattened to a 1D array
    flat_vals = np.asarray(flat_vals).flatten()

    return {name: round(float(flat_vals[i]), 4) for i, name in enumerate(_FEATURE_NAMES)}

def _extract_graph_evidence(state: PCOSState) -> dict:
    graph_knowledge = state.get("graph_knowledge", []) or []
    retrieved_chunks = state.get("retrieved_chunks", []) or []
    edges = []
    paper_hits = []

    for item in graph_knowledge:
        if isinstance(item, dict):
            title = str(item.get("title", ""))
            if "Graph Edge:" in title:
                edges.append({"title": title, "text": str(item.get("text", "")), "score": item.get("hybrid_score", 0.0)})
            elif item.get("is_paper", False):
                paper_hits.append({"title": title, "text": str(item.get("text", "")), "score": item.get("hybrid_score", 0.0)})

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            paper_hits.append({"title": str(chunk.get("title", "Retrieved Evidence")), "text": str(chunk.get("text", "")), "score": chunk.get("hybrid_score", 0.0)})

    edges = sorted(edges, key=lambda x: x.get("score", 0.0), reverse=True)[:6]
    paper_hits = sorted(paper_hits, key=lambda x: x.get("score", 0.0), reverse=True)[:4]
    return {"edges": edges, "papers": paper_hits}

def _counterfactuals(state: PCOSState) -> List[str]:
    raw_input = state.get("raw_input", {}) or {}
    node3_summary = state.get("node3_summary", {}) or {}
    missing_tests = node3_summary.get("missing_tests", []) or []
    cf = []
    
    if _safe_float(raw_input.get("lh_fsh_ratio", 0.0)) >= 2.0:
        cf.append("If LH/FSH were below the threshold, the gonadotropin-driven pattern would weaken.")
    if _safe_float(raw_input.get("fasting_insulin", 0.0)) >= 14.0:
        cf.append("If fasting insulin normalized, the metabolic amplification signal would reduce.")
    
    # FIXED: Calibrated testosterone threshold evaluation bound down to 45.0
    testosterone_val = _safe_float(raw_input.get("free_testosterone") or raw_input.get("testosterone") or 0.0)
    if testosterone_val >= 45.0:
        cf.append("If testosterone were lower, androgen-excess support would be weaker.")
        
    if _safe_float(raw_input.get("bmi", 0.0)) >= 25.0:
        cf.append("If BMI moved into the normal range, adipose-associated contribution would reduce.")
        
    # FIXED: Pull explicit missing systemic panels from Node 3's clean text rules instead of Node 2's LLM array
    if missing_tests:
        cf.append(f"Missing clinical rule-out metrics: {', '.join(missing_tests)}.")
        
    if not cf:
        cf.append("No strong counterfactual trigger detected from the provided values.")
    return cf[:5]

def _humanize_quantum(quantum_scores: dict) -> dict:
    return {
        "interaction_score": quantum_scores.get("quantum_interaction_score", 0.0),
        "entropy": quantum_scores.get("von_neumann_entropy", 0.0),
        "dominant_states": quantum_scores.get("top_states", [])[:3],
        "summary": quantum_scores.get("interpretation", "N/A"),
    }

def node5_xai_fn(state: PCOSState) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("[NODE 5] BUILDING EXPLANATION REPORT")
    print("=" * 60)

    raw_input = state.get("raw_input", {}) or {}
    classical_scores = state.get("classical_scores", {}) or {}
    quantum_scores = state.get("quantum_scores", {}) or {}
    ici_metrics = state.get("ici_metrics", {}) or {}
    clinical_hypothesis = state.get("clinical_hypothesis", {}) or {}
    node3_summary = state.get("node3_summary", {}) or {}

    x_patient = _get_patient_vector(raw_input)
    shap_contrib = _local_shap_attribution(x_patient)
    graph_evidence = _extract_graph_evidence(state)
    counterfactuals = _counterfactuals(state)
    quantum_view = _humanize_quantum(quantum_scores)

    confidence = classical_scores.get("bayesian_credibility_score", 0.0)
    bounds = classical_scores.get("confidence_interval_bounds", [0.0, 1.0])
    width = round(float(bounds[1] - bounds[0]), 4) if len(bounds) == 2 else 0.0

    phenotype = clinical_hypothesis.get("phenotype_assessment", "Unclassified phenotype")
    pcos_likely = clinical_hypothesis.get("pcos_diagnosis_likely", False)
    top_rank = node3_summary.get("hypothesis_rank", "Indeterminate")

    graph_lines = []
    for edge in graph_evidence["edges"]:
        graph_lines.append(f"- {edge['title']} (score={edge['score']:.3f})")
    if not graph_lines:
        graph_lines.append("- No graph edges available.")

    paper_lines = []
    for paper in graph_evidence["papers"]:
        paper_lines.append(f"- {paper['title']} (score={paper['score']:.3f})")
    if not paper_lines:
        paper_lines.append("- No literature hits available.")

    shap_ranked = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_lines = [f"- {k}: {v:+.4f}" for k, v in shap_ranked]

    report_markdown = f"""# CLINICAL INTEGRATED DIAGNOSTICS & EXPLAINABILITY DOSSIER
**System Pipeline Reference:** XAI-LOGIC-CORE-ENG  
**Phenotype Classification Target:** {phenotype}  
**Integrated Clinical Index (ICI Core Score):** {ici_metrics.get('integrated_clinical_index', 0.0):.4f}  
**Node 3 Rank:** {top_rank}  

---
## 1. CLINICAL CONSENSUS
- **PCOS Likelihood:** {"Likely" if pcos_likely else "Uncertain / mimic workup needed"}
- **Consensus Statement:** {clinical_hypothesis.get("clinical_hypothesis", "No consensus text generated.")}
- **Primary Pathway:** {clinical_hypothesis.get("primary_risk_factor", "N/A")}

## 2. EVIDENCE GRAPH SUMMARY
### Supporting Graph Edges
{chr(10).join(graph_lines)}

### Supporting Literature
{chr(10).join(paper_lines)}

## 3. LOCAL BIOMARKER ATTRIBUTION
- **Model Confidence:** {confidence * 100:.2f}%
- **95% Credibility Bounds:** {bounds}
- **Uncertainty Width:** {width}

{chr(10).join(shap_lines)}

## 4. COUNTERFACTUAL INTERPRETATION
{chr(10).join([f"- {line}" for line in counterfactuals])}

## 5. QUANTUM INTERACTION NOTE
- **Interaction Score:** {quantum_view["interaction_score"]:.4f}
- **Entropy:** {quantum_view["entropy"]:.4f}
- **Interpretation:** {quantum_view["summary"]}

### Dominant Quantum States
"""
    for st in quantum_view["dominant_states"]:
        decoded = st.get("decoded", {})
        active = [k for k, v in decoded.items() if v == "ACTIVATED"]
        report_markdown += f"- State |{st.get('state', '0000')}⟩ ({st.get('probability', 0.0) * 100:.2f}%) -> {' + '.join(active) if active else 'baseline'}\n"

    report_markdown += """
## 6. FINAL EXPLANATION
The system combines graph evidence, local biomarker attribution, and counterfactual reasoning to support the current hypothesis while preserving uncertainty about unmeasured mimic conditions.

---
*Report compiled successfully by Node 5 XAI Engine.*
"""

    state["xai_metrics"] = {
        "shap_importance_vectors": shap_contrib,
        "graph_evidence": graph_evidence,
        "counterfactuals": counterfactuals,
        "quantum_summary": quantum_view,
        "confidence_width": width,
    }
    state["xai_report"] = report_markdown
    print("[NODE 5] REPORT BUILT.")
    print("=" * 60 + "\n")
    return state