import numpy as np
import shap
from typing import Dict, Any
from pipeline.state import PCOSState

def node5_xai_fn(state: PCOSState) -> Dict[str, Any]:
    """
    Node 5: eXplainable AI (XAI) Engine.
    Computes game-theoretic attribution weights via official KernelSHAP 
    and decodes quantum Hilbert space density states into clinical language.
    """
    print("\n" + "="*60)
    print("[NODE 5] COMPUTING GAME-THEORETIC SHAP & QUANTUM DECODING...")
    print("="*60)
    
    # 1. Safely extract upstream pipeline features
    raw_input = state.get("raw_input", {})
    classical_scores = state.get("classical_scores", {})
    quantum_scores = state.get("quantum_scores", {})
    ici_metrics = state.get("ici_metrics", {})
    consensus_hypothesis = state.get("clinical_hypothesis", {})
    
    # 2. Extract specific numeric parameters for the SHAP baseline comparison
    lh_fsh = float(raw_input.get("lh_fsh_ratio", 1.0))
    insulin = float(raw_input.get("fasting_insulin", 5.0))
    testosterone = float(raw_input.get("free_testosterone") or raw_input.get("testosterone") or 20.0)
    bmi = float(raw_input.get("bmi", 21.5))
    
    # Formulate patient evaluation vector
    X_patient = np.array([[lh_fsh, insulin, testosterone, bmi]])

    # 3. Model-Agnostic Core Wrapper Function
    def pipeline_score_wrapper(x_matrix: np.ndarray) -> np.ndarray:
        """
        Mimics pipeline boundary thresholds. SHAP perturbs data 
        through this to calculate true Shapley additive importance values.
        """
        scores = []
        for row in x_matrix:
            lh, ins, test, body_mass = row[0], row[1], row[2], row[3]
            # Custom linear/non-linear boundary scoring emulation
            base_score = (lh * 0.15) + (ins * 0.05) + (test * 0.02)
            if body_mass >= 25.0:
                base_score += (body_mass - 25.0) * 0.08
            else:
                base_score -= 0.5
            scores.append(base_score)
        return np.array(scores)

    # Establish an idealized clinical background reference matrix (Healthy Reference Profile)
    X_background = np.array([[1.0, 5.0, 20.0, 21.5]])
    
    # Execute KernelSHAP Explainer
    explainer = shap.KernelExplainer(pipeline_score_wrapper, X_background, link="identity")
    shap_values = explainer.shap_values(X_patient, nsamples="auto")
    
    # Handle optional outer list wrapping common in KernelSHAP outputs
    flat_shap = shap_values[0] if isinstance(shap_values, list) else shap_values
    
    shap_contributions = {
        "Gonadotropin Axis (LH/FSH)": round(float(flat_shap[0][0]), 4),
        "Insulin Pathway (Fasting Insulin)": round(float(flat_shap[0][1]), 4),
        "Hyperandrogenism (Testosterone)": round(float(flat_shap[0][2]), 4),
        "Adipose Mass (BMI)": round(float(flat_shap[0][3]), 4)
    }

    # 4. Process Advanced Quantum Signatures from Node 3
    entropy = quantum_scores.get("von_neumann_entropy", 0.0)
    top_states = quantum_scores.get("top_states", [])
    qubit_activation = quantum_scores.get("qubit_activation", {})
    
    quantum_narrative_lines = []
    for execution in top_states:
        state_str = execution.get("state", "0000")
        probability = round(execution.get("probability", 0.0) * 100, 2)
        
        # Identify which channels are activated in this basis state
        decoded_map = execution.get("decoded", {})
        active_biomarkers = [axis for axis, status in decoded_map.items() if status == "ACTIVATED"]
        
        pathway_summary = " + ".join(active_biomarkers) if active_biomarkers else "Homeostatic Balance State"
        quantum_narrative_lines.append(f"State |{state_str}⟩ ({probability}% Density) ➔ Focus Area: [{pathway_summary}]")

    # 5. Calculate Statistical Dispersion Span
    bounds = classical_scores.get("confidence_interval_bounds", [0.5, 0.5])
    variance_span = round(bounds[1] - bounds[0], 4)

    # 6. Build the Complete Clinical Markdown Dossier
    report_markdown = f"""# 🧬 CLINICAL INTEGRATED DIAGNOSTICS & EXPLAINABILITY DOSSIER
**System Pipeline Reference:** XAI-LOGIC-CORE-ENG  
**Phenotype Classification Target:** {consensus_hypothesis.get('phenotype_assessment', 'Unclassified Phenotype Variant')}  
**Integrated Clinical Index (ICI Core Score):** {ici_metrics.get('integrated_clinical_index', 0.0)}  

---

## 📊 1. CLASSICAL BAYESIAN CREDIBILITY LAYER
* **Posterior Statistical Belief Score:** {classical_scores.get('bayesian_credibility_score', 0.0) * 100:.2f}%
* **95% Posterior Credibility Bounds (Beta Distribution):** {bounds}
* **Statistical Dispersion Delta (Confidence Spread Width):** {variance_span}
* **Clinical Integrity Assessment:** {classical_scores.get('interpretation', 'N/A')}

## ⚛️ 2. QUANTUM SUBSYSTEM DECONSTRUCTION (VQC ANCHOR)
* **Measurement-Space von Neumann Entropy:** {entropy} bits  
* **Cross-Axis Interaction Evaluation:** {quantum_scores.get('interpretation', 'N/A')}  
* **Dominant Sub-Circuit Wavefunction Signatures Decoded:**
"""
    for line in quantum_narrative_lines:
        report_markdown += f"  * {line}\n"
        
    report_markdown += """
## 🎯 3. GAME-THEORETIC MARGINAL ATTRIBUTION (OFFICIAL KERNELSHAP)
Computed additive point deviations comparing current parameters against an idealized, completely healthy physiological baseline reference profile ($X_{background}$):
"""
    for feature, value in shap_contributions.items():
        report_markdown += f"* **{feature}:** {value:+.4f} points\n"
        
    report_markdown += f"""
## 🧠 4. GENERATIVE SYNDROMIC CONSENSUS HYPOTHESIS
* **Primary Threat / Pathway Vector:** {consensus_hypothesis.get('primary_risk_factor', 'N/A')}
* **Biomedical Hypothesis Statement Summary:** {consensus_hypothesis.get('clinical_hypothesis', 'No consensus text generated.')}

---
*Report formatted and compiled successfully by Node 5 XAI Pipeline Agent.*
"""

    # 7. Update PCOSState keys natively
    state["xai_metrics"] = {
        "shap_importance_vectors": shap_contributions,
        "qubit_activation_probabilities": qubit_activation,
        "von_neumann_entropy": entropy,
        "variance_span": variance_span
    }
    state["xai_report"] = report_markdown

    print("[NODE 5] XAI METRICS ENGINES RUN COMPLETED. PAYLOAD ATTACHED TO STATE.")
    print("="*60 + "\n")
    
    return state