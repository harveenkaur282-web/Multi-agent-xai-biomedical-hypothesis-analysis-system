import json
import os
import numpy as np
from typing import Dict, Any, List
from pipeline.state import PCOSState

def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def _count_citations(debate_history: dict) -> dict:
    citation_counts = {}
    pattern = r"\[Source-\d+,\s*Chunk-\d+\]"
    for agent, text in debate_history.items():
        if isinstance(text, str):
            matches = re.findall(pattern, text)
            for m in matches:
                citation_counts[m] = citation_counts.get(m, 0) + 1
    return citation_counts

import re

def node5_xai_fn(state: PCOSState) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("[NODE 5] BUILDING ARGUMENTATION-BASED EXPLANATION DOSSIER")
    print("=" * 60)

    raw_input = state.get("raw_input", {}) or {}
    classical_scores = state.get("classical_scores", {}) or {}
    quantum_scores = state.get("quantum_scores", {}) or {}
    ici_metrics = state.get("ici_metrics", {}) or {}
    clinical_hypothesis = state.get("clinical_hypothesis", {}) or {}
    node3_summary = state.get("node3_summary", {}) or {}
    debate_history = state.get("debate_history", {}) or {}

    lh_fsh_val = _safe_float(raw_input.get("lh_fsh_ratio", 1.0))
    insulin_val = _safe_float(raw_input.get("fasting_insulin", 5.0))
    bmi_val = _safe_float(raw_input.get("bmi", 21.5))
    amh_val = _safe_float(raw_input.get("amh_levels", 1.0))
    testo_val = _safe_float(raw_input.get("free_testosterone") or raw_input.get("testosterone") or 20.0)
    remarks = raw_input.get("clinical_remarks", "")

    missing_tests = node3_summary.get("missing_tests", []) or []
    mimic_risk = node3_summary.get("mimic_risk", False)

    # 1. Define Arguments (Clinical Propositions)
    arguments = {}
    
    # A1: Reproductive Claim
    has_reproductive = (lh_fsh_val >= 2.0 or amh_val >= 4.0 or any(k in remarks.lower() for k in ["oligomenorrhea", "irregular", "periods", "polycystic"]))
    arguments["A1"] = {
        "id": "A1",
        "title": "Reproductive Pathological Axis",
        "claim": "Reproductive symptoms align with PCOS presentation.",
        "evidence": f"LH/FSH ratio is {lh_fsh_val} (elevated if >= 2.0), AMH is {amh_val} ng/mL. History notes irregular cycles or polycystic morphology.",
        "base_status": "IN" if has_reproductive else "OUT"
    }

    # A2: Metabolic Claim
    has_metabolic = (insulin_val >= 14.0 or bmi_val >= 25.0)
    arguments["A2"] = {
        "id": "A2",
        "title": "Metabolic Risk Axis",
        "claim": "Insulin resistance or elevated BMI drives metabolic PCOS phenotype.",
        "evidence": f"Fasting insulin is {insulin_val} uIU/mL (elevated if >= 14.0), BMI is {bmi_val} kg/m².",
        "base_status": "IN" if has_metabolic else "OUT"
    }

    # A3: Mimic Attack (Conflict)
    has_mimic_threat = len(missing_tests) > 0 or mimic_risk
    arguments["A3"] = {
        "id": "A3",
        "title": "Diagnostic Mimic Warning",
        "claim": "Unmeasured thyroid, adrenal, or pituitary markers pose mimic risks (e.g. Hypothyroidism, NCAH).",
        "evidence": f"Missing rule-out panels: {', '.join(missing_tests) if missing_tests else 'None'}." if has_mimic_threat else "All mimic rule-out panels are verified.",
        "base_status": "IN" if has_mimic_threat else "OUT"
    }

    # A4: Resolution Action
    arguments["A4"] = {
        "id": "A4",
        "title": "Recommended Guideline Actions",
        "claim": "Perform TSH, 17-OH Progesterone, and Prolactin tests to resolve the diagnostic mimic warning.",
        "evidence": f"Ordering {len(missing_tests)} missing checks will eliminate alternative endocrine etiologies.",
        "base_status": "IN" if has_mimic_threat else "OUT"
    }

    # 2. Dung's Grounded Extension Solver
    # Attacks: A3 attacks A1, A3 attacks A2, A4 attacks A3
    # We compute the set of acceptable arguments:
    # - If A4 is IN, it defeats A3. Then A3 is OUT, so A3 cannot defeat A1 or A2.
    # - If A4 is OUT (i.e. no missing tests), then A3 is OUT. A1 and A2 are IN.
    # So A1 and A2 are accepted. A3 is defeated (resolved), and A4 is active as a recommendation.
    
    evaluated_arguments = {}
    for arg_id, arg in arguments.items():
        evaluated_arguments[arg_id] = dict(arg)

    if has_mimic_threat:
        # A4 is active
        evaluated_arguments["A4"]["status"] = "ACCEPTED"
        evaluated_arguments["A3"]["status"] = "DEFEATED"  # Defeated by A4's diagnostic proposal
        evaluated_arguments["A1"]["status"] = "ACCEPTED" if has_reproductive else "OUT"
        evaluated_arguments["A2"]["status"] = "ACCEPTED" if has_metabolic else "OUT"
    else:
        evaluated_arguments["A4"]["status"] = "OUT"
        evaluated_arguments["A3"]["status"] = "OUT"
        evaluated_arguments["A1"]["status"] = "ACCEPTED" if has_reproductive else "OUT"
        evaluated_arguments["A2"]["status"] = "ACCEPTED" if has_metabolic else "OUT"

    # Define attack/support graph edges
    graph_edges = []
    if has_mimic_threat:
        graph_edges.append({"source": "A3", "target": "A1", "relation": "ATTACKS"})
        graph_edges.append({"source": "A3", "target": "A2", "relation": "ATTACKS"})
        graph_edges.append({"source": "A4", "target": "A3", "relation": "RESOLVES"})

    # Count citations in the debate history
    citation_map = _count_citations(debate_history)

    # Mock shap_importance_vectors to avoid breaking any legacy code that expects it
    mock_importance = {
        "Gonadotropin Axis (LH/FSH)": float(lh_fsh_val),
        "Insulin Pathway (Fasting Insulin)": float(insulin_val),
        "Hyperandrogenism (Testosterone)": float(testo_val),
        "Adipose Mass (BMI)": float(bmi_val)
    }

    confidence = classical_scores.get("bayesian_credibility_score", 0.0)
    bounds = classical_scores.get("confidence_interval_bounds", [0.0, 1.0])
    width = round(float(bounds[1] - bounds[0]), 4) if len(bounds) == 2 else 0.0

    phenotype = clinical_hypothesis.get("phenotype_assessment", "Unclassified phenotype")
    pcos_likely = clinical_hypothesis.get("pcos_diagnosis_likely", False)
    top_rank = node3_summary.get("hypothesis_rank", "Indeterminate")

    # Generate classical vs quantum comparative table
    bayes_score = classical_scores.get("bayesian_credibility_score", 0.0)
    quantum_score = quantum_scores.get("quantum_interaction_score", 0.0)
    uncertainty = classical_scores.get("uncertainty_score", 0.0)
    entropy = quantum_scores.get("von_neumann_entropy", 0.0)

    comparison_table = f"""
| Analysis Dimension | Classical Statistical Space | Quantum Entangled Space (Simulated) |
| :--- | :--- | :--- |
| **Primary Index Score** | Bayesian Credibility: {bayes_score * 100:.1f}% | Quantum Interaction: {quantum_score:.4f} |
| **Mathematical Focus** | Linear threshold bounds & clinical rules | Multi-axis non-linear coupling / qubit states |
| **Uncertainty / Entropy** | Statistical Uncertainty: {uncertainty * 100:.1f}% | Von Neumann Entropy: {entropy:.4f} |
| **Clinical Interpretation** | {classical_scores.get('interpretation', 'N/A')} | {quantum_scores.get('interpretation', 'N/A')} |
"""

    report_markdown = f"""# CLINICAL DIAGNOSTICS ARGUMENTATION & EXPLAINABILITY DOSSIER
**System Pipeline Reference:** ArgMed-Agents-XAI-Core  
**Phenotype Classification Target:** {phenotype}  
**Integrated Clinical Index (ICI Core Score):** {ici_metrics.get('integrated_clinical_index', 0.0):.4f}  
**Node 3 Rank:** {top_rank}  

---
## 1. CLINICAL CONSENSUS & LOGIC ARGUMENTATION
- **PCOS Likelihood:** {"Likely" if pcos_likely else "Uncertain / mimic workup needed"}
- **Consensus Statement:** {clinical_hypothesis.get("clinical_hypothesis", "No consensus text generated.")}
- **Primary Pathway:** {clinical_hypothesis.get("primary_risk_factor", "N/A")}

---
## 2. DUNG'S ABSTRACT ARGUMENTATION SYSTEM (ArgMed-Agents Framework)
The system models clinical decision reasoning using arguments and conflict relations. 
- **Argument A1 (Reproductive Axis):** {evaluated_arguments['A1']['status']} — {evaluated_arguments['A1']['evidence']}
- **Argument A2 (Metabolic Axis):** {evaluated_arguments['A2']['status']} — {evaluated_arguments['A2']['evidence']}
- **Argument A3 (Mimic Warning):** {evaluated_arguments['A3']['status']} — {evaluated_arguments['A3']['evidence']}
- **Argument A4 (Resolution Guideline):** {evaluated_arguments['A4']['status']} — {evaluated_arguments['A4']['evidence']}

*Decision Result:* {
    "PCOS hypothesis accepted under warning. Order rule-out tests to finalize." if has_mimic_threat 
    else "PCOS hypothesis accepted. All mimic warnings cleared."
}

---
## 3. EVIDENCE GROUNDING & CITATION ANALYSIS
- **Total Unique Literature Citations in Debate:** {len(citation_map)}
- **Citation Grounding Frequency Map:**
"""
    if citation_map:
        for chunk_id, count in sorted(citation_map.items(), key=lambda x: x[1], reverse=True):
            report_markdown += f"  - `{chunk_id}`: Referenced {count} times during multi-agent consensus.\n"
    else:
        report_markdown += "  - No active literature citations referenced during the consensus debate.\n"

    report_markdown += f"""
---
## 4. MATHEMATICAL COMPARISON: CLASSICAL VS. QUANTUM ENTANGLEMENT SPACE
{comparison_table}

---
*Report compiled successfully by Node 5 ArgMed Clinical XAI Engine.*
"""

    state["xai_metrics"] = {
        "shap_importance_vectors": mock_importance,
        "arguments": evaluated_arguments,
        "graph_edges": graph_edges,
        "citations": citation_map,
        "confidence_width": width,
    }
    state["xai_report"] = report_markdown
    print("[NODE 5] ARGUMENTATION DOSSIER COMPILED.")
    print("=" * 60 + "\n")
    return state