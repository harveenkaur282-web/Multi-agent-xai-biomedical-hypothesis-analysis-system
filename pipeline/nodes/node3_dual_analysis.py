import json
import os
import scipy.stats as stats
import numpy as np
from pipeline.state import PCOSState
from utils.quantum_utils import run_quantum_analysis

def interpret_node3_results(classical_scores: dict, quantum_scores: dict, ici_score: float) -> dict:
    credibility = classical_scores.get("bayesian_credibility_score", 0.0)
    interaction = quantum_scores.get("quantum_interaction_score", 0.0)
    
    if credibility >= 0.80:
        classical_text = "Highly Stable. The empirical biomarker presentation matches verified clinical distributions."
    elif credibility >= 0.60:
        classical_text = "Withstands Verification. The presentation displays standard variations requiring regular monitoring."
    else:
        classical_text = "Low Statistical Alignment. The clinical markers show an anomalous presentation compared to baseline data trends."
        
    if interaction >= 0.70:
        quantum_text = "Strong State Correlation. The simulated entanglement layer shows high co-dependence between metabolic and endocrine feature vectors."
    elif interaction >= 0.45:
        quantum_text = "Moderate Interaction. The circuit shows partial, non-linear co-dependence between metabolic and endocrine feature axes."
    else:
        quantum_text = "Weak State Correlation. The feature interaction layers display isolated vector properties."
        
    if ici_score >= 0.75:
        ici_text = "CRITICAL HYBRID CONVERGENCE: Both classical probabilistic evidence and quantum state mapping strongly confirm a metabolic-endocrine phenotype pattern."
    elif ici_score >= 0.50:
        ici_text = "MODERATE HYBRID ALIGNMENT: System indicates active path developments. Early clinical interventions recommended."
    else:
        ici_text = "LOW HYBRID SIGNATURE: Biomarker vector variations fall primarily within standard baseline limits."
        
    return {
        "classical_interpretation": classical_text,
        "quantum_interpretation": quantum_text,
        "ici_interpretation": ici_text
    }

def node3_dual_analysis_fn(state: PCOSState) -> dict:
    raw_case = state.get("raw_input", {})
    hypothesis = state.get("consensus_hypothesis", {})
    
    # Load externalized medical parameters configuration dynamically
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        # High-integrity structural fallback if local configuration path leaks
        config_data = {
            "lab_thresholds": {
                "fasting_insulin_uiu_ml": {"elevated_min": 12.0},
                "lh_fsh_ratio": {"elevated_min": 2.0},
                "bmi": {"elevated_min": 25.0},
                "amh_ng_ml": {"elevated_min": 4.0},
                "testosterone_ng_dl": {"elevated_min": 45.0}
            },
            "normalization_ranges": {
                "fasting_insulin_uiu_ml": {"min": 2.0, "max": 30.0},
                "lh_fsh_ratio": {"min": 0.5, "max": 4.5},
                "age": {"min": 18.0, "max": 45.0}
            }
        }
    
    thresholds = config_data["lab_thresholds"]
    norm_ranges = config_data["normalization_ranges"]
    
    # ──── CONFIG-DRIVEN FEATURE VECTOR NORMALIZATION FOR QML LAYER ────
    # Mapping qualitative labels from CrewAI task 3 down to numeric scalars
    confidence_map = {"High": 0.9, "Medium": 0.6, "Low": 0.3}
    agent_confidence = confidence_map.get(hypothesis.get("agent_confidence_level", "Medium"), 0.6)
    
    raw_insulin = float(raw_case.get("fasting_insulin", 0.0))
    raw_ratio = float(raw_case.get("lh_fsh_ratio", 0.0))
    raw_age = float(raw_case.get("age", 27.0))
    
    def scale_val(val, range_key):
        r_min = norm_ranges[range_key]["min"]
        r_max = norm_ranges[range_key]["max"]
        return (val - r_min) / (r_max - r_min) if val > r_min else 0.4

    norm_insulin = scale_val(raw_insulin, "fasting_insulin_uiu_ml")
    norm_ratio = scale_val(raw_ratio, "lh_fsh_ratio")
    norm_age = scale_val(raw_age, "age")
    
    safely_scaled_feature_vector = [agent_confidence, norm_insulin, norm_ratio, norm_age]

    # ──── CONFIG-DRIVEN BAYESIAN PROBABILISTIC CONJUGATE CORNER ────
    abnormal_markers = 0
    total_markers = 0

    mapping = {
        "fasting_insulin": "fasting_insulin_uiu_ml",
        "lh_fsh_ratio": "lh_fsh_ratio",
        "bmi": "bmi",
        "amh_levels": "amh_ng_ml",
        "free_testosterone": "testosterone_ng_dl"
    }

    for case_key, json_key in mapping.items():
        patient_val = raw_case.get(case_key)
        if patient_val is not None:
            total_markers += 1
            threshold_cutoff = thresholds[json_key]["elevated_min"]
            if float(patient_val) >= threshold_cutoff:
                abnormal_markers += 1

    # Apply conjugate beta update rule
    alpha_prior, beta_prior = 7, 3
    posterior_alpha = alpha_prior + abnormal_markers
    posterior_beta = beta_prior + (total_markers - abnormal_markers)
    classical_credibility = float(posterior_alpha / (posterior_alpha + posterior_beta))
    
    classical_summary = {
        "bayesian_credibility_score": round(classical_credibility, 4),
        "confidence_interval_bounds": [round(x, 3) for x in stats.beta.interval(0.95, posterior_alpha, posterior_beta)]
    }

    # Execute simulated quantum evaluation matrix
    quantum_summary = run_quantum_analysis(safely_scaled_feature_vector)
    quantum_score = quantum_summary.get("quantum_interaction_score", 0.0)
    
    # ──── ENSEMBLE SYSTEM FUSION MATRIX (INTEGRATED CLINICAL INDEX) ────
    w1, w2 = 0.5, 0.5
    ici_score = float((w1 * classical_credibility) + (w2 * quantum_score))

    interpretations = interpret_node3_results(classical_summary, quantum_summary, ici_score)
    
    return {
        "classical_scores": {
            **classical_summary,
            "interpretation": interpretations["classical_interpretation"]
        },
        "quantum_scores": {
            **quantum_summary,
            "interpretation": interpretations["quantum_interpretation"]
        },
        "ici_metrics": {
            "integrated_clinical_index": round(ici_score, 4),
            "interpretation": interpretations["ici_interpretation"]
        }
    }