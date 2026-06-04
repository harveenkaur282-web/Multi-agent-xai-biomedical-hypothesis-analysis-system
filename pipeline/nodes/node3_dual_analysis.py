import json
import os
import scipy.stats as stats
import numpy as np
from pipeline.state import PCOSState
from utils.quantum_utils import run_quantum_analysis

def _get_labs(raw_case: dict) -> dict:
    """
    Handles both flat keys (legacy) and nested labs dict (mock_patients.json format).
    Always returns a flat dict with standardized key names.
    """
    labs = raw_case.get("labs", {})
    return {
        "fasting_insulin": float(
            labs.get("fasting_insulin_uiu_ml") or
            raw_case.get("fasting_insulin") or 0.0
        ),
        "lh_fsh_ratio": float(
            labs.get("lh_fsh_ratio") or
            raw_case.get("lh_fsh_ratio") or 0.0
        ),
        "testosterone": float(
            labs.get("testosterone_ng_dl") or
            raw_case.get("free_testosterone") or 0.0
        ),
        "bmi": float(
            raw_case.get("bmi") or 0.0
        ),
        "amh": float(
            labs.get("amh_ng_ml") or
            raw_case.get("amh_levels") or 0.0
        ),
        "homa_ir": float(
            labs.get("homa_ir") or
            raw_case.get("homa_ir") or 0.0
        ),
        "age": float(raw_case.get("age") or 27.0)
    }

def _normalize(val: float, range_key: str, norm_ranges: dict) -> float:
    r = norm_ranges.get(range_key, {})
    r_min = r.get("min", 0.0)
    r_max = r.get("max", 1.0)
    if r_max == r_min:
        return 0.5
    return float(np.clip((val - r_min) / (r_max - r_min), 0.0, 1.0))

def interpret_node3_results(
    classical_scores: dict,
    quantum_scores: dict,
    ici_score: float
) -> dict:
    credibility = classical_scores.get("bayesian_credibility_score", 0.0)
    interaction = quantum_scores.get("quantum_interaction_score", 0.0)

    if credibility >= 0.80:
        classical_text = "Highly Stable. Empirical biomarker presentation matches verified clinical distributions."
    elif credibility >= 0.60:
        classical_text = "Withstands Verification. Standard variations present, requiring regular monitoring."
    else:
        classical_text = "Low Statistical Alignment. Anomalous presentation compared to baseline data trends."

    if interaction >= 0.70:
        quantum_text = "Strong State Correlation. High co-dependence between metabolic and endocrine feature vectors."
    elif interaction >= 0.45:
        quantum_text = "Moderate Interaction. Partial non-linear co-dependence between metabolic and endocrine axes."
    else:
        quantum_text = "Weak State Correlation. Feature interaction layers display isolated vector properties."

    if ici_score >= 0.75:
        ici_text = "CRITICAL HYBRID CONVERGENCE: Both engines strongly confirm a metabolic-endocrine phenotype pattern."
    elif ici_score >= 0.50:
        ici_text = "MODERATE HYBRID ALIGNMENT: Active pathway developments detected. Early intervention recommended."
    else:
        ici_text = "LOW HYBRID SIGNATURE: Biomarker variations fall within standard baseline limits."

    return {
        "classical_interpretation": classical_text,
        "quantum_interpretation":   quantum_text,
        "ici_interpretation":        ici_text
    }

def node3_dual_analysis_fn(state: PCOSState) -> dict:
    raw_case   = state.get("raw_input", {})
    hypothesis = state.get("clinical_hypothesis") or {}

    # Load config
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except FileNotFoundError:
        config_data = {
            "lab_thresholds": {
                "fasting_insulin_uiu_ml": {"elevated_min": 14.0},
                "lh_fsh_ratio":           {"elevated_min": 2.0},
                "bmi":                    {"elevated_min": 25.0},
                "amh_ng_ml":              {"elevated_min": 6.0},
                "testosterone_ng_dl":     {"elevated_min": 70.0},
                "homa_ir":                {"elevated_min": 2.5}
            },
            "normalization_ranges": {
                "fasting_insulin_uiu_ml": {"min": 2.0,  "max": 30.0},
                "lh_fsh_ratio":           {"min": 0.5,  "max": 4.5},
                "testosterone_ng_dl":     {"min": 10.0, "max": 120.0},
                "bmi":                    {"min": 16.0, "max": 45.0},
                "age":                    {"min": 18.0, "max": 45.0}
            }
        }

    thresholds  = config_data["lab_thresholds"]
    norm_ranges = config_data["normalization_ranges"]

    # ── Unified lab extraction (handles nested + flat formats) ──────────────
    labs = _get_labs(raw_case)

    # ── Feature vector: pure biomarkers only, aligned to VQC qubit mapping ──
    # Qubit 0 → Gonadotropin Axis (LH/FSH)
    # Qubit 1 → Insulin Pathway   (Fasting Insulin)
    # Qubit 2 → Hyperandrogenism  (Testosterone)
    # Qubit 3 → Adipose Mass      (BMI)
    norm_lh_fsh      = _normalize(labs["lh_fsh_ratio"],  "lh_fsh_ratio",           norm_ranges)
    norm_insulin     = _normalize(labs["fasting_insulin"],"fasting_insulin_uiu_ml", norm_ranges)
    norm_testosterone= _normalize(labs["testosterone"],   "testosterone_ng_dl",     norm_ranges)
    norm_bmi         = _normalize(labs["bmi"],            "bmi",                    norm_ranges)

    safely_scaled_feature_vector = [
        norm_lh_fsh,        # qubit 0
        norm_insulin,       # qubit 1
        norm_testosterone,  # qubit 2
        norm_bmi            # qubit 3
    ]

    print(f"[Node3] Feature vector: {[round(x,3) for x in safely_scaled_feature_vector]}")

    # ── Bayesian conjugate update using actual abnormal lab count ───────────
    abnormal_markers = 0
    total_markers    = 0

    threshold_checks = {
        "fasting_insulin": "fasting_insulin_uiu_ml",
        "lh_fsh_ratio":    "lh_fsh_ratio",
        "bmi":             "bmi",
        "amh":             "amh_ng_ml",
        "testosterone":    "testosterone_ng_dl",
        "homa_ir":         "homa_ir"
    }

    for lab_key, config_key in threshold_checks.items():
        val = labs.get(lab_key)
        threshold = thresholds.get(config_key, {}).get("elevated_min")
        if val is not None and threshold is not None and val > 0:
            total_markers += 1
            if val >= threshold:
                abnormal_markers += 1

    print(f"[Node3] Bayesian: {abnormal_markers}/{total_markers} markers abnormal")

    alpha_prior, beta_prior = 7, 3
    posterior_alpha = alpha_prior + abnormal_markers
    posterior_beta  = beta_prior  + (total_markers - abnormal_markers)
    classical_credibility = float(posterior_alpha / (posterior_alpha + posterior_beta))

    classical_summary = {
        "bayesian_credibility_score":   round(classical_credibility, 4),
        "confidence_interval_bounds":   [
            round(x, 3) for x in
            stats.beta.interval(0.95, posterior_alpha, posterior_beta)
        ],
        "abnormal_markers":  abnormal_markers,
        "total_markers":     total_markers,
        # Store normalized values so Node 5 SHAP can read them directly
        "feature_vector": {
            "norm_lh_fsh":       round(norm_lh_fsh,       4),
            "norm_insulin":      round(norm_insulin,       4),
            "norm_testosterone": round(norm_testosterone,  4),
            "norm_bmi":          round(norm_bmi,           4)
        }
    }

    # ── Quantum evaluation ──────────────────────────────────────────────────
    quantum_summary = run_quantum_analysis(safely_scaled_feature_vector)
    quantum_score   = quantum_summary.get("quantum_interaction_score", 0.0)

    # ── ICI fusion ──────────────────────────────────────────────────────────
    ici_score = float((0.5 * classical_credibility) + (0.5 * quantum_score))
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
            "interpretation":            interpretations["ici_interpretation"]
        }
    }