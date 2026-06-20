import json
import os
import numpy as np
from pipeline.state import PCOSState
from utils.quantum_utils import run_quantum_analysis

def _load_thresholds() -> dict:
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {
            "lab_thresholds": {
                "fasting_insulin_uiu_ml": {"elevated_min": 14.0},
                "lh_fsh_ratio": {"elevated_min": 2.0},
                "bmi": {"elevated_min": 25.0},
                "amh_ng_ml": {"elevated_min": 4.0},
                "testosterone_ng_dl": {"elevated_min": 45.0},
                "homa_ir": {"elevated_min": 2.5},
            },
            "normalization_ranges": {
                "fasting_insulin_uiu_ml": {"min": 2.0, "max": 30.0},
                "lh_fsh_ratio": {"min": 0.5, "max": 4.5},
                "testosterone_ng_dl": {"min": 5.0, "max": 100.0},
                "bmi": {"min": 16.0, "max": 45.0},
                "age": {"min": 18.0, "max": 45.0},
            },
        }

def _get_labs(raw_case: dict) -> dict:
    # Safely handle both flat input frameworks and nested "labs" dictionary payloads
    labs_obj = raw_case.get("labs")
    labs = labs_obj if isinstance(labs_obj, dict) else {}
    
    return {
        "fasting_insulin": float(labs.get("fasting_insulin_uiu_ml") or raw_case.get("fasting_insulin") or raw_case.get("fasting_insulin_uiu_ml") or 0.0),
        "lh_fsh_ratio": float(labs.get("lh_fsh_ratio") or raw_case.get("lh_fsh_ratio") or 0.0),
        "testosterone": float(labs.get("testosterone_ng_dl") or raw_case.get("free_testosterone") or raw_case.get("testosterone") or raw_case.get("testosterone_ng_dl") or 0.0),
        "bmi": float(raw_case.get("bmi") or 0.0),
        "amh": float(labs.get("amh_ng_ml") or raw_case.get("amh_levels") or raw_case.get("amh_ng_ml") or 0.0),
        "homa_ir": float(labs.get("homa_ir") or raw_case.get("homa_ir") or 0.0),
        "age": float(raw_case.get("age") or 27.0),
    }

def _normalize(val: float, rmin: float, rmax: float) -> float:
    if rmax == rmin:
        return 0.5
    return float(np.clip((val - rmin) / (rmax - rmin), 0.0, 1.0))

def _extract_clinical_flags(raw_case: dict) -> dict:
    text = " ".join([
        str(raw_case.get("clinical_remarks", "")), 
        str(raw_case.get("remarks", ""))
    ]).lower()
    
    has_oligo = any(k in text for k in ["oligomenorrhea", "oligomenorrhoea", "periods", "menstrual"])
    has_acne = any(k in text for k in ["acne", "inflammatory jawline", "comedones"])
    has_pom = any(k in text for k in ["polycystic", "pearls", "antral", "follicle", "ultrasound"])
    has_hyperandrogen = any(k in text for k in ["hirsutism", "androgen", "acne", "testosterone"])
    
    return {
        "has_oligomenorrhea": has_oligo,
        "has_acne": has_acne,
        "has_pom": has_pom,
        "has_hyperandrogenism": has_hyperandrogen,
        "clinical_text": text,
    }

def interpret_node3_results(classical_scores: dict, quantum_scores: dict, ici_score: float, mimic_risk: bool) -> dict:
    support = classical_scores.get("classical_support_score", 0.0)
    uncertainty = classical_scores.get("uncertainty_score", 0.0)
    interaction = quantum_scores.get("quantum_interaction_score", 0.0)

    if support >= 0.75:
        classical_text = "Strong evidence support. Clinical features align robustly with Rotterdam diagnostic parameters."
    elif support >= 0.50:
        classical_text = "Moderate evidence support. The presentation is medically plausible but lacks full marker validation."
    else:
        classical_text = "Weak evidence support. Presentation missing secondary features."

    if uncertainty >= 0.6:
        uncertainty_text = "High uncertainty. Major rule-out evaluation metrics remain missing."
    elif uncertainty >= 0.35:
        uncertainty_text = "Moderate uncertainty. Minimal baseline mimic panels should be confirmed."
    else:
        uncertainty_text = "Low uncertainty. Patient diagnostic panel contains high structural density."

    if interaction >= 0.65:
        quantum_text = "Strong multi-axis coupling. Biomarker cross-sections present non-linear entanglement signals."
    elif interaction >= 0.45:
        quantum_text = "Moderate multi-axis coupling. Inter-axis alignment present."
    else:
        quantum_text = "Independent pathway signals. Traditional sequential correlation patterns apply."

    if ici_score >= 0.75:
        ici_text = "High composite index confidence."
    elif ici_score >= 0.5:
        ici_text = "Moderate composite index confidence."
    else:
        ici_text = "Low cumulative framework support."

    if mimic_risk:
        ici_text += " Mimic differential rules apply."

    return {
        "classical_interpretation": classical_text,
        "quantum_interpretation": quantum_text,
        "ici_interpretation": ici_text,
        "uncertainty_interpretation": uncertainty_text,
    }

def node3_dual_analysis_fn(state: PCOSState) -> dict:
    raw_case = state.get("raw_input", {}) or {}
    thresholds_cfg = _load_thresholds()
    thresholds = thresholds_cfg["lab_thresholds"]
    norm_ranges = thresholds_cfg["normalization_ranges"]

    labs = _get_labs(raw_case)
    clinical_flags = _extract_clinical_flags(raw_case)

    norm_lh_fsh = _normalize(labs["lh_fsh_ratio"], norm_ranges["lh_fsh_ratio"]["min"], norm_ranges["lh_fsh_ratio"]["max"])
    norm_insulin = _normalize(labs["fasting_insulin"], norm_ranges["fasting_insulin_uiu_ml"]["min"], norm_ranges["fasting_insulin_uiu_ml"]["max"])
    norm_testosterone = _normalize(labs["testosterone"], norm_ranges["testosterone_ng_dl"]["min"], norm_ranges["testosterone_ng_dl"]["max"])
    norm_bmi = _normalize(labs["bmi"], norm_ranges["bmi"]["min"], norm_ranges["bmi"]["max"])

    feature_vector = [norm_lh_fsh, norm_insulin, norm_testosterone, norm_bmi]

    checks = {
        "fasting_insulin": "fasting_insulin_uiu_ml",
        "lh_fsh_ratio": "lh_fsh_ratio",
        "bmi": "bmi",
        "amh": "amh_ng_ml",
        "testosterone": "testosterone_ng_dl",
    }

    abnormal_markers = 0
    total_markers = 0
    for lab_key, config_key in checks.items():
        val = labs.get(lab_key, 0.0)
        thr = thresholds.get(config_key, {}).get("elevated_min")
        if val is not None and val > 0:
            total_markers += 1
            if val >= thr:
                abnormal_markers += 1

    pcos_support_markers = sum([
        clinical_flags["has_oligomenorrhea"],
        clinical_flags["has_hyperandrogenism"],
        clinical_flags["has_pom"],
    ])

    biomarker_support = abnormal_markers / max(total_markers, 1)
    pattern_support = pcos_support_markers / 3.0
    
    classical_support_score = round(float((0.50 * biomarker_support) + (0.50 * pattern_support)), 4)

    mimic_risk = any(k in clinical_flags["clinical_text"] for k in ["thyroid", "prolactin", "cushing", "virilization", "adrenal"])
    
    missing_tests = []
    if not any(k in clinical_flags["clinical_text"] for k in ["17-oh", "progesterone", "17oh"]):
        missing_tests.append("17-OH hydroxyprogesterone")
    if not any(k in clinical_flags["clinical_text"] for k in ["dhea", "dheas"]):
        missing_tests.append("DHEA-S")
    if "prolactin" not in clinical_flags["clinical_text"]:
        missing_tests.append("Serum prolactin")
    if not any(k in clinical_flags["clinical_text"] for k in ["tsh", "thyroid", "t4"]):
        missing_tests.append("TSH + Free T4")

    uncertainty_score = round(float(np.clip(0.10 + 0.10 * len(missing_tests) + (0.15 if mimic_risk else 0.0), 0.0, 1.0)), 4)
    pattern_alignment_score = round(float(np.clip((0.6 * classical_support_score) + (0.4 * biomarker_support), 0.0, 1.0)), 4)

    quantum_summary = run_quantum_analysis(feature_vector)
    quantum_score = quantum_summary.get("quantum_interaction_score", 0.0)

    if pcos_support_markers == 3:
        hypothesis_rank = "Confirmed Rotterdam PCOS Presentation"
    elif pcos_support_markers == 2:
        hypothesis_rank = "Probable PCOS Phenotype"
    else:
        hypothesis_rank = "Indeterminate Endocrine Pattern"

    classical_summary = {
        "classical_support_score": classical_support_score,
        "pattern_alignment_score": pattern_alignment_score,
        "uncertainty_score": uncertainty_score,
        "abnormal_markers": abnormal_markers,
        "total_markers": total_markers,
        "feature_vector": {
            "norm_lh_fsh": round(norm_lh_fsh, 4),
            "norm_insulin": round(norm_insulin, 4),
            "norm_testosterone": round(norm_testosterone, 4),
            "norm_bmi": round(norm_bmi, 4),
        },
    }

    ici_score = round(float(np.clip((0.5 * classical_support_score) + (0.3 * pattern_alignment_score) + (0.2 * quantum_score), 0.0, 1.0)), 4)
    interpretations = interpret_node3_results(classical_summary, quantum_summary, ici_score, mimic_risk)

    # FIXED: Explicit safety fallback check to completely bypass upstream explicit None assignments
    upstream_ici = state.get("ici_metrics")
    base_ici_metrics = upstream_ici if isinstance(upstream_ici, dict) else {}

    return {
        "classical_scores": {
            **classical_summary,
            "bayesian_credibility_score": classical_support_score,
            "confidence_interval_bounds": [
                round(max(0.0, classical_support_score - uncertainty_score / 3), 3),
                round(min(1.0, classical_support_score + uncertainty_score / 3), 3),
            ],
            "interpretation": interpretations["classical_interpretation"],
        },
        "quantum_scores": {
            **quantum_summary,
            "interpretation": interpretations["quantum_interpretation"],
        },
        "ici_metrics": {
            **base_ici_metrics,
            "integrated_clinical_index": ici_score,
            "interpretation": interpretations["ici_interpretation"],
        },
        "node3_summary": {
            "hypothesis_rank": hypothesis_rank,
            "mimic_risk": mimic_risk,
            "missing_tests": missing_tests,
            "uncertainty_interpretation": interpretations["uncertainty_interpretation"],
        },
    }