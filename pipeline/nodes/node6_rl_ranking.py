# pipeline/nodes/node6_rl_ranking.py
import os
import json
import numpy as np
from typing import Dict, Any, Literal
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.state import PCOSState

POLICY_FILE = "utils/rl_policy_matrix.json"

# Shared embedding model for reward cosine similarity computation
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Four clinical states matching Node 2 _get_patient_state() exactly ────────
VALID_STATES = [
    "NORMAL_METABOLIC_NORMAL_GONADOTROPIN",
    "NORMAL_METABOLIC_HIGH_GONADOTROPIN",
    "HIGH_METABOLIC_NORMAL_GONADOTROPIN",
    "HIGH_METABOLIC_HIGH_GONADOTROPIN",
]


def load_policy_matrix() -> Dict[str, Any]:
    """Loads or initialises the persistent Q-table memory store."""
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, "r") as f:
            return json.load(f)

    return {
        "Q_table": {state: [0.5, 0.5] for state in VALID_STATES},
        "learning_rate":          0.2,
        "total_episodes_trained": 0,
    }


def save_policy_matrix(policy_data: Dict[str, Any]):
    """Saves the updated policy matrix back to disk."""
    os.makedirs(os.path.dirname(POLICY_FILE), exist_ok=True)
    with open(POLICY_FILE, "w") as f:
        json.dump(policy_data, f, indent=4)


def _get_patient_state(lh_fsh: float, insulin: float) -> str:
    """
    Mirrors the identical 4-state discretization used in Node 2
    so that action selection and Q-value updates always reference
    the same state key.
    """
    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            thresholds = json.load(f)["lab_thresholds"]
        insulin_threshold = float(thresholds["fasting_insulin_uiu_ml"]["elevated_min"])
        lh_fsh_threshold  = float(thresholds["lh_fsh_ratio"]["elevated_min"])
    except Exception:
        insulin_threshold = 14.0
        lh_fsh_threshold  = 2.0

    high_metabolic    = insulin >= insulin_threshold
    high_gonadotropin = lh_fsh  >= lh_fsh_threshold

    if high_metabolic and high_gonadotropin:
        return "HIGH_METABOLIC_HIGH_GONADOTROPIN"
    elif high_metabolic:
        return "HIGH_METABOLIC_NORMAL_GONADOTROPIN"
    elif high_gonadotropin:
        return "NORMAL_METABOLIC_HIGH_GONADOTROPIN"
    else:
        return "NORMAL_METABOLIC_NORMAL_GONADOTROPIN"


def _compute_reward(state: PCOSState) -> float:
    """
    Action-Dependent Reward Function (Version 2.0).

    Replaces the previous patient-severity-based reward (which measured how
    sick the patient was rather than how good the prompt strategy was).

    The new reward evaluates the *quality of the generated hypothesis* using
    two independent, programmatic signals:

    1. Literature Coherence Score (0 – 1, weight 0.70):
       Cosine similarity between the generated consensus hypothesis text and
       the retrieved PubMed/cache abstracts from Node 1.  A high score means
       the LLM hypothesis is semantically aligned with the evidence base.

    2. Phenotypic Consistency Bonus/Penalty (±0.30, weight 1.0 applied additively):
       If the patient's fasting insulin is NORMAL (< threshold) but the
       hypothesis mentions insulin resistance / metabolic syndrome, a penalty
       of -0.30 is applied.  If the hypothesis correctly avoids those terms,
       a +0.10 bonus is applied.  This catches hallucination errors.

    Final reward is clipped to [0.0, 1.0].
    """
    # ── 1. Literature coherence ───────────────────────────────────────────────
    hypothesis_dict  = state.get("clinical_hypothesis", {})
    hypothesis_text  = hypothesis_dict.get("clinical_hypothesis", "")

    retrieved_chunks = state.get("retrieved_chunks", [])
    paper_texts      = [
        c.get("text", "") for c in retrieved_chunks
        if isinstance(c, dict) and c.get("is_paper") and c.get("text")
    ]

    literature_score = 0.5  # neutral default if no papers available
    if hypothesis_text and paper_texts:
        try:
            hyp_vec  = _embedding_model.encode([hypothesis_text])
            lit_vecs = _embedding_model.encode(paper_texts)
            sims     = cosine_similarity(hyp_vec, lit_vecs).flatten()
            literature_score = float(np.mean(sims))
        except Exception as emb_err:
            print(f"[RL Reward] Embedding computation failed: {emb_err}. Using neutral 0.5.")

    # ── 2. Phenotypic consistency check ──────────────────────────────────────
    raw_patient       = state.get("raw_input", {})
    fasting_insulin   = float(raw_patient.get("fasting_insulin", 0.0))

    config_path = os.path.join("data", "pcos_thresholds.json")
    try:
        with open(config_path) as f:
            insulin_threshold = float(
                json.load(f)["lab_thresholds"]["fasting_insulin_uiu_ml"]["elevated_min"]
            )
    except Exception:
        insulin_threshold = 14.0

    consistency_adjustment = 0.0
    metabolic_keywords = [
        "insulin resistance", "metabolic syndrome", "hyperinsulinemia",
        "impaired glucose", "type 2 diabetes", "insulin resistant"
    ]
    hypothesis_lower = hypothesis_text.lower()
    mentions_metabolic = any(kw in hypothesis_lower for kw in metabolic_keywords)

    if fasting_insulin < insulin_threshold:
        if mentions_metabolic:
            # LLM hallucinated metabolic pathology for a normal-insulin patient
            consistency_adjustment = -0.30
            print("[RL Reward] Consistency PENALTY: hypothesis mentions metabolic issues "
                  f"but patient insulin ({fasting_insulin}) is NORMAL.")
        else:
            # Correctly avoided metabolic framing
            consistency_adjustment = +0.10
            print("[RL Reward] Consistency BONUS: hypothesis correctly avoids metabolic framing.")

    # ── 3. Combine and clip ───────────────────────────────────────────────────
    reward = float(np.clip(literature_score + consistency_adjustment, 0.0, 1.0))
    return round(reward, 4)


def node6_rl_ranking_fn(state: PCOSState) -> Dict[str, Any]:
    """
    Node 6: Reinforcement Learning (RL) Policy Optimizer — Version 2.0.

    Key changes from Version 1.0:
    ─────────────────────────────
    • State discretization: Now uses the identical 4-state clinical biomarker
      mapping as Node 2 (NORMAL/HIGH × METABOLIC/GONADOTROPIN).  The old
      entropy/variance states from downstream XAI telemetry have been removed
      because they were unavailable at action-selection time in Node 2,
      creating a state mismatch.

    • Reward function: Now measures the *quality of the generated hypothesis*
      (literature coherence + phenotypic consistency) rather than the patient's
      clinical severity.  The previous design meant 90 % of the reward was
      constant for any given patient regardless of which action was chosen.

    • Contextual Bandit update (γ = 0): The previous design used a full
      Bellman TD update with γ = 0.9, which bootstraps future Q-values.
      This is only valid in a sequential MDP where actions cause physical
      state transitions.  A single patient run is a single-step decision
      (Contextual Bandit), so γ = 0 is the correct formulation:
          Q(s, a) ← Q(s, a) + α * (R − Q(s, a))

    • Linear pipeline (no loop-back): The router always returns __end__.
      Learning accumulates across patient runs (persisted on disk) rather
      than within a single run, eliminating the double-LLM-debate latency.
    """
    print("\n" + "="*60)
    print("[NODE 6] RL POLICY OPTIMIZER (Contextual Bandit v2.0)...")
    print("="*60)

    policy_memory = load_policy_matrix()
    q_table       = policy_memory["Q_table"]
    alpha         = policy_memory["learning_rate"]

    # Ensure Q-table contains all four valid states (backwards-compatibility)
    for s in VALID_STATES:
        if s not in q_table:
            q_table[s] = [0.5, 0.5]

    # ── Retrieve patient values to identify the current state ─────────────────
    raw_patient   = state.get("raw_input", {})
    lh_fsh_val    = float(raw_patient.get("lh_fsh_ratio",   2.1))
    insulin_val   = float(raw_patient.get("fasting_insulin", 14.2))
    current_state = _get_patient_state(lh_fsh_val, insulin_val)

    # ── Retrieve the action chosen by Node 2 ─────────────────────────────────
    chosen_action = int(
        state.get("clinical_hypothesis", {}).get("selected_action_policy", 0)
    )

    # ── Compute action-dependent reward ──────────────────────────────────────
    reward = _compute_reward(state)

    # ── Contextual Bandit Q-update (γ = 0, single-step) ──────────────────────
    old_q = q_table[current_state][chosen_action]
    new_q = old_q + alpha * (reward - old_q)
    new_q = round(float(new_q), 4)
    q_table[current_state][chosen_action] = new_q

    print(f"[RL Update] State        : {current_state}")
    print(f"[RL Update] Action chosen: Path {chosen_action}")
    print(f"[RL Update] Reward R(s,a): {reward:.4f}")
    print(f"[RL Update] Q-value      : {old_q:.4f} → {new_q:.4f}  (α={alpha})")

    # ── Persist updated policy matrix ────────────────────────────────────────
    policy_memory["total_episodes_trained"] += 1
    save_policy_matrix(policy_memory)

    # ── Write RL metadata to state for dashboard display ─────────────────────
    state["rl_policy_metadata"] = {
        "calculated_reward":            reward,
        "current_state_discretization": current_state,
        "chosen_action_index":          chosen_action,
        "updated_q_value":              new_q,
        "policy_action_rank":           "OPTIMAL_PATHWAY" if new_q >= 0.5 else "SUBOPTIMAL_PATHWAY",
        "total_training_cycles":        policy_memory["total_episodes_trained"],
    }

    print(f"[RL Update] Total patient episodes trained: {policy_memory['total_episodes_trained']}")
    print("="*60 + "\n")
    return state


def rl_policy_router_fn(state: PCOSState) -> Literal["__end__"]:
    """
    Linear pipeline router — Version 2.0.

    Always routes to __end__.  Learning is persisted across separate patient
    runs (episodes) via the Q-table on disk.  Within a single patient run,
    the pipeline executes exactly once, eliminating the double-LLM-debate
    latency of the V1.0 loop-back design.
    """
    print("[RL ROUTER] Linear pipeline: routing to END. "
          "Q-table updated on disk for next patient run.")
    return "__end__"