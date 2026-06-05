# pipeline/nodes/node6_rl_ranking.py
import os
import json
import numpy as np
from typing import Dict, Any, Literal
from pipeline.state import PCOSState

POLICY_FILE = "utils/rl_policy_matrix.json"

def load_policy_matrix() -> Dict[str, Any]:
    """Loads or initializes the persistent Q-table memory store."""
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, "r") as f:
            return json.load(f)
    # Default state discretizations linked to 2 discrete Agent Prompt Strategies:
    # Action 0: Conservative Homeostatic Analysis | Action 1: Aggressive Metabolic Analysis
    return {
        "Q_table": {
            "LOW_ENTROPY_LOW_VARIANCE": [0.5, 0.5],
            "LOW_ENTROPY_HIGH_VARIANCE": [0.5, 0.5],
            "HIGH_ENTROPY_LOW_VARIANCE": [0.5, 0.5],
            "HIGH_ENTROPY_HIGH_VARIANCE": [0.5, 0.5]
        },
        "learning_rate": 0.2,
        "discount_factor": 0.9,
        "total_episodes_trained": 0
    }

def save_policy_matrix(policy_data: Dict[str, Any]):
    """Saves the updated policy matrix back to disk."""
    os.makedirs(os.path.dirname(POLICY_FILE), exist_ok=True)
    with open(POLICY_FILE, "w") as f:
        json.dump(policy_data, f, indent=4)

def node6_rl_ranking_fn(state: PCOSState) -> Dict[str, Any]:
    """
    Node 6: Reinforcement Learning (RL) Policy Optimizer.
    Uses temporal-difference Q-learning mechanics to reward or penalize
    upstream multi-agent execution tracks, permanently saving state-action values.
    """
    print("\n" + "="*60)
    print("[NODE 6] ADVANCED Q-LEARNING POLICY MATRIX ENGINE ACTIVE...")
    print("="*60)
    
    # 1. Load running RL memory
    policy_memory = load_policy_matrix()
    q_table = policy_memory["Q_table"]
    alpha = policy_memory["learning_rate"]
    gamma = policy_memory["discount_factor"]
    
    # 2. Discretize the environment state space (S)
    xai_metrics = state.get("xai_metrics", {})
    entropy = xai_metrics.get("von_neumann_entropy", 0.0)
    variance_span = xai_metrics.get("variance_span", 0.0)
    
    state_str = ""
    state_str += "HIGH_ENTROPY_" if entropy > 2.5 else "LOW_ENTROPY_"
    state_str += "HIGH_VARIANCE" if variance_span > 0.5 else "LOW_VARIANCE"
    
    # 3. Track chosen Action (A) from Node 2 (Default to 0 if untracked)
    chosen_action = int(state.get("clinical_hypothesis", {}).get("selected_action_policy", 0))
    
    # 4. Compute Environmental Reward (R)
    # Objective: Minimize data contradiction (entropy/variance) while maximizing agent confidence
    confidence = float(state.get("clinical_hypothesis", {}).get("agent_confidence_level", "Medium") == "High")
    reward = 1.0
    if entropy > 2.5: reward -= 0.5
    if variance_span > 0.5: reward -= 0.3
    if confidence == 1.0: reward += 0.2
    
    # 5. Apply the Bellman Optimality Temporal Difference Update Step
    # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
    current_q = q_table[state_str][chosen_action]
    max_future_q = max(q_table[state_str]) # In standard environments, this looks at next state S'
    
    updated_q = current_q + alpha * (reward + (gamma * max_future_q) - current_q)
    q_table[state_str][chosen_action] = round(updated_q, 4)
    
    policy_memory["total_episodes_trained"] += 1
    save_policy_matrix(policy_memory)
    
    # 6. Evaluate Policy Success or Optimization Loop Requirement
    # If the updated action value drops too low, flag policy path as suboptimal
    is_optimal = updated_q >= 0.3
    
    state["rl_policy_metadata"] = {
        "calculated_reward": round(reward, 4),
        "current_state_discretization": state_str,
        "chosen_action_index": chosen_action,
        "updated_q_value": round(updated_q, 4),
        "policy_action_rank": "OPTIMAL_PATHWAY" if is_optimal else "SUBOPTIMAL_PATHWAY",
        "total_training_cycles": policy_memory["total_episodes_trained"]
    }
    
    print(f"[RL Bellman Step] State: {state_str} | Action: {chosen_action}")
    print(f"[RL Bellman Step] Reward Received: {reward:.2f} ➔ New Q-Value Matrix: {q_table[state_str]}")
    print(f"[RL Action Selection] Final Action Alignment Assessment: {state['rl_policy_metadata']['policy_action_rank']}")
    print("="*60 + "\n")
    return state

def rl_policy_router_fn(state: PCOSState) -> Literal["node2_hypothesis", "__end__"]:
    """RL router evaluated by the compiled execution workflow graph."""
    rl_meta = state.get("rl_policy_metadata", {})
    if rl_meta.get("policy_action_rank") == "SUBOPTIMAL_PATHWAY":
        print("[RL REWARD LOOP] Suboptimal policy return value. Routing trajectory back to Node 2 for hyperparameter adjustment.")
        return "node2_hypothesis"
    return "__end__"