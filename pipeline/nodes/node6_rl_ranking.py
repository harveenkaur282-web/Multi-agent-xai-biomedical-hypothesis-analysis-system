# pipeline/nodes/node6_rl_ranking.py
import os
import json
from typing import Dict, Any, Literal
from pipeline.state import PCOSState

POLICY_FILE = "utils/rl_policy_matrix.json"

def load_policy_matrix() -> Dict[str, Any]:
    """Loads or initializes the persistent Q-table memory store."""
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, "r") as f:
            return json.load(f)
 
    return {
        "Q_table": {
            "LOW_ENTROPY_LOW_VARIANCE": [0.5, 0.5],
            "LOW_ENTROPY_HIGH_VARIANCE": [0.5, 0.5],
            "HIGH_ENTROPY_LOW_VARIANCE": [0.5, 0.5],
            "HIGH_ENTROPY_HIGH_VARIANCE": [0.5, 0.5]
        },
        "learning_rate": 0.2,
        "discount_factor": 0.9,
        "total_episodes_trained": 0,
        "last_state": None,       
        "last_action": None       
    }

def save_policy_matrix(policy_data: Dict[str, Any]):
    """Saves the updated policy matrix back to disk."""
    os.makedirs(os.path.dirname(POLICY_FILE), exist_ok=True)
    with open(POLICY_FILE, "w") as f:
        json.dump(policy_data, f, indent=4)

def node6_rl_ranking_fn(state: PCOSState) -> Dict[str, Any]:
    """
    Node 6: Reinforcement Learning (RL) Policy Optimizer.
    Correctly computes Bellman temporal difference tracking using S and S'.
    """
    print("\n" + "="*60)
    print("[NODE 6] ADVANCED Q-LEARNING POLICY MATRIX ENGINE ACTIVE...")
    print("="*60)
    
    policy_memory = load_policy_matrix()
    q_table = policy_memory["Q_table"]
    alpha = policy_memory["learning_rate"]
    gamma = policy_memory["discount_factor"]

    xai_metrics = state.get("xai_metrics", {})
    entropy = xai_metrics.get("von_neumann_entropy", 0.0)
    variance_span = xai_metrics.get("variance_span", 0.0)
    
    current_state_str = ""
    current_state_str += "HIGH_ENTROPY_" if entropy > 2.5 else "LOW_ENTROPY_"
    current_state_str += "HIGH_VARIANCE" if variance_span > 0.5 else "LOW_VARIANCE"

    last_state = policy_memory.get("last_state")
    last_action = policy_memory.get("last_action")
    
    chosen_action = int(state.get("clinical_hypothesis", {}).get("selected_action_policy", 0))
    
    confidence = float(state.get("clinical_hypothesis", {}).get("agent_confidence_level", "Medium") == "High")
    reward = 1.0
    if entropy > 2.5: reward -= 0.5
    if variance_span > 0.5: reward -= 0.3
    if confidence == 1.0: reward += 0.2
    
    updated_q = 0.5 
    
    # bellman temporal difference step
    # If a previous state exists, update the transition value: Q(s, a) -> next state s'
    if last_state and last_action is not None:
        max_future_q = max(q_table[current_state_str]) # max_a Q(s', a)
        old_q = q_table[last_state][last_action]
        
        # Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        new_q = old_q + alpha * (reward + (gamma * max_future_q) - old_q)
        q_table[last_state][last_action] = round(new_q, 4)
        updated_q = new_q
        print(f"[RL Bellman Step] Updated historical path Q({last_state}, Action {last_action}) using transition to {current_state_str}.")
    else:

        old_q = q_table[current_state_str][chosen_action]
        new_q = old_q + alpha * (reward - old_q)
        q_table[current_state_str][chosen_action] = round(new_q, 4)
        updated_q = new_q
        print("[RL Cold Boot] Updating current state matrix directly due to lack of predecessor history.")

    policy_memory["last_state"] = current_state_str
    policy_memory["last_action"] = chosen_action
    policy_memory["total_episodes_trained"] += 1
    save_policy_matrix(policy_memory)

    is_optimal = updated_q >= 0.3
    
    state["rl_policy_metadata"] = {
        "calculated_reward": round(reward, 4),
        "current_state_discretization": current_state_str,
        "chosen_action_index": chosen_action,
        "updated_q_value": round(updated_q, 4),
        "policy_action_rank": "OPTIMAL_PATHWAY" if is_optimal else "SUBOPTIMAL_PATHWAY",
        "total_training_cycles": policy_memory["total_episodes_trained"]
    }
    
    print(f"[RL Metrics] Active State: {current_state_str} | Evaluated Action: {chosen_action}")
    print(f"[RL Metrics] Calculated Reward: {reward:.2f} ➔ Q-Value Array: {q_table[current_state_str]}")
    print("="*60 + "\n")
    return state

def rl_policy_router_fn(state: PCOSState) -> Literal["node2_hypothesis", "__end__"]:
    """RL router evaluated by the compiled execution workflow graph."""
    rl_meta = state.get("rl_policy_metadata", {})
    if rl_meta.get("policy_action_rank") == "SUBOPTIMAL_PATHWAY":
        print("[RL REWARD LOOP] Suboptimal policy return value. Routing trajectory back to Node 2 for hyperparameter adjustment.")
        return "node2_hypothesis"
    return "__end__"