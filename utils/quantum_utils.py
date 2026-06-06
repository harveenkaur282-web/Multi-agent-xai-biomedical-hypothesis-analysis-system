import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# Qubit-to-biology mapping (aligned with node3_dual_analysis.py feature vector)
# Qubit 0 → Gonadotropin Axis    (norm_lh_fsh_ratio)
# Qubit 1 → Insulin Pathway      (norm_fasting_insulin)
# Qubit 2 → Hyperandrogenism     (norm_testosterone)
# Qubit 3 → Adipose Mass         (norm_bmi)
QUBIT_BIOLOGY_MAP = {
    0: "Gonadotropin Axis (LH/FSH)",
    1: "Insulin Pathway (Fasting Insulin)",
    2: "Hyperandrogenism (Testosterone)",
    3: "Adipose Mass (BMI)"
}

def create_parameterized_circuit(num_qubits: int = 4):
    """Creates a Variational Quantum Circuit (VQC) with ParameterVector for training."""
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector('theta', num_qubits)
    
    # Base superposition
    for i in range(num_qubits):
        qc.h(i)
        
    # Variational rotation layer (to be optimized)
    for i in range(num_qubits):
        qc.ry(theta[i], i)
        
    # Strongly entangling topology (linear + cyclic Q3->Q0 cross-axis)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(3, 0)
    
    # Final depth
    for i in range(num_qubits):
        qc.rz(0.25 * (i + 1), i)
        
    qc.measure_all()
    return qc, theta

def get_zz_expectation(counts: dict, shots: int) -> float:
    """Calculates ZZ expectation value between Q0 and Q3."""
    zz_val = 0.0
    for bitstring, count in counts.items():
        padded = bitstring.zfill(4)
        q0 = int(padded[3])
        q3 = int(padded[0])
        sign = 1.0 if q0 == q3 else -1.0
        zz_val += sign * (count / shots)
    return zz_val

def _execute_circuit(qc, bound_params, angles, shots=512):
    """Helper to bind and execute the circuit."""
    # First, append data encoding (Rx rotations based on features)
    encode_qc = QuantumCircuit(4)
    for i in range(4):
        encode_qc.rx(angles[i], i)
    
    # Combine encoding with parameterized VQC
    full_qc = encode_qc.compose(qc)
    bound_qc = full_qc.assign_parameters(bound_params)
    
    simulator = AerSimulator()
    job = simulator.run(bound_qc, shots=shots)
    return job.result().get_counts()

def run_quantum_analysis(normalized_features: list) -> dict:
    """
    Executes an Inference-Time Anchor Calibration using Reference Phenotypic Ensembles.
    Uses COBYLA optimizer to dynamically train VQC parameters against static extremes
    to map the Hilbert space effectively before classifying the patient.
    """
    print("\n[QuantumVQC] Initializing Inference-Time Anchor Calibration using Reference Phenotypic Ensembles...")
    
    # ── 1. Static Hardcoded Deterministic Clinical Anchors ───────────────────
    # Zero leakage reference dataset for boundary definitions
    # Format: [LH/FSH, Insulin, Test, BMI] mapped to [0, pi] angles
    synthetic_anchors = [
        {"features": [0.1, 0.1, 0.1, 0.1], "target_zz": -1.0, "label": "Healthy Optimal"},
        {"features": [0.9, 0.9, 0.9, 0.9], "target_zz": 1.0,  "label": "Severe PCOS Variant"}
    ]
    
    vqc, theta = create_parameterized_circuit()
    
    # ── 2. Loss Function for Optimization ───────────────────────────────────
    def cost_function(params):
        mse_loss = 0.0
        for anchor in synthetic_anchors:
            angles = np.array(anchor["features"]) * np.pi
            counts = _execute_circuit(vqc, params, angles, shots=512)
            zz = get_zz_expectation(counts, 512)
            mse_loss += (zz - anchor["target_zz"])**2
        return mse_loss / len(synthetic_anchors)
        
    # ── 3. Execute COBYLA VQC Training ──────────────────────────────────────
    init_params = np.random.uniform(0, 2*np.pi, 4)
    
    res = minimize(
        cost_function,
        init_params,
        method='COBYLA',
        options={'maxiter': 15, 'tol': 0.01}
    )
    
    optimized_weights = res.x
    training_loss = res.fun
    
    print(f"[QuantumVQC] Anchor Calibration Complete. Final Loss: {training_loss:.4f} | Evals: {res.nfev}")
    
    # ── 4. Patient Inference ────────────────────────────────────────────────
    # Enforce global scaling matrix implicitly through the normalized input
    features = np.array(normalized_features[:4], dtype=float)
    features = np.clip(np.pad(features, (0, 4 - len(features)), constant_values=0.5), 0.0, 1.0)
    patient_angles = features * np.pi
    
    print(f"[QuantumVQC] Running patient inference with optimized weights: LH/FSH={patient_angles[0]:.3f}, Ins={patient_angles[1]:.3f}")
    
    final_counts = _execute_circuit(vqc, optimized_weights, patient_angles, shots=1024)
    zz_patient = get_zz_expectation(final_counts, 1024)
    
    # Map ZZ [-1, 1] -> [0, 1] interaction score
    quantum_score = float(np.clip((zz_patient + 1.0) / 2.0, 0.0, 1.0))
    
    qubit_activation = _compute_qubit_marginals(final_counts)
    top_states = _get_top_states(final_counts, n=3)
    entropy = _compute_entropy(final_counts)
    
    return {
        "quantum_interaction_score":  round(quantum_score, 4),
        "quantum_plausibility_score": round(quantum_score, 4),
        "training_metadata": {
            "final_loss": round(float(training_loss), 4),
            "iterations": res.nfev,
            "optimized_weights": [round(float(w), 4) for w in optimized_weights]
        },
        "raw_counts":           final_counts,
        "top_states":           top_states,
        "qubit_activation":     qubit_activation,
        "von_neumann_entropy":  round(entropy, 4),
        "dominant_state_frequency": round(max(final_counts.values()) / 1024, 4),
        "zz_expectation_value":     round(zz_patient, 4),
        "qubit_biology_map":        QUBIT_BIOLOGY_MAP,
        "circuit_depth":            vqc.depth() + 1,
        "shots":                    1024
    }

def _compute_qubit_marginals(counts: dict) -> dict:
    total = sum(counts.values())
    activation = {i: 0.0 for i in range(4)}
    for bitstring, count in counts.items():
        padded = bitstring.zfill(4)
        for qubit_idx in range(4):
            bit = int(padded[3 - qubit_idx])
            activation[qubit_idx] += bit * count
    result = {}
    for qubit_idx, raw_count in activation.items():
        label = QUBIT_BIOLOGY_MAP[qubit_idx]
        result[label] = round(raw_count / total, 4)
    return result

def _get_top_states(counts: dict, n: int = 3) -> list:
    total = sum(counts.values())
    sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = []
    for bitstring, count in sorted_states[:n]:
        padded = bitstring.zfill(4)
        probability = round(count / total, 4)
        decoded_axes = {}
        for qubit_idx in range(4):
            bit = int(padded[3 - qubit_idx])
            label = QUBIT_BIOLOGY_MAP[qubit_idx]
            decoded_axes[label] = "ACTIVATED" if bit == 1 else "baseline"
        top.append({
            "state":       padded,
            "count":       count,
            "probability": probability,
            "decoded":     decoded_axes
        })
    return top

def _compute_entropy(counts: dict) -> float:
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return float(entropy)