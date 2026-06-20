import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

QUBIT_BIOLOGY_MAP = {
    0: "Gonadotropin Axis (LH/FSH)",
    1: "Insulin Pathway (Fasting Insulin)",
    2: "Hyperandrogenism (Testosterone)",
    3: "Adipose Mass (BMI)",
}

def create_parameterized_circuit(num_qubits: int = 4):
    qreg = QuantumRegister(num_qubits, "q")
    creg = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qreg, creg)
    theta = ParameterVector("theta", num_qubits)

    for i in range(num_qubits):
        qc.h(qreg[i])
    for i in range(num_qubits):
        qc.ry(theta[i], qreg[i])
    for i in range(num_qubits - 1):
        qc.cx(qreg[i], qreg[i + 1])
    qc.cx(qreg[3], qreg[0])
    for i in range(num_qubits):
        qc.rz(0.25 * (i + 1), qreg[i])

    qc.measure(qreg, creg)
    return qc, theta

def get_zz_expectation(counts: dict, shots: int) -> float:
    zz_val = 0.0
    for bitstring, count in counts.items():
        padded = bitstring.zfill(4)
        q0 = int(padded[3])
        q3 = int(padded[0])
        sign = 1.0 if q0 == q3 else -1.0
        zz_val += sign * (count / shots)
    return float(zz_val)


def _execute_circuit(qc, bound_params, angles, shots=512):
    qreg = QuantumRegister(4, "e")
    encode_qc = QuantumCircuit(qreg)
    for i in range(4):
        encode_qc.rx(float(angles[i]), qreg[i])

    full_qc = encode_qc.compose(qc)
    bound_qc = full_qc.assign_parameters(bound_params)
    simulator = AerSimulator()
    job = simulator.run(bound_qc, shots=shots)
    return job.result().get_counts()


def _compute_qubit_marginals(counts: dict) -> dict:
    total = sum(counts.values()) or 1
    activation = {i: 0.0 for i in range(4)}
    for bitstring, count in counts.items():
        padded = bitstring.zfill(4)
        for qubit_idx in range(4):
            bit = int(padded[3 - qubit_idx])
            activation[qubit_idx] += bit * count
    return {QUBIT_BIOLOGY_MAP[i]: round(activation[i] / total, 4) for i in range(4)}


def _get_top_states(counts: dict, n: int = 3) -> list:
    total = sum(counts.values()) or 1
    top = []
    for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]:
        padded = bitstring.zfill(4)
        decoded = {}
        for qubit_idx in range(4):
            bit = int(padded[3 - qubit_idx])
            decoded[QUBIT_BIOLOGY_MAP[qubit_idx]] = "ACTIVATED" if bit == 1 else "baseline"
        top.append(
            {
                "state": padded,
                "count": int(count),
                "probability": round(count / total, 4),
                "decoded": decoded,
            }
        )
    return top


def _compute_entropy(counts: dict) -> float:
    total = sum(counts.values()) or 1
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return float(entropy)


def run_quantum_analysis(normalized_features: list, seed: int = 42, shots: int = 1024) -> dict:
    rng = np.random.default_rng(seed)
    synthetic_anchors = [
        {"features": [0.1, 0.1, 0.1, 0.1], "target_zz": -1.0},
        {"features": [0.9, 0.9, 0.9, 0.9], "target_zz": 1.0},
    ]

    vqc, theta = create_parameterized_circuit()

    def cost_function(params):
        mse_loss = 0.0
        for anchor in synthetic_anchors:
            angles = np.array(anchor["features"]) * np.pi
            counts = _execute_circuit(vqc, params, angles, shots=256)
            zz = get_zz_expectation(counts, 256)
            mse_loss += (zz - anchor["target_zz"]) ** 2
        return mse_loss / len(synthetic_anchors)

    init_params = rng.uniform(0, 2 * np.pi, 4)

    # FIX: Increased maxiter from 15 to 80 and adjusted tolerance threshold
    res = minimize(
        cost_function, 
        init_params, 
        method="COBYLA", 
        options={"maxiter": 80, "tol": 0.001} 
    )
    optimized_weights = res.x

    features = np.array(normalized_features[:4], dtype=float)
    if len(features) < 4:
        features = np.pad(features, (0, 4 - len(features)), constant_values=0.5)
    features = np.clip(features, 0.0, 1.0)
    patient_angles = features * np.pi

    final_counts = _execute_circuit(vqc, optimized_weights, patient_angles, shots=shots)
    zz_patient = get_zz_expectation(final_counts, shots)
    quantum_score = float(np.clip((zz_patient + 1.0) / 2.0, 0.0, 1.0))

    return {
        "quantum_interaction_score": round(quantum_score, 4),
        "quantum_plausibility_score": round(quantum_score, 4),
        "training_metadata": {
            "final_loss": round(float(res.fun), 4),
            "iterations": int(res.nfev),
            "optimized_weights": [round(float(w), 4) for w in optimized_weights],
        },
        "raw_counts": final_counts,
        "top_states": _get_top_states(final_counts, n=3),
        "qubit_activation": _compute_qubit_marginals(final_counts),
        "von_neumann_entropy": round(_compute_entropy(final_counts), 4),
        "dominant_state_frequency": round(max(final_counts.values()) / shots, 4),
        "zz_expectation_value": round(float(zz_patient), 4),
        "qubit_biology_map": QUBIT_BIOLOGY_MAP,
        "circuit_depth": int(vqc.depth() + 1),
        "shots": int(shots),
    }