import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

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

def run_quantum_analysis(normalized_features: list) -> dict:
    """
    Executes a Parameterized Ansatz Variational Quantum Circuit (VQC).

    Input: 4 min-max normalized biomarker values in [0.0, 1.0]
        [norm_lh_fsh, norm_insulin, norm_testosterone, norm_bmi]

    NOTE: Fixed ansatz parameters used throughout (non-trained, research prototype).
    The ZZ correlator between Q0 (Gonadotropin) and Q3 (Adipose) measures
    the cross-axis entanglement signature between reproductive and metabolic poles.
    """
    # ── Input sanitization ──────────────────────────────────────────────────
    features = np.array(normalized_features[:4], dtype=float)
    if len(features) < 4:
        features = np.pad(features, (0, 4 - len(features)), constant_values=0.5)

    # Clip to [0, 1] before scaling to rotation angles
    features = np.clip(features, 0.0, 1.0)

    # Map [0, 1] → [0, π] for Bloch sphere rotation angles
    angles = features * np.pi

    print(f"[QuantumVQC] Input biomarker angles (rad): "
          f"LH/FSH={angles[0]:.3f}, Insulin={angles[1]:.3f}, "
          f"Testosterone={angles[2]:.3f}, BMI={angles[3]:.3f}")

    # ── Circuit construction ────────────────────────────────────────────────
    qc = QuantumCircuit(4)

    # Layer A: Quantum Feature Map (Hadamard superposition + Rx encoding)
    # Each qubit encodes one normalized biomarker as a rotation angle
    for i in range(4):
        qc.h(i)
        qc.rx(angles[i], i)

    # Layer B: Fixed Ansatz Projection Layer
    # NOTE: Non-trained fixed weights — research prototype
    # These project the encoded state into a richer Hilbert subspace
    theta_weights = [0.15, 0.42, 0.88, 0.31]
    for i in range(4):
        qc.ry(theta_weights[i], i)

    # Layer C: Strongly Entangling Cyclic Topology
    # Linear chain captures adjacent biomarker correlations
    # Cyclic link Q3→Q0 captures the metabolic↔reproductive cross-axis
    for i in range(3):
        qc.cx(i, i + 1)
    qc.cx(3, 0)  # Cyclic: Adipose Mass ↔ Gonadotropin Axis

    # Layer D: Final Variational Rotation (depth extension)
    for i in range(4):
        qc.rz(0.25 * (i + 1), i)

    qc.measure_all()

    # ── Execution ───────────────────────────────────────────────────────────
    simulator = AerSimulator()
    job    = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts()

    # ── ZZ Correlator: Q0 (Gonadotropin) ↔ Q3 (Adipose) ───────────────────
    # Measures cross-axis entanglement between reproductive and metabolic poles
    # Qiskit little-endian: rightmost char = qubit 0, leftmost = qubit 3
    # +1 if correlated (same state), -1 if anti-correlated (different states)
    zz_expectation = 0.0
    for bitstring, count in counts.items():
        q0 = int(bitstring[3])   # rightmost → qubit 0 (Gonadotropin)
        q3 = int(bitstring[0])   # leftmost  → qubit 3 (Adipose)
        sign = 1.0 if q0 == q3 else -1.0
        zz_expectation += sign * (count / 1024)

    # Map ZZ expectation [-1, 1] → quantum interaction score [0, 1]
    quantum_score = float(np.clip((zz_expectation + 1.0) / 2.0, 0.0, 1.0))

    # ── Per-qubit marginal probabilities ────────────────────────────────────
    # Probability that each biological axis is in the |1⟩ (activated) state
    # Used by Node 5 Layer 2 for per-axis entropy decoding
    qubit_activation = _compute_qubit_marginals(counts)

    # ── Top-3 states for Node 5 quantum narrative decoding ──────────────────
    top_states = _get_top_states(counts, n=3)

    # ── Von Neumann entropy (Shannon approximation over measurement dist) ───
    entropy = _compute_entropy(counts)

    print(f"[QuantumVQC] ZZ expectation: {zz_expectation:.4f} → "
          f"interaction score: {quantum_score:.4f}")
    print(f"[QuantumVQC] Entropy: {entropy:.4f} bits | "
          f"Dominant state: {top_states[0]['state']} "
          f"({top_states[0]['probability']:.3f})")

    return {
        # Primary score (read by Node 3 ICI fusion)
        "quantum_interaction_score":  round(quantum_score, 4),
        "quantum_plausibility_score": round(quantum_score, 4),  # alias for app.py

        # Measurement statistics (read by Node 5 Layer 2)
        "raw_counts":           counts,
        "top_states":           top_states,
        "qubit_activation":     qubit_activation,
        "von_neumann_entropy":  round(entropy, 4),

        # Circuit metadata (read by Node 5 XAI report)
        "dominant_state_frequency": round(max(counts.values()) / 1024, 4),
        "zz_expectation_value":     round(zz_expectation, 4),
        "qubit_biology_map":        QUBIT_BIOLOGY_MAP,
        "circuit_depth":            qc.depth(),
        "shots":                    1024
    }


def _compute_qubit_marginals(counts: dict) -> dict:
    """
    Computes marginal probability of each qubit being in |1⟩ state.
    Qiskit little-endian: bitstring[-1] = q0, bitstring[0] = q3.
    Returns dict keyed by biology label.
    """
    total = sum(counts.values())
    # 4 qubits → index positions in little-endian bitstring
    # qubit i is at position (3 - i) from the left, or position i from the right
    activation = {i: 0.0 for i in range(4)}

    for bitstring, count in counts.items():
        padded = bitstring.zfill(4)  # ensure always 4 chars
        for qubit_idx in range(4):
            # little-endian: qubit 0 is rightmost (index 3), qubit 3 is leftmost (index 0)
            bit = int(padded[3 - qubit_idx])
            activation[qubit_idx] += bit * count

    result = {}
    for qubit_idx, raw_count in activation.items():
        label = QUBIT_BIOLOGY_MAP[qubit_idx]
        result[label] = round(raw_count / total, 4)

    return result


def _get_top_states(counts: dict, n: int = 3) -> list:
    """
    Returns top-n most frequent measurement outcomes with decoded biology.
    Each entry includes the bitstring, probability, and per-qubit biological interpretation.
    """
    total = sum(counts.values())
    sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    top = []
    for bitstring, count in sorted_states[:n]:
        padded = bitstring.zfill(4)
        probability = round(count / total, 4)

        # Decode each qubit's state into biological meaning
        decoded_axes = {}
        for qubit_idx in range(4):
            bit = int(padded[3 - qubit_idx])   # little-endian
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
    """
    Computes Shannon entropy over measurement probability distribution.
    Approximates von Neumann entropy of the output state.
    S = -sum(p_i * log2(p_i))
    Returns entropy in bits. Max for 4 qubits = 4.0 bits (uniform distribution).
    """
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    return float(entropy)