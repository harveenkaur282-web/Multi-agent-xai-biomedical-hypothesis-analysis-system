import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def run_quantum_analysis(normalized_features: list) -> dict:
    """
    Executes a Parameterized Ansatz Quantum Circuit to evaluate feature interaction.
    NOTE: Fixed ansatz parameters are used here (non-trained, research prototype layout).
    """
    features = np.array(normalized_features[:4])
    if len(features) < 4:
        features = np.pad(features, (0, 4 - len(features)), 'constant')
        
    angles = features * np.pi
    qc = QuantumCircuit(4)
    
    # LAYER A: Quantum Feature Map
    for i in range(4):
        qc.h(i)
        qc.rx(angles[i], i)
        
    # LAYER B: Parameterized Ansatz (Fixed weights for state projection)
    theta_weights = [0.15, 0.42, 0.88, 0.31] 
    for i in range(4):
        qc.ry(theta_weights[i], i)
        
    # LAYER C: Strongly Entangling Cyclic Topology
    for i in range(3):
        qc.cx(i, i+1)
    qc.cx(3, 0) # Cyclic boundary link (connects Q0 and Q3)
    
    # LAYER D: Final Variational Rotation
    for i in range(4):
        qc.rz(0.25 * (i + 1), i)
        
    qc.measure_all()
    
    # Execution
    simulator = AerSimulator()
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts()
    
    # ──── EXPECTATION VALUE OF ZZ CORRELATOR BETWEEN Q0 AND Q3 ────
    zz_expectation = 0.0
    for bitstring, count in counts.items():
        # Qiskit is little-endian: index 3 is Qubit 0 (rightmost), index 0 is Qubit 3 (leftmost)
        q0 = int(bitstring[3])
        q3 = int(bitstring[0])
        
        # +1 if correlated (both 0 or both 1), -1 if anti-correlated (different)
        sign = 1.0 if q0 == q3 else -1.0
        zz_expectation += sign * (count / 1024)
        
    # Map expectation value from [-1.0, 1.0] seamlessly to a [0.0, 1.0] score
    quantum_score = float((zz_expectation + 1.0) / 2.0)
    final_score = round(min(quantum_score, 1.0), 4)
    
    return {
        "quantum_interaction_score": final_score,
        "quantum_plausibility_score": final_score,  # FIX: Aligned to prevent app.py frontend KeyError
        "dominant_state_frequency": round(max(counts.values()) / 1024, 4),
        "raw_counts": counts
    }