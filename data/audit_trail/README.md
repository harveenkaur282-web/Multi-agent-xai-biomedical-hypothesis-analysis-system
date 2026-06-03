# Audit Trail

Each pipeline run writes one JSON file here named:
`{patient_id}_{timestamp}.json`

Schema per file:
- input_payload: raw case JSON
- node1_chunks: top-10 retrieved documents with hybrid scores  
- node2_consensus: full hypothesis + agent confidence scores
- node3_classical: Bayesian credibility score + CI bounds
- node3_quantum: quantum interaction score + raw counts
- node4_fused: weighted fusion result
- run_metadata: timestamps, model version, total duration