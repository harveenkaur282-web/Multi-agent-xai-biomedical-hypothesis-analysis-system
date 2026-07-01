import pytest
from pipeline.state import PCOSState
from pipeline.nodes.node_judge import node_judge_fn

def test_node_judge_success():
    # Setup mock state representing a mock patient and hypothesis
    state = PCOSState(
        raw_input={
            "age": 27,
            "bmi": 24.5,
            "lh_fsh_ratio": 2.1,
            "fasting_insulin": 14.2,
            "amh_levels": 4.2,
            "free_testosterone": 55.0,
            "clinical_remarks": "Oligomenorrhea, acne, polycystic ovary morphology on ultrasound",
            "llm_choice": "Ollama Local"
        },
        clinical_hypothesis={
            "phenotype_assessment": "lean hyperandrogenic PCOS phenotype, with possible neuroendocrine dominance",
            "clinical_hypothesis": "The patient profile is consistent with lean hyperandrogenic PCOS, driven by neuroendocrine dysfunction.",
            "primary_risk_factor": "Androgen Excess / Neuroendocrine Axis",
            "agent_confidence_level": "High",
            "recommended_biomarkers": ["17-OH progesterone", "DHEA-S", "prolactin", "TSH", "free T4"],
            "pcos_diagnosis_likely": True
        },
        retrieved_chunks=[
            {
                "id": "[Source-1, Chunk-1]",
                "title": "LH/FSH ratio in lean PCOS",
                "text": "Elevated LH/FSH ratio and hyperandrogenism are hallmark signs of lean PCOS phenotype.",
                "is_paper": True
            }
        ]
    )

    # Run the judge node function
    result_state = node_judge_fn(state)

    # Verify metrics are stored
    assert "llm_judge_evaluation" in result_state
    eval_data = result_state["llm_judge_evaluation"]
    assert "scores" in eval_data
    assert "table_markdown" in eval_data
    assert "rag_eval_metrics" in result_state

    # Verify scores are bounded between 0 and 100 or None for RAG metrics
    scores = eval_data["scores"]
    for metric in ["rotterdam_accuracy", "clinical_correlation", "differential_logic"]:
        assert metric in scores
        assert scores[metric] is not None
        assert 0 <= scores[metric] <= 100
    for metric in ["faithfulness", "context_relevance", "answer_relevance"]:
        assert metric in scores
        assert scores[metric] is None or (0 <= scores[metric] <= 100)

    # Verify table markdown contains the headers and correct keys
    table = eval_data["table_markdown"]
    assert "Evaluation Dimension" in table
    assert "Score" in table
    assert "Rotterdam Criteria Accuracy" in table
