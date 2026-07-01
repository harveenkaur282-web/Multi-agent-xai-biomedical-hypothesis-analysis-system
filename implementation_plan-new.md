# PCOS Multi-Agent XAI Clinical Argumentation Implementation Plan

This plan outlines the architectural transition of the Explainable AI (XAI) engine from a synthetic SHAP surrogate model to an **agent-native clinical argumentation system** inspired by the research paper **"ArgMed-Agents: Explainable Clinical Decision Reasoning with LLM Discussion via Argumentation Schemes"**.

---

## User Review Required

> [!IMPORTANT]
> **Shift from SHAP to Argumentation Theory**: We will deprecate the local Logistic Regression surrogate and SHAP feature attribution in Node 5. In its place, we will implement a formal **Clinical Argumentation Graph** representing the claims made by the agents, conflict relationships (attacks) arising from missing biomarkers or mimic risks, and their resolution using abstract argumentation principles.
> This matches the project's title and academic rigor much more closely than synthetic data SHAP plots.

---

## Proposed Changes

### 1. State Expansion
We will update the state dictionary to hold the raw debate transcripts/outputs from the individual CrewAI agents.

#### [MODIFY] [state.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/state.py)
* Add `debate_history`: Dictionary containing the raw text outputs of individual tasks (reproductive, metabolic, differential) to build the debate lineage.

---

### 2. Debate History Capture in Node 2 & Crew Orchestration
We will retrieve the intermediate agent outputs from CrewAI tasks and store them in the state.

#### [MODIFY] [crew_setup.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/agents/crew_setup.py)
* Modify `run_pcos_debate` to return a tuple or unified dictionary:
  ```python
  return {
      "consensus": consensus_json,
      "debate_history": {
          "reproductive": task1_reproductive.output.raw,
          "metabolic": task2_metabolic.output.raw,
          "differential": task3_differential.output.raw
      }
  }
  ```

#### [MODIFY] [node2_hypothesis.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node2_hypothesis.py)
* Save the `debate_history` key returned by the crew into the Graph state.

---

### 3. Argumentation Scheme & Conflict Solver (Node 5)
We will replace the SHAP-related synthetic normal generators in Node 5 with a symbolic argumentation solver.

#### [MODIFY] [node5_xai.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node5_xai.py)
* **Argument Scheme Formulation**:
  Define a set of Clinical Arguments (nodes) representing the case findings:
  * **A1 (Endocrine Claim)**: Represents the hypothesis of reproductive PCOS (LH/FSH, AMH).
  * **A2 (Metabolic Claim)**: Represents the metabolic risk hypotheis (Insulin, BMI).
  * **A3 (Missing Biomarker Attack)**: Formulates a conflict (ATTACKS A1 and A2) stating that the diagnosis cannot be finalized without rule-out markers (TSH, Prolactin, 17-OH, etc.).
  * **A4 (Clinician Resolution)**: Represents the clinician verification pathway that resolves/defeats the attack.
* **Symbolic Solver**:
  Compute Dung's abstract argumentation framework extensions to determine which arguments are "Accepted" (In), "Defeated" (Out), or "Undecided" (Undecided).
* **Graph Structure Generation**:
  Construct a JSON-serializable Argument Graph containing edges with relation types (`SUPPORTS`, `ATTACKS`, `RESOLVES`) and individual node evaluation statuses.
* Generate a clinical dossier report explaining the argumentation tree, conflict relations, and citation counts.

---

### 4. Interactive Argument Graph Dashboard UI
We will render the clinical argumentation graph visually in the Streamlit UI using Graphviz.

#### [MODIFY] [app.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/app.py)
* Replace the SHAP bar chart under the "Explainable AI Engine" tab with a **Interactive Argumentation Graph**.
* Render node colors dynamically based on their status:
  * **Accepted Arguments**: Green.
  * **Defeated/Attacked Arguments**: Red.
  * **Resolution Nodes**: Blue.
* Render conflict edges (ATTACKS) as dashed red arrows and support/resolution edges as solid grey/blue arrows.
* Display the individual agent transcripts under expanding panels to show how the debate evolved.
* Display a **Citation Grounding Matrix** showing the most frequently cited papers.

---

## Verification Plan

### Automated Tests
- Run `python -m pytest tests/` to verify state integration remains intact.
- Update tests if necessary to assert the presence of `debate_history` in the state.

### Manual Verification
- Launch the Streamlit dashboard and verify:
  1. The Argumentation Graph renders with proper nodes (`A1`, `A2`, `A3`) showing conflicts when biomarkers are missing.
  2. The debate history transcripts are readable.
  3. The RAG metrics say "Unavailable" when local fallback is active.
