# PCOS Multi-Agent XAI Diagnostics Framework Upgrade Plan

This plan details the architectural and procedural upgrades for the PCOS Multi-Agent XAI Diagnostics Framework. It incorporates human-in-the-loop validation, an LLM-as-a-judge evaluation node, a proper chunked RAG pipeline using Reciprocal Rank Fusion (RRF) with citation mapping, RAG evaluation metrics, and a high-fidelity clinical reference dataset for the SHAP surrogate model in explainable AI (XAI).

---

## User Review Required

> [!IMPORTANT]
> **Human-in-the-Loop State persistence**: To support pausing and resuming the LangGraph workflow, we will add a local memory saver (`MemorySaver`) checkpointer to the Compiled Graph. In the Streamlit UI, if the pipeline halts at the approval step, the app will render an interface showing the generated hypothesis and letting the user edit, approve, or reject it.
> - **Approve/Edit**: The graph updates the state and resumes execution to Node 3.
> - **Reject**: The graph saves the user feedback and the system stops there itself to preserve computational resources. 

> [!WARNING]
> **LLM API Usage**: Using Qwen-2.5-7B/14B locally or Groq API (`llama3-70b-8192` or `mixtral-8x7b-32768`) will improve reasoning quality and JSON stability compared to the local `llama3.2:3b`. We will add a selection dropdown in the Streamlit sidebar. If Groq is chosen, it will use the `GROQ_API_KEY` in the `.env` file.
---

## Open Questions

None at this stage. You can specify if you have preferred weights for RRF or specific LLM judge criteria.

---

## Proposed Changes

### 1. State & Graph Layout
We will expand the graph state to include variables for human feedback, judge evaluations, chunk metadata, and RAG evaluation scores. We will introduce a conditional branch for human approval, keeping **Node 3 (DualAnalysisNode)**, **Node 4 (AssemblyNode)**, and **Node 5 (ExplainableAINode)** in their correct sequential roles.

#### Key Design Decisions
* **In-memory vector store (NumPy arrays)**: The corpus (~80 abstracts → ~400 chunks) is small enough to embed and index in RAM in < 2 seconds. No SQLite/ChromaDB overhead needed.
* **Single LLM for all CrewAI agents**: One model is used across all 4 agents (Endocrinologist, Metabolic, Differential, Consensus) for consistency, reproducibility, and clean citation in the report. The model is selectable from the Streamlit sidebar.
* **Terminate on rejection (no loop)**: If the clinician rejects the hypothesis, the pipeline terminates and saves the feedback to state. The user can re-run from scratch with adjusted inputs. This conserves computational resources.

#### Graph Execution Flow
1.  **Node 1 (Ingestion)** → **Node 2 (Hypothesis)**.
2.  **JudgeNode [NEW]** runs automatically right after Node 2 to calculate **RAG Evaluation Metrics** (Faithfulness, Relevance) and **LLM Judge Scores**.
3.  **Human-in-the-Loop Breakpoint**: Graph halts execution. Streamlit displays the hypothesis, citations, Judge scores, and RAG evaluation percentages.
4.  **Clinician Action (Streamlit UI)**:
    *   *If Approved*: The graph resumes to **Node 3 (Dual Analysis)** → **Node 4 (Assembly)** → **Node 5 (XAI)** → **END**.
    *   *If Rejected*: The clinician's text feedback is saved to state. The graph routes to **END**. A message is displayed: *"Pipeline terminated. Clinician feedback has been recorded and will inform future iterations."*

#### [MODIFY] [state.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/state.py)
* Add new state keys:
  * `human_feedback`: string with clinician's comments if rejected.
  * `human_approved`: boolean flag indicating whether the clinician approved the hypothesis.
  * `llm_judge_evaluation`: dict containing quantitative scores (0-100%) and textual reasoning from the LLM judge.
  * `rag_eval_metrics`: dict containing Faithfulness, Context Relevance, and Answer Relevance scores.
  * `retrieved_chunks`: updated schema to support chunking metadata, source paper Title/PMID/Abstract, and unique IDs for citations.

#### [MODIFY] [graph.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/graph.py)
* Initialize `MemorySaver` checkpointer.
* Add new node `JudgeNode` executing `node_judge_fn`.
* Configure the entry point and edges:
  * `IngestionNode` → `HypothesisNode` → `JudgeNode` → interrupt breakpoint.
  * Define conditional routing from the breakpoint:
    * If `human_approved` is `True`, route to `DualAnalysisNode` (Node 3).
    * If `human_approved` is `False`, route to `END` (save feedback, terminate gracefully).



---

### 2. Advanced Ingestion & Retrieval (Node 1)
We will transition Node 1 from whole-abstract matching to a proper chunked RAG pipeline with dense & sparse indexing and Reciprocal Rank Fusion (RRF). To ensure evaluation stability, reproducibility, and system reliability, we will support both a curated offline corpus and a live API search.

#### [NEW] [pcos_literature_corpus.json](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/data/pcos_literature_corpus.json)
* Build a curated local database of ~30-40 real clinical research abstracts fetched from PubMed, categorized by domain:
  * Ovarian physiology & Rotterdam Diagnostic Criteria
  * Hyperinsulinemia & insulin resistance pathways (metabolic phenotype)
  * Gonadotropin axis dysregulation & LH/FSH inversion (lean phenotype)
  * Differential diagnosis mimics (TSH/thyroid, Prolactin, 17-OH progesterone for NCAH, Cushing's disease)

#### [MODIFY] [node1_ingestion.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node1_ingestion.py)
* Add a `use_live_search` boolean parameter to the raw input schema.
* If `use_live_search` is `True`, run the dynamic NCBI search; otherwise, load the local `pcos_literature_corpus.json` as the target document corpus.
* Implement a chunking utility to split abstracts into passages (e.g. 500 characters with 100 character overlap).
* Set up an in-memory vector store matching each chunk to its source document (title, abstract, PMID).
* Generate dense embeddings for all chunks using `SentenceTransformer("all-MiniLM-L6-v2")`.
* Calculate sparse BM25 scores on the tokenized chunks.
* Implement **Reciprocal Rank Fusion (RRF)** to combine the dense and sparse retrievers:
  $$RRF(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{dense}}(d)}$$
* Return the top 5 chunks with unique citation labels (e.g., `[Source-1, Chunk-2]`) and store them in the state.


---

### 3. Hypothesis Generation & Citation (Node 2)
We will configure CrewAI agents to use the chunked documents, citing specific chunk IDs to ensure accountability. We will also integrate Groq and Qwen models.

#### [MODIFY] [crew_setup.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/agents/crew_setup.py)
* Support dynamic LLM configuration based on the user's sidebar selection:
  - Default: `ollama/llama3.2:3b`
  - Groq Option: `groq/llama3-70b-8192` (using `GROQ_API_KEY` from the environment or `.env` file).
* Pass literature chunks explicitly as context to the crew.
* Update agent system prompts and tasks to strictly mandate **source citations** from the provided chunks (e.g., citing the specific `[Source-X, Chunk-Y]` ID).
* Ensure that the consensus hypothesis outputs include cited sources correctly.

#### [MODIFY] [node2_hypothesis.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node2_hypothesis.py)
* Feed structured chunk data (including their unique IDs `[Source-X, Chunk-Y]`) to the debate crew.
* Include the clinician's `human_feedback` (if routing back or loaded from previous iteration) in the debate context so agents adapt and fix errors.

#### [MODIFY] [app.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/app.py)
* Implement the selection for LLM backends (Ollama or Groq).
* Manage LangGraph state thread persistence via `configurable` thread IDs to enable interruption and resuming.
* Render the **Human-in-the-Loop decision layout** when graph execution is paused at `JudgeNode`:
  - Show the generated clinical hypothesis and its citations.
  - If approved, update state with `human_approved = True` and resume execution.
  - If rejected, prompt user for text feedback, update state with `human_approved = False` and `human_feedback = feedback`, resume the graph execution (which gracefully routes to `END`), and stop right there to save resources.

---

### 4. LLM as a Judge & RAG Evaluation (New Node)
We will introduce evaluation layers immediately following Node 2 to compute reliability metrics, addressing the professor's demand for quantitative proof.

#### [NEW] [node_judge.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node_judge.py)
* Create `node_judge_fn(state)` that uses an LLM to evaluate both the hypothesis and the retriever.
* **LLM-as-a-Judge Metrics**:
  * **Rotterdam Criteria Accuracy (0-100%)**: Evaluates verification of Oligomenorrhea, Hyperandrogenism, and Polycystic Ovaries.
  * **Clinical Correlation (0-100%)**: Evaluates consistency with the patient's blood/BMI metrics.
  * **Differential Rule-out Logic (0-100%)**: Checks completeness of missing biomarker identification.
* **RAG Evaluation Metrics**:
  * **Faithfulness (0-100%)**: Measures if the generated hypothesis is strictly grounded in the retrieved chunks (hallucination check).
  * **Context Relevance (0-100%)**: Measures how relevant the retrieved chunks are to the user's initial clinical case.
  * **Answer Relevance (0-100%)**: Measures if the hypothesis addresses the clinical query.
* Generates a detailed audit score report.
#### [MODIFY] [node3_dual_analysis.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node3_dual_analysis.py)
* Keeps all current Classical and Quantum score calculations intact.
* Pulls the RAG evaluation and LLM judge metrics from the state to display in interpretations and logs.
* Packages scores into a normalized dictionary specifically structured to feed a comparative analysis table (aligning Bayesian Credibility vs. Quantum Interaction, and Statistical Uncertainty vs. Quantum Von Neumann Entropy).

---

### 5. Trusted Explainable AI (Node 5)
We will replace the uniform synthetic data used for the local surrogate SHAP model with a biologically sound clinical reference generator. We will also compile a structured Classical vs. Quantum Comparison Table inside the final markdown report.

#### Scientific Justification for Covariance Alignment
Standard SHAP explanations treat features as independent, which creates **out-of-distribution (OOD) perturbation errors** (evaluating the model on physically impossible combinations like a BMI of 15 and Fasting Insulin of 40). To resolve this and ensure the explanation is "True to the Data," the background dataset must reflect the biological correlation and covariance structure of the clinical biomarkers.
*   **Methodology Citation**: Chen, H., Covert, I. C., Lundberg, S. M., & Lee, S. I. (2020). *"True to the Model or True to the Data?"* (arXiv:2006.16230). This study proves that conditioning on the data distribution (respecting covariance) prevents OOD explanations and ensures local explanations are faithful to the domain biology.

#### Clinical Parameter Reference & Correlations
The covariance matrix ($\Sigma$) will be calibrated using Pearson correlation coefficients ($\rho$) and variances ($\sigma^2$) published in clinical studies:
1.  **Fasting Insulin and Free Testosterone** ($\rho \approx 0.35$): Hyperinsulinemia directly drives excess ovarian androgen production by downregulating Sex Hormone-Binding Globulin (SHBG).
    *   **Clinical Citation**: Dunaif, A., et al. (1989). *"Insulin resistance and hyperandrogenism in polycystic ovary syndrome."* *Endocrine Reviews*, 10(2), 205-224.
2.  **BMI and Fasting Insulin** ($\rho \approx 0.45$): Adiposity is strongly coupled with insulin resistance in classic metabolic PCOS.
    *   **Clinical Citation**: Kim, S. Y., et al. (2012). *"Relationship between body mass index, insulin resistance, and endocrine parameters in women with polycystic ovary syndrome."* *Korean Journal of Obstetrics & Gynecology*, 55(7), 485-492.
3.  **LH/FSH Ratio and Free Testosterone** ($\rho \approx 0.40$): Gonadotropin axis dysfunction drives androgen excess in lean hyperandrogenic phenotypes.

#### [MODIFY] [node5_xai.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/pipeline/nodes/node5_xai.py)
*   Replace independent uniform random sampling with a **covariance-aligned clinical reference dataset** generated via `numpy.random.multivariate_normal` parameterized by the values above.
*   Train the local Logistic Regression surrogate model on this realistic clinical manifold, ensuring SHAP feature attributions are biologically sound and trustworthy.
*   **Generate Classical vs. Quantum Comparative Table**: Add code to format a clean markdown comparison table comparing the analytical parameters of both domains:
    | Analysis Dimension | Classical Statistical Space | Quantum Entangled Space (Simulated) |
    | :--- | :--- | :--- |
    | **Primary Index Score** | `Bayesian Credibility: {bayes_score}%` | `Quantum Interaction: {quantum_score}` |
    | **Mathematical Focus** | Linear threshold bounds & clinical rules | Multi-axis non-linear coupling / qubit states |
    | **Uncertainty/Entropy** | `Statistical Uncertainty: {uncertainty}%` | `Von Neumann Entropy: {entropy}` |
    | **Clinical Interpretation**| {classical_interpretation} | {quantum_interpretation} |
    This table will be written directly into `xai_report` and output in Streamlit.

---


### 6. User Interface & Human-in-the-Loop Panel
We will upgrade the Streamlit application to display the new evaluation metrics, manage the LLM selection, and provide the interactive approval dashboard.

#### [MODIFY] [app.py](file:///c:/Users/harve/OneDrive/docs/GitHub/Multi-agent-xai-biomedical-hypothesis-analysis-system/app.py)
* Add sidebar options to choose the LLM backend (Ollama Local, Groq API) and input API keys.
* Show the interactive **Human-in-the-Loop Decision Panel** when the graph is interrupted:
  * Display the generated hypothesis, citations, and LLM-as-a-Judge evaluation scores.
  * Provide an approval button to resume execution.
  * Provide a rejection textbox for feedback to route back to Node 2.
* Display RAG evaluation percentages, citations, and the improved SHAP attribution plots.


---


## Verification Plan

### Automated Tests
- Run `pytest` on existing and new tests to ensure the state schema remains valid:
  ```powershell
  python -m pytest tests/test_ingestion.py
  ```
- Create a new integration test `tests/test_pipeline_evals.py` to verify the RAG metrics and LLM judge nodes.

### Manual Verification
- Launch the Streamlit dashboard:
  ```powershell
  streamlit run app.py
  ```
- Input the `PCOS-CLASSIC-001` test patient.
- Validate that:
  * Retrieved chunks list clear chunk IDs and source titles.
  * The LLM judge scores and RAG evaluation metrics (Faithfulness, Relevance) display correctly.
  * The Human-in-the-Loop section pauses the pipeline, allows edits, and properly routes execution based on approval/rejection.
  * SHAP attributions reflect biological realism (e.g. insulin dominance for metabolic patients).
