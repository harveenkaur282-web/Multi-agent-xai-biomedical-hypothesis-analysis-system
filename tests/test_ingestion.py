import json
from pipeline.nodes.node1_ingestion import node1_ingestion_fn

def run_isolated_ingestion_test():
    print("=" * 60)
    print("STARTING ISOLATED NODE 1 INGESTION UNIT TEST")
    print("=" * 60)

    # 1. Mock the exact Streamlit user input state payload
    mock_state = {
        "raw_input": {
            "patient_id": "TEST_LEAN_PCOS_001",
            "age": 22,
            "bmi": 18.5,
            "family_history": 0,
            "lh_fsh_ratio": 3.41,
            "fasting_insulin": 4.28,
            "amh_levels": 6.8,
            "free_testosterone": 51.0,
            "clinical_remarks": (
                "22-year-old female presenting with severe, treatment-resistant inflammatory "
                "acne clustered along the jawline and chest, alongside clinical hirsutism. "
                "Patient reports persistent oligomenorrhea, experiencing only 2 irregular "
                "menstrual cycles over the trailing 12 months. Transvaginal pelvic ultrasound "
                "demonstrates prominent polycystic ovary morphology bilateral, with an antral "
                "follicle count exceeding 24 per ovary, displaying the classic string-of-pearls sign. "
                "Metabolic profiling indicates exceptional glucose tolerance; fasting insulin, HbA1c, "
                "and lipid vectors are completely stable within optimal ranges. No central adiposity "
                "or signs of insulin resistance detected."
            )
        },
        "retrieved_chunks": [] # Initialize empty state field
    }

    try:
        # 2. Fire the isolated ingestion node directly
        updated_state = node1_ingestion_fn(mock_state)
        
        print("\n" + "=" * 60)
        print(" NODE 1 EXECUTION COMPLETE - ANALYZING PAYLOAD SUCCESS")
        print("=" * 60)
        
        chunks = updated_state.get("retrieved_chunks", [])
        
        # 3. Audit results instantly in the terminal
        print(f"\n Total Chunks Deposited into State Memory: {len(chunks)}")
    
        papers_found = [c for c in chunks if isinstance(c, dict) and c.get("is_paper")]
        graph_edges = [c for c in chunks if not (isinstance(c, dict) and c.get("is_paper"))]
        chunks_deposited = updated_state.get("retrieved_chunks", [])
        graph_data = updated_state.get("graph_knowledge", [])
        print(f"|--  Graph Structure Edges: {len(graph_edges)}")
        print(f"|--  PubMed Paper Abstracts Found: {len(papers_found)}\n")
        
        print("--- Detailed Abstract Manifest ---")
        if not papers_found:
            print("WARNING: 'retrieved_chunks' contains ZERO literature abstracts.")
        else:
            for idx, paper in enumerate(papers_found):
                print(f"\n[{idx + 1}] TITLE: {paper.get('title')}")
                print(f"  HYBRID SCORE: {paper.get('hybrid_score')}")
                print(f"  SNIPPET: {paper.get('text')[:120]}...")

    except Exception as e:
        print(f"\n CRITICAL RUNTIME ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_isolated_ingestion_test()