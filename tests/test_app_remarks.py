import os
from pipeline.nodes.node1_ingestion import extract_biomedical_entities
from utils.neo4j_client import Neo4jMedicalGraph
from dotenv import load_dotenv

load_dotenv()

app_remarks = (
    "Patient presents with severe, treatment-resistant inflammatory acne along the jawline and neck. "
    "Reports a history of profound oligomenorrhea, experiencing only 3 irregular menstrual periods in the "
    "past 12 months. Transvaginal pelvic ultrasound reveals marked bilateral polycystic ovary morphology "
    "with an antral follicle count of 22 on the left and 26 on the right, presenting a classic "
    "string-of-pearls arrangement."
)

print("--- Running SciSpacy on app.py remarks ---")
entities = extract_biomedical_entities(app_remarks)
print(f"Extracted Entities: {entities}")

print("\n--- Querying Neo4j with these entities ---")
db = Neo4jMedicalGraph()
subgraph = db.get_clinical_subgraph(entities)
print(f"Neo4j Returned Subgraph: {subgraph}")
db.close()
