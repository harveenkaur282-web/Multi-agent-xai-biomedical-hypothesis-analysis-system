"""
Standalone script that populates the Neo4j graph database with a validated
PCOS biomedical pathway seed dataset.

Without this script the production Cypher query in neo4j_client.py
(`MATCH (d:Condition {name: 'PCOS'}) MATCH (s:Phenotype) WHERE s.name IN $entities ...`)
always returns an empty subgraph because no nodes or relationships exist.

Node schema used (must match neo4j_client.py query labels):
  (:Condition)  — top-level clinical diagnoses
  (:Phenotype)  — observable patient presentations / symptoms
  (:Biomarker)  — measurable lab / imaging values
  (:Pathway)    — underlying biological mechanisms

Relationship types (must match neo4j_client.py ASSOCIATED_WITH|DRIVES|PART_OF):
  ASSOCIATED_WITH  — condition ↔ phenotype / biomarker (bidirectional evidence)
  DRIVES           — upstream pathway drives downstream phenotype/biomarker
  PART_OF          — component membership (biomarker belongs to pathway)

Clinical sources:
  • Rotterdam Criteria 2003 (PCOS phenotype definitions)
  • Androgen Excess Society Guidelines 2006
  • Franks S. (1995) NEJM — PCOS pathophysiology review
  • Azziz R. et al. (2009) JCE&M — hyperandrogenism and HPO axis
  • Diamanti-Kandarakis E. & Dunaif A. (2012) ER — insulin resistance in PCOS

Usage (run once before launching the Streamlit app):
    python utils/neo4j_seed.py

Environment variables required (same as the main app):
    NEO4J_URI      e.g. bolt://localhost:7687
    NEO4J_USER     e.g. neo4j
    NEO4J_PASSWORD e.g. your_password
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from neo4j import GraphDatabase
except ImportError:
    print("[neo4j_seed] ERROR: neo4j driver not installed. Run: pip install neo4j")
    sys.exit(1)


# Each entry in NODES is (label, name, optional_properties_dict).
# Each entry in RELATIONSHIPS is (src_label, src_name, rel_type, tgt_label, tgt_name).
#
# Phenotype node names must exactly match what SciSpacy NER will extract from
# clinical_remarks and what extract_biomedical_entities() returns (title-cased).

NODES = [
    ("Condition", "PCOS",           {"full_name": "Polycystic Ovary Syndrome"}),
    ("Condition", "Type 2 Diabetes",{"full_name": "Type 2 Diabetes Mellitus"}),
    ("Condition", "Metabolic Syndrome", {"full_name": "Metabolic Syndrome"}),

    ("Phenotype", "Oligomenorrhea",        {"description": "Infrequent or irregular menstruation"}),
    ("Phenotype", "Acne",                  {"description": "Inflammatory sebaceous skin lesion"}),
    ("Phenotype", "Hirsutism",             {"description": "Excess male-pattern hair growth"}),
    ("Phenotype", "Alopecia",              {"description": "Androgen-driven scalp hair thinning"}),
    ("Phenotype", "Anovulation",           {"description": "Absence of ovulation"}),
    ("Phenotype", "Polycystic Ovary Morphology", {"description": "≥12 antral follicles per ovary or ovarian volume >10 mL"}),
    ("Phenotype", "Hyperandrogenism",      {"description": "Elevated androgens: clinical or biochemical"}),
    ("Phenotype", "Insulin Resistance",    {"description": "Impaired cellular glucose uptake in response to insulin"}),
    ("Phenotype", "Obesity",               {"description": "BMI ≥ 30 kg/m²"}),
    ("Phenotype", "Infertility",           {"description": "Failure to conceive after 12 months of unprotected intercourse"}),

    ("Biomarker", "LH/FSH Ratio",          {"unit": "ratio",    "threshold_elevated": 2.0}),
    ("Biomarker", "Fasting Insulin",       {"unit": "uIU/mL",  "threshold_elevated": 14.0}),
    ("Biomarker", "Testosterone",          {"unit": "ng/dL",   "threshold_elevated": 70.0}),
    ("Biomarker", "AMH",                   {"unit": "ng/mL",   "threshold_elevated": 6.0}),
    ("Biomarker", "HOMA-IR",               {"unit": "index",   "threshold_elevated": 2.5}),
    ("Biomarker", "Fasting Glucose",       {"unit": "mg/dL",   "threshold_elevated": 100.0}),
    ("Biomarker", "SHBG",                  {"unit": "nmol/L",  "threshold_low": 30.0}),
    ("Biomarker", "Free Androgen Index",   {"unit": "ratio",   "threshold_elevated": 4.0}),

    ("Pathway", "HPO Axis Dysregulation",  {"description": "Hypothalamic-Pituitary-Ovarian axis dysfunction"}),
    ("Pathway", "Hyperinsulinemia",        {"description": "Chronically elevated circulating insulin"}),
    ("Pathway", "Gonadotropin Imbalance",  {"description": "Elevated LH pulse frequency relative to FSH"}),
    ("Pathway", "Ovarian Steroidogenesis", {"description": "Theca cell androgen overproduction"}),
    ("Pathway", "Adipose Inflammation",    {"description": "Adipokine-driven systemic inflammatory state"}),
]

RELATIONSHIPS = [
    # PCOS ↔ Phenotypes
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Oligomenorrhea"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Acne"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Hirsutism"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Alopecia"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Anovulation"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Polycystic Ovary Morphology"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Hyperandrogenism"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Insulin Resistance"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Phenotype", "Infertility"),

    # PCOS ↔ Biomarkers
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Biomarker", "LH/FSH Ratio"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Biomarker", "Testosterone"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Biomarker", "AMH"),
    ("Condition", "PCOS", "ASSOCIATED_WITH", "Biomarker", "HOMA-IR"),

    # PCOS → downstream conditions
    ("Condition", "PCOS", "DRIVES", "Condition", "Type 2 Diabetes"),
    ("Condition", "PCOS", "DRIVES", "Condition", "Metabolic Syndrome"),

    # Pathways → Phenotypes (mechanism-to-manifestation)
    ("Pathway", "HPO Axis Dysregulation",  "DRIVES", "Phenotype", "Oligomenorrhea"),
    ("Pathway", "HPO Axis Dysregulation",  "DRIVES", "Phenotype", "Anovulation"),
    ("Pathway", "HPO Axis Dysregulation",  "DRIVES", "Phenotype", "Infertility"),
    ("Pathway", "Gonadotropin Imbalance",  "DRIVES", "Phenotype", "Hyperandrogenism"),
    ("Pathway", "Gonadotropin Imbalance",  "DRIVES", "Biomarker", "LH/FSH Ratio"),
    ("Pathway", "Ovarian Steroidogenesis", "DRIVES", "Phenotype", "Hirsutism"),
    ("Pathway", "Ovarian Steroidogenesis", "DRIVES", "Phenotype", "Acne"),
    ("Pathway", "Ovarian Steroidogenesis", "DRIVES", "Biomarker", "Testosterone"),
    ("Pathway", "Hyperinsulinemia",        "DRIVES", "Pathway",   "Ovarian Steroidogenesis"),
    ("Pathway", "Hyperinsulinemia",        "DRIVES", "Phenotype", "Obesity"),
    ("Pathway", "Hyperinsulinemia",        "DRIVES", "Phenotype", "Insulin Resistance"),
    ("Pathway", "Adipose Inflammation",    "DRIVES", "Pathway",   "Hyperinsulinemia"),
    ("Pathway", "Adipose Inflammation",    "DRIVES", "Pathway",   "HPO Axis Dysregulation"),

    # Biomarkers PART_OF pathways
    ("Biomarker", "Fasting Insulin",     "PART_OF", "Pathway", "Hyperinsulinemia"),
    ("Biomarker", "HOMA-IR",             "PART_OF", "Pathway", "Hyperinsulinemia"),
    ("Biomarker", "Fasting Glucose",     "PART_OF", "Pathway", "Hyperinsulinemia"),
    ("Biomarker", "LH/FSH Ratio",        "PART_OF", "Pathway", "Gonadotropin Imbalance"),
    ("Biomarker", "AMH",                 "PART_OF", "Pathway", "Ovarian Steroidogenesis"),
    ("Biomarker", "Testosterone",        "PART_OF", "Pathway", "Ovarian Steroidogenesis"),
    ("Biomarker", "SHBG",                "PART_OF", "Pathway", "Ovarian Steroidogenesis"),
    ("Biomarker", "Free Androgen Index", "PART_OF", "Pathway", "Ovarian Steroidogenesis"),
]

def seed_graph(driver):
    with driver.session() as session:
        # 1. Optionally clear existing PCOS-related data to avoid duplicates
        print("[neo4j_seed] Clearing existing PCOS pathway nodes...")
        session.run("MATCH (n) WHERE n:Condition OR n:Phenotype OR n:Biomarker OR n:Pathway DETACH DELETE n")

        # 2. Create all nodes with MERGE (idempotent)
        print(f"[neo4j_seed] Creating {len(NODES)} nodes...")
        for (label, name, props) in NODES:
            prop_str = ", ".join(f'n.{k} = ${k}' for k in props)
            set_clause = f"SET {prop_str}" if props else ""
            query = f"MERGE (n:{label} {{name: $name}}) {set_clause}"
            session.run(query, name=name, **props)

        print(f"[neo4j_seed] Creating {len(RELATIONSHIPS)} relationships...")
        for (src_label, src_name, rel_type, tgt_label, tgt_name) in RELATIONSHIPS:
            query = (
                f"MATCH (a:{src_label} {{name: $src_name}}) "
                f"MATCH (b:{tgt_label} {{name: $tgt_name}}) "
                f"MERGE (a)-[:{rel_type}]->(b)"
            )
            session.run(query, src_name=src_name, tgt_name=tgt_name)

        result = session.run(
            "MATCH (n) WHERE n:Condition OR n:Phenotype OR n:Biomarker OR n:Pathway "
            "RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label"
        )
        print("\n[neo4j_seed] Verification — node counts after seeding:")
        for record in result:
            print(f"  {record['label']:20s} : {record['count']}")

        # 5. Verify relationship counts
        rel_result = session.run(
            "MATCH ()-[r]->() WHERE type(r) IN ['ASSOCIATED_WITH', 'DRIVES', 'PART_OF'] "
            "RETURN type(r) AS rel_type, count(r) AS count ORDER BY rel_type"
        )
        print("\n[neo4j_seed] Verification — relationship counts after seeding:")
        for record in rel_result:
            print(f"  {record['rel_type']:20s} : {record['count']}")

        print("\n[neo4j_seed] Seed complete. Neo4j graph is ready for the PCOS pipeline.")


def main():
    uri      = os.getenv("NEO4J_URI")
    user     = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        print(
            "[neo4j_seed] ERROR: NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD "
            "must all be set in your .env file or environment."
        )
        sys.exit(1)

    print(f"[neo4j_seed] Connecting to {uri} as '{user}'...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("[neo4j_seed] Connection verified.\n")
        seed_graph(driver)
    except Exception as e:
        print(f"[neo4j_seed] ERROR: {e}")
        sys.exit(1)
    finally:
        try:
            driver.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
