import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

entities = ['oligomenorrhea', 'acne', 'Polycystic Ovary Morphology', 'Oligomenorrhea', 'polycystic ovary morphology', 'Acne']

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        query = """
        MATCH (d:Condition {name: 'PCOS'})
        MATCH (s:Phenotype) WHERE s.name IN $entities
        MATCH path = shortestPath((d)-[:ASSOCIATED_WITH|DRIVES|PART_OF*1..3]-(s))
        UNWIND nodes(path) AS node
        UNWIND relationships(path) AS rel
        RETURN DISTINCT 
            startNode(rel).name AS source, 
            endNode(rel).name AS target, 
            type(rel) AS relationship_type
        """
        res = session.run(query, entities=entities)
        records = list(res)
        print(f"Records found: {len(records)}")
        for r in records:
            print(f"Source: {r['source']}, Target: {r['target']}, Type: {r['relationship_type']}")
            
    driver.close()
except Exception as e:
    print(f"Error: {e}")
