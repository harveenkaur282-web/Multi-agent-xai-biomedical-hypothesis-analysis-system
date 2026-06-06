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
        # Step 1: Check Condition PCOS
        print("Checking PCOS node:")
        res = session.run("MATCH (d:Condition {name: 'PCOS'}) RETURN d")
        records = list(res)
        print(f"PCOS nodes found: {len(records)}")
        for r in records:
            print(r)
            
        # Step 2: Check Phenotypes in entities
        print("\nChecking Phenotype nodes matching entities:")
        res = session.run("MATCH (s:Phenotype) WHERE s.name IN $entities RETURN s.name as name, labels(s) as labels", entities=entities)
        records = list(res)
        print(f"Phenotype nodes found: {len(records)}")
        for r in records:
            print(f"Name: {r['name']}, Labels: {r['labels']}")
            
        # Step 3: Check paths
        print("\nChecking shortest path query:")
        query = """
        MATCH (d:Condition {name: 'PCOS'})
        MATCH (s:Phenotype) WHERE s.name IN $entities
        MATCH path = shortestPath((d)-[:ASSOCIATED_WITH|DRIVES|PART_OF*1..3]-(s))
        RETURN path, length(path) as len
        """
        res = session.run(query, entities=entities)
        records = list(res)
        print(f"Paths found: {len(records)}")
        for r in records:
            print(f"Path length: {r['len']}")
            
    driver.close()
except Exception as e:
    print(f"Error: {e}")
