import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a) AS labels_a, a.name AS name_a, 
               type(r) AS rel_type, 
               labels(b) AS labels_b, b.name AS name_b
        """
        result = session.run(query)
        print("--- All Database Relationships ---")
        for record in result:
            print(f"({record['labels_a']}:{record['name_a']}) -[:{record['rel_type']}]-> ({record['labels_b']}:{record['name_b']})")
    driver.close()
except Exception as e:
    print(f"Error: {e}")
