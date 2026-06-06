import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to URI: {uri}")
print(f"User: {user}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Check node count
        result = session.run("MATCH (n) RETURN count(n) AS node_count")
        for record in result:
            print(f"Total nodes in database: {record['node_count']}")
            
        # Check some sample nodes
        result = session.run("MATCH (n) RETURN labels(n) AS labels, n.name AS name LIMIT 10")
        print("Sample Nodes:")
        for record in result:
            print(f"Labels: {record['labels']}, Name: {record['name']}")
            
        # Check relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
        for record in result:
            print(f"Total relationships in database: {record['rel_count']}")
            
    driver.close()
except Exception as e:
    print(f"Error querying Neo4j: {e}")
