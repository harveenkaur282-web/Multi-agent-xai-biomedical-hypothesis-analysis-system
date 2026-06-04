import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

class Neo4jMedicalGraph:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                f"Critical Error: Environment variables missing from configuration mapping! "
                f"Extracted: URI={self.uri}, USER={self.user}, PASSWORD={'***' if self.password else None}. "
                f"Please check your project root directory for a valid .env file."
            )
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def get_clinical_subgraph(self, extracted_entities: list) -> list:
        """
        Dynamically queries the database to find multi-hop biological pathways 
        connecting the patient's symptoms back to the target condition.
        """
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
        subgraph_payload = []
        try:
            with self.driver.session() as session:
                result = session.run(query, entities=extracted_entities)
                for record in result:
                    subgraph_payload.append({
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["relationship_type"]
                    })
            return subgraph_payload
        except Exception as e:
            print(f"[Neo4j Database Error] Graph traversal failed: {e}")
            return [] # Fallback to empty list gracefully if DB is offline