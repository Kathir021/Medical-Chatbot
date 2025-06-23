from neo4j import GraphDatabase

uri = "neo4j+s://687e1313.databases.neo4j.io"
user = "neo4j"
password = "58dgLm9KwXIgU3LEPD1UrfU9ujCrrHnK5PXxjcV2MTY"

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("✅ Successfully connected to Neo4j Aura!")
    driver.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
