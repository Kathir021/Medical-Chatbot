import csv
import os
import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase

class ManualGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), encrypted=False)
        print("✅ Connected to Neo4j")

    def close(self):
        self.driver.close()

    def build_manual_graph(self, study_id: str):
        with self.driver.session() as session:
            session.execute_write(self._create_manual_graph, study_id)

    @staticmethod
    def _create_manual_graph(tx, study_id: str):
        tx.run("MERGE (s:Study {id: $study_id})", study_id=study_id)

        entries = [
            ("Sponsor", "NIH", "HAS_SPONSOR"),
            ("Condition", "Diabetes", "INVESTIGATES_CONDITION"),
            ("Intervention", "Metformin", "USES_INTERVENTION"),
            ("Facility", "Global Health Center", "CONDUCTED_AT"),
            ("City", "New York", "LOCATED_IN_CITY"),
            ("State", "NY", "PART_OF_STATE"),
            ("Country", "USA", "IN_COUNTRY"),
            ("Phase", "Phase 3", "IN_PHASE"),
            ("PrimaryPurpose", "Treatment", "HAS_PURPOSE"),
            ("Document", "Protocol File", "HAS_DOCUMENT"),
            ("InformedConsentForm", "Consent Form v2", "IS_TYPE")
        ]

        #Query to create or match node
        for label, name, rel in entries:
            tx.run(f"""
                MERGE (n:{label} {{name: $name}})  
                WITH n
                MATCH (s:Study {{id: $study_id}})
                MERGE (s)-[:{rel}]->(n)
            """, name=name, study_id=study_id)

    def import_csv(self, csv_files):
        for path in csv_files:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col, val in row.items():
                        if val:
                            with self.driver.session() as session:
                                session.execute_write(self._merge_csv_node, col.strip(), val.strip(), path)

    @staticmethod
    def _merge_csv_node(tx, label, value, filename):
        label = label.replace(" ", "_").replace("-", "_")
        tx.run(f"""
            MERGE (n:{label} {{name: $value}})
            SET n.source = $filename
        """, value=value, filename=os.path.basename(filename))

    def process_url(self, url):
        study_id = url.split("/")[-1]
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            self.build_manual_graph(study_id)

            rows = soup.select("div.ct-body3 tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) == 2:
                    label = cells[0].text.strip().replace(":", "").replace(" ", "_")
                    value = cells[1].text.strip()
                    if label and value:
                        with self.driver.session() as session:
                            session.execute_write(self._merge_study_detail_node, study_id, label, value, url)
        except Exception as e:
            print(f"❌ Failed for {url}: {e}")

    @staticmethod
    def _merge_study_detail_node(tx, study_id, label, value, url):
        tx.run(f"""
            MERGE (n:{label} {{name: $value}})
            SET n.source = $url
            WITH n
            MATCH (s:Study {{id: $study_id}})
            MERGE (s)-[:HAS_{label}]->(n)
        """, value=value, study_id=study_id, url=url)

    def match_shared_nodes(self):
        with self.driver.session() as session:
            session.execute_write(self._match_similar_nodes)

    @staticmethod
    def _match_similar_nodes(tx):
        tx.run("""
            MATCH (a), (b)
            WHERE a.name = b.name AND elementId(a) <> elementId(b)
            MERGE (a)-[:RELATED_TO]->(b)
        """)
    def enrich_additional_relationships(self):
        with self.driver.session() as session:
            session.execute_write(self._add_extra_relationships)

    @staticmethod
    def _add_extra_relationships(tx):
        tx.run("""
            MATCH (d:Drug), (c:Condition)
            WHERE toLower(d.name) CONTAINS toLower(c.name) OR toLower(c.name) CONTAINS toLower(d.name)
            MERGE (d)-[:TARGETS_CONDITION]->(c)
        """)
        tx.run("""
            MATCH (f:Facility), (city:City)
            WHERE toLower(f.name) CONTAINS toLower(city.name)
            MERGE (f)-[:LOCATED_IN_CITY]->(city)
        """)
        tx.run("""
            MATCH (city:City), (state:State)
            MERGE (city)-[:PART_OF_STATE]->(state)
        """)
        tx.run("""
            MATCH (state:State), (country:Country)
            MERGE (state)-[:IN_COUNTRY]->(country)
        """)
        tx.run("""
            MATCH (doc:Document), (consent:InformedConsentForm)
            MERGE (doc)-[:HAS_VERSION]->(consent)
        """)
        tx.run("""
            MATCH (p:Phase), (purpose:PrimaryPurpose)
            MERGE (p)-[:HAS_PURPOSE]->(purpose)
        """)

# === Run Everything ===
if __name__ == "__main__":
    builder = ManualGraphBuilder("neo4j_uri", "neo4j", "neo4j_password")

    # Load your CSV
    csv_files = ["file_name.csv"]
    builder.import_csv(csv_files)

    # Load URLs from .txt file
    with open("file_name.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    for url in urls:
        builder.process_url(url)

    # Add cross-matches + enrich relationships
    builder.match_shared_nodes()
    builder.enrich_additional_relationships()

    builder.close()
