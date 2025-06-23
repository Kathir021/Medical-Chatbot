import streamlit as st
from neo4j import GraphDatabase
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re
from typing import List, Dict, Any, Tuple

# === Configuration ===
CONFIG = {
    "model": "txgemma-9b-chat",
    "base_url": "http://127.0.0.1:1234/v1",
    "temperature": 0.5,
    "max_tokens": 2000
}

# Neo4j Aura Configuration
# Replace these with your actual Aura instance details
NEO4J_URI = "neo4j+s://your-instance-id.databases.neo4j.io"  # Your Aura URI
NEO4J_USER = "neo4j"  # Usually 'neo4j'
NEO4J_PASSWORD = "your-aura-password"  # Your Aura instance password

# Alternative: Use environment variables for security
# import os
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your-instance-id.databases.neo4j.io")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your-password")

# === Enhanced Neo4j Graph Access ===
class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            # For Aura, we don't need to specify encrypted=False as it uses TLS by default
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Successfully connected to Neo4j Aura")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def extract_multiple_entities(self, question: str) -> List[str]:
        """Extract multiple potential entities from the question using various strategies"""
        entities = []
        
        # Strategy 1: Direct medical/drug terms (common patterns)
        medical_patterns = [
            r'\b([A-Z][a-z]+(?:mab|nib|zumab|tinib|pril|olol|ide|ine|cin|mycin))\b',  # Drug suffixes
            r'\b([A-Z][a-z]*(?:diabetes|cancer|hypertension|infection|syndrome|disease))\b',  # Conditions
            r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b'  # General capitalized terms
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            entities.extend(matches)
        
        # Strategy 2: Use LLM for entity extraction
        llm_entities = self.detect_entities_with_llm(question)
        entities.extend(llm_entities)
        
        # Remove duplicates and filter
        unique_entities = list(set([e.strip() for e in entities if len(e.strip()) > 2]))
        return unique_entities[:5]  # Limit to top 5 entities

    def detect_entities_with_llm(self, question: str) -> List[str]:
        """Use LLM to detect entities with better prompt"""
        messages = [
            {"role": "system", "content": """Extract all medical entities (drugs, conditions, procedures, organizations, locations) from the user's question. 
            Return them as a comma-separated list. If no entities found, return 'none'.
            Examples:
            - "What is metformin used for?" → "metformin"
            - "Tell me about diabetes studies in New York" → "diabetes, New York"
            - "NIH sponsored trials for cancer" → "NIH, cancer"
            """},
            {"role": "user", "content": question}
        ]
        
        try:
            res = requests.post(
                f"{CONFIG['base_url']}/chat/completions",
                json={
                    "model": CONFIG["model"],
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 500
                }
            )
            response = res.json()["choices"][0]["message"]["content"].strip()
            if response.lower() == 'none':
                return []
            return [entity.strip() for entity in response.split(',') if entity.strip()]
        except Exception as e:
            print(f"LLM entity detection failed: {e}")
            return []

    def search_graph_comprehensive(self, entities: List[str]) -> Tuple[str, List[str]]:
        """Search graph comprehensively for entities and related information"""
        if not entities:
            return self.get_general_graph_overview(), []
        
        found_entities = []
        all_context = []
        
        with self.driver.session() as session:
            for entity in entities:
                # Search with case-insensitive partial matching
                result = session.run("""
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower($entity)
                       OR toLower($entity) CONTAINS toLower(n.name)
                    WITH n, labels(n)[0] AS node_label
                    OPTIONAL MATCH (n)-[r]->(related)
                    WITH n, node_label, 
                         collect(DISTINCT {
                             rel_type: type(r), 
                             target_label: labels(related)[0], 
                             target_name: related.name,
                             target_props: properties(related)
                         }) AS relationships
                    OPTIONAL MATCH (connected)-[r2]->(n)
                    WITH n, node_label, relationships,
                         collect(DISTINCT {
                             rel_type: type(r2),
                             source_label: labels(connected)[0],
                             source_name: connected.name
                         }) AS incoming_rels
                    RETURN n.name AS name, 
                           properties(n) AS props,
                           node_label,
                           relationships,
                           incoming_rels
                    LIMIT 5
                """, entity=entity)
                
                records = list(result)
                if records:
                    found_entities.append(entity)
                    entity_context = self.format_entity_context(entity, records)
                    all_context.append(entity_context)
        
        if not all_context:
            # Fallback: search for any keyword matches
            return self.search_by_keywords(entities), found_entities
        
        return "\n\n".join(all_context), found_entities

    def format_entity_context(self, entity: str, records: List) -> str:
        """Format the entity context in a readable way for natural conversation"""
        context_parts = [f"=== DATABASE INFORMATION FOR: {entity.upper()} ==="]
        
        for record in records:
            name = record["name"]
            props = record["props"]
            node_label = record["node_label"]
            relationships = record["relationships"]
            incoming_rels = record["incoming_rels"]
            
            # Basic entity info
            context_parts.append(f"\n🎯 {node_label}: {name}")
            
            # Add properties if available
            if props:
                prop_text = []
                for key, value in props.items():
                    if key != 'name' and value:
                        prop_text.append(f"{key}: {value}")
                if prop_text:
                    context_parts.append(f"Properties: {', '.join(prop_text)}")
            
            # Add outgoing relationships in natural language
            if relationships and any(rel['target_name'] for rel in relationships):
                context_parts.append("\nConnected to:")
                for rel in relationships[:10]:  # Limit to prevent overflow
                    if rel['target_name']:
                        connection_desc = self.describe_connection(rel['rel_type'], rel['target_label'], rel['target_name'], direction="to")
                        context_parts.append(f"• {connection_desc}")
            
            # Add incoming relationships in natural language
            if incoming_rels and any(rel['source_name'] for rel in incoming_rels):
                context_parts.append("\nAlso connected from:")
                for rel in incoming_rels[:8]:  # Limit incoming
                    if rel['source_name']:
                        connection_desc = self.describe_connection(rel['rel_type'], rel['source_label'], rel['source_name'], direction="from")
                        context_parts.append(f"• {connection_desc}")
        
        return "\n".join(context_parts)

    def describe_connection(self, rel_type: str, entity_label: str, entity_name: str, direction: str) -> str:
        """Convert relationship types into natural language descriptions"""
        
        # Mapping relationship types to natural language
        if direction == "to":
            descriptions = {
                "RELATED_TO": f"associated with {entity_label.lower()} '{entity_name}'",
                "HAS_SPONSOR": f"sponsored by {entity_name}",
                "INVESTIGATES_CONDITION": f"studies the condition {entity_name}",
                "USES_INTERVENTION": f"uses the intervention {entity_name}",
                "CONDUCTED_AT": f"conducted at {entity_name}",
                "LOCATED_IN_CITY": f"located in {entity_name}",
                "PART_OF_STATE": f"in the state of {entity_name}",
                "IN_COUNTRY": f"in {entity_name}",
                "IN_PHASE": f"in {entity_name}",
                "HAS_PURPOSE": f"has the purpose of {entity_name}",
                "HAS_DOCUMENT": f"has documentation: {entity_name}",
                "IS_TYPE": f"is a type of {entity_name}",
                "TARGETS_CONDITION": f"targets the condition {entity_name}",
                "HAS_INDICATION": f"indicated for {entity_name}",
                "HAS_SIDE_EFFECT": f"may cause {entity_name}",
                "MANUFACTURED_BY": f"manufactured by {entity_name}",
                "APPROVED_BY": f"approved by {entity_name}",
                "HAS_VARIANT": f"has variant {entity_name}",
                "BELONGS_TO_CLASS": f"belongs to the class {entity_name}"
            }
        else:  # direction == "from"
            descriptions = {
                "RELATED_TO": f"{entity_name} ({entity_label.lower()}) is associated with it",
                "HAS_SPONSOR": f"{entity_name} sponsors it",
                "INVESTIGATES_CONDITION": f"{entity_name} studies it",
                "USES_INTERVENTION": f"{entity_name} uses it as an intervention",
                "CONDUCTED_AT": f"{entity_name} conducts it",
                "LOCATED_IN_CITY": f"{entity_name} contains it",
                "PART_OF_STATE": f"{entity_name} contains it",
                "IN_COUNTRY": f"{entity_name} contains it",
                "TARGETS_CONDITION": f"{entity_name} targets it",
                "HAS_INDICATION": f"{entity_name} is indicated for it",
                "MANUFACTURED_BY": f"{entity_name} manufactures it",
                "APPROVED_BY": f"{entity_name} approved it"
            }
        
        return descriptions.get(rel_type, f"{entity_name} ({entity_label.lower()})")


    def search_by_keywords(self, keywords: List[str]) -> str:
        """Fallback search using keywords across all node properties"""
        with self.driver.session() as session:
            # Search across all text properties
            keyword_query = " OR ".join([f"toLower(toString(n.name)) CONTAINS toLower('{kw}')" for kw in keywords])
            
            result = session.run(f"""
                MATCH (n)
                WHERE {keyword_query}
                WITH n, labels(n)[0] AS label
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN label, n.name AS name, 
                       collect(DISTINCT type(r) + ': ' + m.name)[0..5] AS connections
                LIMIT 10
            """)
            
            records = list(result)
            if not records:
                return "No relevant information found in the graph database."
            
            context_parts = ["=== KEYWORD SEARCH RESULTS ==="]
            for record in records:
                context_parts.append(f"\n🏷️ {record['label']}: {record['name']}")
                if record['connections']:
                    context_parts.append(f"   Connections: {', '.join(filter(None, record['connections']))}")
            
            return "\n".join(context_parts)

    def get_general_graph_overview(self) -> str:
        """Get a general overview of the graph when no specific entities are found"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WITH labels(n)[0] AS label, count(n) AS count
                RETURN label, count
                ORDER BY count DESC
                LIMIT 10
            """)
            
            overview = ["=== GRAPH DATABASE OVERVIEW ==="]
            overview.append("Available data types:")
            
            for record in result:
                overview.append(f"• {record['label']}: {record['count']} entries")
            
            overview.append("\nAsk me about specific drugs, conditions, studies, or facilities!")
            return "\n".join(overview)

def generate_response(query: str, chat_history: List, context: str = None):
    """Generate response with improved context handling"""
    template = """
You are a highly knowledgeable medical expert speaking to a colleague. You have access to a comprehensive clinical trials and pharmaceutical database.

GRAPH DATABASE CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION: {question}

INSTRUCTIONS FOR RESPONSE STYLE:
- Speak as an expert explaining to another professional - authoritative but conversational
- Use the database context to provide detailed, flowing explanations
- Describe connections naturally without mentioning technical relationship names
- Use phrases like "The drug is connected to...", "This entity is linked to...", "According to our database..."
- Focus on the meaning and significance of the connections, not the technical details
- Provide comprehensive information in a natural, expert consultation style
- When describing connections, explain what they mean in medical/research context

EXAMPLE STYLE:
"The drug metformin is connected to several important entities in our database. It's associated with Type 2 diabetes as its primary therapeutic target, and our records show it's commonly used in diabetes treatment protocols. The database also indicates connections to various clinical studies conducted at major medical centers, and it's linked to other antidiabetic medications through shared therapeutic classifications."

ANSWER:
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(
        base_url=CONFIG["base_url"],
        api_key="not-needed",
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens"]
    )
    chain = prompt | llm | StrOutputParser()
    
    # Format chat history
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in chat_history[-4:]  # Last 4 messages for context
    ])
    
    try:
        return chain.stream({
            "context": context or "No relevant information found in the database.",
            "chat_history": history_text,
            "question": query
        })
    except Exception as e:
        return iter([f"Error generating response: {str(e)}"])

# === Enhanced Streamlit UI ===
def main():
    st.set_page_config("Chatbot", page_icon="💊", layout="wide")
    st.title("Medical Chatbot")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your medical research assistant. I can help you find information about drugs, clinical studies, conditions, and more from our graph database. What would you like to know?")
        ]
    if "neo4j" not in st.session_state:
        try:
            st.session_state.neo4j = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            st.success("✅ Connected to Neo4j database")
        except Exception as e:
            st.error(f"❌ Failed to connect to Neo4j: {e}")
            return

    # Sidebar with database info
    with st.sidebar:
        st.markdown("Imercfy Labs")
        
        if st.button("🔄 Refresh Connection"):
            try:
                st.session_state.neo4j.close()
                st.session_state.neo4j = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                st.success("✅ Reconnected!")
            except Exception as e:
                st.error(f"❌ Reconnection failed: {e}")

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
            st.markdown(message.content)

    # User input
    if user_query := st.chat_input("Ask me about drugs, studies, conditions, or anything medical..."):
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching graph database..."):
                # Extract entities and search graph
                entities = st.session_state.neo4j.extract_multiple_entities(user_query)
                context, found_entities = st.session_state.neo4j.search_graph_comprehensive(entities)
            
            # Show what was found
            col1, col2 = st.columns([1, 1])
            with col1:
                if found_entities:
                    st.success(f"🎯 Found entities: {', '.join(found_entities)}")
                else:
                    st.info("🔍 No specific entities found - showing general results")
            
            with col2:
                if entities:
                    st.info(f"🔎 Searched for: {', '.join(entities)}")
            
            # Show context in expandable section
            if context and context != "No relevant information found in the graph database.":
                with st.expander("📊 Graph Database Context (Click to expand)", expanded=False):
                    st.code(context, language="text")
            
            # Generate and stream response
            response_container = st.empty()
            full_response = ""
            
            response_stream = generate_response(user_query, st.session_state.chat_history, context)
            
            for chunk in response_stream:
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            # Add to chat history
            st.session_state.chat_history.append(AIMessage(content=full_response))
            
            # Show data source
            if found_entities:
                st.caption("📊 Answer from Neo4j graph database")
            else:
                st.caption("🧠 Answer from TxGemma Model")
            
            # Add separator line after each response
            st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown("Medical Chatbot")

if __name__ == "__main__":
    main()