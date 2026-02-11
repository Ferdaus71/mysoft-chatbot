"""
Streamlit Web Interface for Mysoft Heaven (BD) Ltd. RAG Chatbot
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from vector_store import MysoftVectorStore
from rag_pipeline import MysoftRAGPipeline

# Page configuration
st.set_page_config(
    page_title="Mysoft Heaven AI Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* User message */
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.8rem;
        border-radius: 15px 15px 0 15px;
        margin-left: 20%;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #e9ecef;
        color: black;
        padding: 0.8rem;
        border-radius: 15px 15px 15px 0;
        margin-right: 20%;
    }
    
    /* Confidence indicator */
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Sidebar */
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    
    /* Company header */
    .company-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background-color: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "company_info" not in st.session_state:
        st.session_state.company_info = {}
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

def load_vector_store():
    """Load or create vector store"""
    vector_db_path = "mysoft_vector_db"
    pdf_path = "data/Mysoftheaven-Profile 2026.pdf"
    
    vector_processor = MysoftVectorStore()
    
    # Try to load existing vector store
    vectorstore = vector_processor.load_vectorstore(vector_db_path)
    
    if vectorstore is None:
        st.info("🔄 First-time setup: Processing company profile PDF...")
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            st.error(f"❌ PDF not found at {pdf_path}")
            st.info("📁 Please ensure 'Mysoftheaven-Profile 2026.pdf' is in the 'data' folder")
            return None
        
        # Process PDF and create vector store
        with st.spinner("📚 Reading and indexing company documents..."):
            vectorstore, chunks = vector_processor.process_pdf_and_create_db(pdf_path, vector_db_path)
            st.success(f"✅ Successfully processed {len(chunks)} document chunks!")
    
    return vectorstore

def initialize_chatbot():
    """Initialize the RAG pipeline"""
    if not st.session_state.initialized:
        with st.spinner("🤖 Initializing AI assistant..."):
            # Load vector store
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = load_vector_store()
            
            if st.session_state.vectorstore:
                # Create RAG pipeline
                try:
                    st.session_state.pipeline = MysoftRAGPipeline(
                        vectorstore=st.session_state.vectorstore,
                        model_type="ollama"  # Change to "openai" if using OpenAI
                    )
                    st.session_state.initialized = True
                    st.success("✅ AI assistant is ready!")
                    
                    # Extract company info for sidebar
                    extract_company_info()
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize AI assistant: {e}")
                    st.info("💡 Please install Ollama: https://ollama.ai and run: ollama pull mistral")
            else:
                st.error("❌ Could not load vector store")

def extract_company_info():
    """Extract key company information for display"""
    if st.session_state.pipeline and not st.session_state.company_info:
        try:
            info = {}
            
            # Get company stats
            queries = {
                "years of experience": "15+",
                "government clients": "100+",
                "private clients": "300+",
                "offshore clients": "500+",
                "team members": "120+",
                "citizen served": "1100M+",
                "revenue collected": "90,000M+ Taka"
            }
            
            for key, default in queries.items():
                response = st.session_state.pipeline.process_query(f"How many {key}?")
                if response["relevant"] and response["confidence"] > 0.6:
                    info[key] = response["answer"]
                else:
                    info[key] = default
            
            # Get core services
            services_response = st.session_state.pipeline.process_query("What are the core services?")
            if services_response["relevant"]:
                info["services"] = services_response["answer"]
            
            st.session_state.company_info = info
            
        except Exception as e:
            print(f"Error extracting company info: {e}")

def display_sidebar():
    """Display sidebar with company information"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80?text=Mysoft+Heaven", use_column_width=True)
        
        st.markdown("## 🏢 Mysoft Heaven (BD) Ltd.")
        st.markdown("---")
        
        # Company stats
        if st.session_state.company_info:
            st.markdown("### 📊 Key Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Experience**")
                st.markdown("**Govt Clients**")
                st.markdown("**Private Clients**")
                st.markdown("**Team Members**")
            
            with col2:
                st.markdown(f"{st.session_state.company_info.get('years of experience', '15+')}")
                st.markdown(f"{st.session_state.company_info.get('government clients', '100+')}")
                st.markdown(f"{st.session_state.company_info.get('private clients', '300+')}")
                st.markdown(f"{st.session_state.company_info.get('team members', '120+')}")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_conversation()
                st.session_state.messages = []
                st.rerun()
        
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_qs = [
            "What services do you offer?",
            "How many government clients?",
            "Tell me about Land Development Tax Service",
            "What are your core products?",
            "Where is your office located?",
            "What certifications do you have?"
        ]
        
        for q in sample_qs:
            if st.button(q, use_container_width=True, key=f"sample_{q[:10]}"):
                st.session_state.current_query = q
        
        st.markdown("---")
        
        # Contact info
        st.markdown("### 📞 Contact")
        st.markdown("**Phone:** 0241001094")
        st.markdown("**Email:** info@mysoftheaven.com")
        st.markdown("**Web:** www.mysoftheaven.com")
        st.markdown("**Address:** P.R.Tower, Level 8, Dhaka-1216")
        
        # Version info
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Powered by:** RAG + Mistral")

def display_chat_interface():
    """Display main chat interface"""
    
    # Company header
    st.markdown("""
    <div class="company-header">
        <h1>🏢 Mysoft Heaven (BD) Ltd. - AI Assistant</h1>
        <p style="color: #e0e0e0;">Ask me anything about our company, services, products, and projects</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show confidence for assistant messages
                if message["role"] == "assistant" and "confidence" in message:
                    confidence = message["confidence"]
                    if confidence > 0.7:
                        conf_class = "confidence-high"
                        conf_text = "High confidence"
                    elif confidence > 0.4:
                        conf_class = "confidence-medium"
                        conf_text = "Medium confidence"
                    else:
                        conf_class = "confidence-low"
                        conf_text = "Low confidence"
                    
                    st.markdown(f"<span class='{conf_class}'>📊 {conf_text}</span>", 
                              unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about Mysoft Heaven..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            source_placeholder = st.empty()
            
            with st.spinner("🔍 Searching company documents..."):
                # Process query
                response = st.session_state.pipeline.process_query(prompt)
                
                # Display response
                message_placeholder.markdown(response["answer"])
                
                # Show confidence
                confidence = response["confidence"]
                if confidence > 0.7:
                    st.markdown("✅ *High confidence response*")
                elif confidence > 0.4:
                    st.markdown("⚠️ *Medium confidence - verify with official sources*")
                else:
                    st.markdown("❓ *Low confidence - please contact company directly*")
                
                # Show sources
                if response.get("sources") and response["relevant"]:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(f"```\n{source['content']}\n```")
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "confidence": confidence,
                    "sources": response.get("sources", [])
                })
                
                # Increment query counter
                st.session_state.query_count += 1

def display_document_insights():
    """Display insights from the document"""
    
    tab1, tab2, tab3 = st.tabs(["📊 Key Stats", "🛠️ Services", "📱 Products"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👥 Clients")
            st.metric("Government", "100+")
            st.metric("Private", "300+")
            st.metric("Offshore", "500+")
        
        with col2:
            st.markdown("### 📈 Scale")
            st.metric("Team Size", "120+")
            st.metric("Citizens Served", "1100M+")
            st.metric("Govt Offices", "80K+")
        
        with col3:
            st.markdown("### 💰 Revenue")
            st.metric("Total Collection", "90,000M+ Taka")
            st.metric("Daily Collection", "10-12M Taka")
            st.metric("Transactions", "2800Cr+")
    
    with tab2:
        st.markdown("### 🛠️ Core Services")
        services = [
            "AR & VR", "Machine Learning", "Artificial Intelligence",
            "Blockchain", "BIG Data", "GIS Services",
            "E-Governance Solution", "SaaS Product Development",
            "Customized Software Development", "Mobile app & Game Development",
            "GIS Data digitization", "Business Process Automation",
            "IT Consultancy & Training"
        ]
        
        cols = st.columns(3)
        for i, service in enumerate(services):
            cols[i % 3].markdown(f"✅ {service}")
    
    with tab3:
        st.markdown("### 📱 Key Products")
        products = [
            "Card Management System (CMS)",
            "Document Management System (DMS)",
            "Fraud Analyzer",
            "Customer Complain Management System",
            "Real Time Transaction Monitoring",
            "Remittance Management System",
            "Legal & Recovery Management System",
            "Loan Management System",
            "Asset Management System",
            "Environment Monitoring System",
            "Land Development Tax Service",
            "Case Management System",
            "Unique Business ID System",
            "Land Service Gateway",
            "E-Court System",
            "Uttoradhikar Calculator"
        ]
        
        cols = st.columns(2)
        for i, product in enumerate(products):
            cols[i % 2].markdown(f"🔹 {product}")

def main():
    """Main application"""
    
    # Load CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    display_sidebar()
    
    # Main content area
    if not st.session_state.initialized:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h1>🏢 Mysoft Heaven (BD) Ltd.</h1>
            <h2>AI-Powered Company Assistant</h2>
            <p style='font-size: 1.2rem; margin: 2rem 0;'>
                Ask questions about services, projects, clients, and company information.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Launch AI Assistant", use_container_width=True, type="primary"):
                initialize_chatbot()
                st.rerun()
    else:
        # Display chat interface
        display_chat_interface()
        
        # Document insights section (collapsible)
        with st.expander("📊 Company Overview & Statistics"):
            display_document_insights()

if __name__ == "__main__":
    main()