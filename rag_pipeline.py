"""
RAG Pipeline for Mysoft Heaven Chatbot
Handles LLM integration, prompt engineering, and response generation
"""

from typing import Dict, List, Tuple, Optional
import re

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: OpenAI fallback
# from langchain_openai import ChatOpenAI

class MysoftRAGPipeline:
    """
    RAG Pipeline for Mysoft Heaven chatbot with strict document-only responses
    """
    
    def __init__(self, vectorstore, model_type: str = "ollama"):
        """
        Initialize RAG pipeline with vector store and LLM
        
        Args:
            vectorstore: FAISS vector store
            model_type: "ollama" (free) or "openai" (paid)
        """
        self.vectorstore = vectorstore
        self.model_type = model_type
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        self.qa_chain = self._create_qa_chain()
        
        # Company-specific context
        self.company_name = "Mysoft Heaven (BD) Ltd."
        
    def _initialize_llm(self):
        """Initialize LLM based on available option"""
        
        if self.model_type == "ollama":
            try:
                # Try Ollama first (free, local)
                llm = Ollama(
                    model="mistral",  # Can also use "llama2", "phi", etc.
                    temperature=0.1,  # Low temperature for factual responses
                    num_ctx=2048,
                    top_p=0.9,
                    stop=["<|im_end|>", "Human:", "Assistant:"]
                )
                print("✅ Using Ollama with Mistral model")
                return llm
            except Exception as e:
                print(f"⚠️ Ollama not available: {e}")
                print("❌ Please install Ollama from https://ollama.ai")
                print("   Then run: ollama pull mistral")
                raise
        
        elif self.model_type == "openai":
            # Uncomment if you have OpenAI API key
            # from langchain_openai import ChatOpenAI
            # llm = ChatOpenAI(
            #     model="gpt-3.5-turbo",
            #     temperature=0.1,
            #     api_key=os.getenv("OPENAI_API_KEY")
            # )
            # return llm
            raise ValueError("OpenAI support requires API key - using Ollama recommended")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _initialize_memory(self):
        """Initialize conversation memory with window buffer"""
        return ConversationBufferWindowMemory(
            k=3,  # Remember last 3 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create strict prompt template to ensure document-only responses
        """
        template = """
        You are an AI assistant for {company_name}. 
        
        IMPORTANT RULES:
        1. ONLY answer questions using the information in the CONTEXT section below
        2. If the answer is not in the context, say: "This information is not available in the company profile."
        3. Do NOT use any external knowledge or make up information
        4. Do NOT speculate or provide opinions
        5. Be precise - quote statistics, dates, and facts exactly as they appear
        6. For greetings, respond professionally but stay within company context
        7. For company contact details, refer to the exact information in the context
        
        CONTEXT:
        {context}
        
        Chat History:
        {chat_history}
        
        Human: {question}
        
        Assistant: Let me answer based on the company profile information provided above.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question", "company_name"]
        )
    
    def _create_qa_chain(self):
        """Create conversational retrieval chain"""
        
        prompt = self._create_prompt_template()
        
        # Create retriever with score threshold
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context"
            },
            verbose=False,
            return_source_documents=True,
            rephrase_question=False  # Important: use original question
        )
        
        return chain
    
    def calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence score based on similarity scores"""
        if not scores:
            return 0.0
        
        # Convert FAISS scores (L2 distance) to similarity (0-1)
        # Lower L2 distance = higher similarity
        similarities = [1 / (1 + score) for score in scores]
        avg_similarity = sum(similarities) / len(similarities)
        
        return avg_similarity
    
    def is_query_relevant(self, query: str, threshold: float = 0.65) -> Tuple[bool, List[Document], List[float]]:
        """
        Check if query is relevant to Mysoft Heaven documents
        
        Returns:
            (is_relevant, relevant_docs, confidence_scores)
        """
        # Get documents with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=3)
        
        if not docs_with_scores:
            return False, [], []
        
        docs = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Calculate average confidence
        confidence = self.calculate_confidence(scores)
        
        return confidence >= threshold, docs, scores
    
    def process_query(self, query: str) -> Dict:
        """
        Process user query and generate response
        
        Args:
            query: User's question
            
        Returns:
            Dict with response, confidence, sources, etc.
        """
        # Clean query
        query = query.strip()
        
        # Check for greetings (always allow)
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in query.lower() for greeting in greetings):
            return {
                "answer": f"Hello! I'm the {self.company_name} assistant. How can I help you with information about our company, services, products, or projects?",
                "confidence": 1.0,
                "sources": [],
                "relevant": True
            }
        
        # Check relevance
        is_relevant, docs, scores = self.is_query_relevant(query)
        
        if not is_relevant:
            return {
                "answer": "This information is not available in the company profile. Please ask about Mysoft Heaven's services, products, clients, projects, or company details.",
                "confidence": 0.0,
                "sources": [],
                "relevant": False
            }
        
        try:
            # Get response from QA chain
            response = self.qa_chain.invoke({
                "question": query,
                "company_name": self.company_name
            })
            
            # Calculate confidence
            confidence = self.calculate_confidence(scores)
            
            # Extract sources
            sources = []
            if "source_documents" in response:
                for i, doc in enumerate(response["source_documents"][:3]):
                    sources.append({
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    })
            
            # Add disclaimer for low confidence
            answer = response["answer"]
            if confidence < 0.7:
                answer = "Based on available information: " + answer
            
            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "relevant": True,
                "full_response": response
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": "I encountered an error processing your question. Please try again or contact support.",
                "confidence": 0.0,
                "sources": [],
                "relevant": False,
                "error": str(e)
            }
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory.clear()
        print("Conversation memory cleared")

# Multi-company support class
class MultiCompanyRAGPipeline:
    """
    RAG pipeline supporting multiple companies
    """
    
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.active_pipelines = {}
        
    def get_pipeline_for_company(self, company_id: str) -> MysoftRAGPipeline:
        """Get or create RAG pipeline for specific company"""
        if company_id not in self.active_pipelines:
            vectorstore = self.vector_store_manager.load_company_store(company_id)
            if vectorstore:
                pipeline = MysoftRAGPipeline(vectorstore)
                pipeline.company_name = company_id.replace("_", " ").title()
                self.active_pipelines[company_id] = pipeline
            else:
                raise ValueError(f"Cannot create pipeline for {company_id}")
        
        return self.active_pipelines[company_id]
    
    def query_company(self, company_id: str, query: str) -> Dict:
        """Query specific company"""
        pipeline = self.get_pipeline_for_company(company_id)
        return pipeline.process_query(query)