"""
Vector Store Management for Mysoft Heaven Chatbot
Handles document processing, chunking, embeddings, and FAISS operations
"""

import os
import pickle
from typing import List, Tuple, Optional
import numpy as np

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# PDF processing
import PyPDF2
import pdfplumber
from pathlib import Path

class MysoftVectorStore:
    """Manages vector store operations for Mysoft Heaven documents"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with embedding model
        """
        self.model_name = model_name
        self.embeddings = self._load_embeddings()
        self.vectorstore = None
        self.text_splitter = self._create_text_splitter()
        
    def _load_embeddings(self):
        """Load HuggingFace embeddings model"""
        print(f"Loading embeddings model: {self.model_name}")
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _create_text_splitter(self):
        """
        Create text splitter with optimal chunking strategy
        Chunk size: 500 chars with 100 overlap for semantic coherence
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for better accuracy
        """
        full_text = ""
        
        # Method 1: PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        full_text += f"\n--- Page {page_num + 1} ---\n"
                        full_text += text
        except Exception as e:
            print(f"PyPDF2 extraction error: {e}")
        
        # Method 2: pdfplumber (better for tables and layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        full_text += f"\n--- Page {page_num + 1} (pdfplumber) ---\n"
                        full_text += text
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")
        
        return full_text
    
    def create_chunks_from_text(self, text: str, source: str = "Mysoft Heaven Profile") -> List[Document]:
        """
        Convert extracted text into document chunks
        """
        # Clean text
        text = self.clean_text(text)
        
        # Create document
        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "company": "Mysoft Heaven (BD) Ltd."
            }
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_total"] = len(chunks)
            
        print(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove multiple newlines
        text = '\n'.join([line for line in text.split('\n') if line.strip()])
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Fix common PDF extraction issues
        replacements = {
            '  ': ' ',
            '•': '- ',
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '—': '-',
            '–': '-',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def create_vectorstore(self, chunks: List[Document], save_path: str = "mysoft_vector_db"):
        """
        Create FAISS vector store from document chunks
        """
        print("Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save vector store
        self.save_vectorstore(save_path)
        print(f"Vector store created and saved to {save_path}")
        
        return self.vectorstore
    
    def save_vectorstore(self, path: str = "mysoft_vector_db"):
        """
        Save FAISS vector store to disk
        """
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
    
    def load_vectorstore(self, path: str = "mysoft_vector_db") -> Optional[FAISS]:
        """
        Load FAISS vector store from disk
        """
        try:
            self.vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from {path}")
            return self.vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 5) -> Tuple[List[Document], List[float]]:
        """
        Perform similarity search with scores
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Load or create vector store first.")
        
        # Get documents with similarity scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        documents = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        
        return documents, scores
    
    def process_pdf_and_create_db(self, pdf_path: str, save_path: str = "mysoft_vector_db"):
        """
        End-to-end processing: PDF -> Chunks -> Vector Store
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from PDF")
        
        # Create chunks
        chunks = self.create_chunks_from_text(text)
        
        # Create vector store
        self.create_vectorstore(chunks, save_path)
        
        # Save chunks as backup
        chunks_path = f"{save_path}_chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"Chunks saved to {chunks_path}")
        
        return self.vectorstore, chunks
    
    def get_relevant_context(self, query: str, k: int = 5, threshold: float = 0.65) -> Tuple[str, float, List[float]]:
        """
        Get relevant context with confidence score
        """
        docs, scores = self.similarity_search(query, k=k)
        
        # Calculate average confidence
        avg_confidence = sum(scores) / len(scores) if scores else 0
        
        # Filter by threshold
        relevant_docs = []
        relevant_scores = []
        
        for doc, score in zip(docs, scores):
            if score >= threshold:
                relevant_docs.append(doc)
                relevant_scores.append(score)
        
        # Format context
        if relevant_docs:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
        else:
            context = ""
            
        return context, avg_confidence, relevant_scores

# Multi-company support class
class MultiCompanyVectorStore:
    """
    Support multiple companies by managing separate FAISS indexes
    """
    
    def __init__(self, base_path: str = "company_vector_stores"):
        self.base_path = base_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstores = {}
        
    def create_company_store(self, company_id: str, documents: List[Document]) -> FAISS:
        """Create vector store for specific company"""
        store_path = f"{self.base_path}/{company_id}"
        os.makedirs(store_path, exist_ok=True)
        
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(store_path)
        
        self.vectorstores[company_id] = vectorstore
        return vectorstore
    
    def load_company_store(self, company_id: str) -> Optional[FAISS]:
        """Load vector store for specific company"""
        store_path = f"{self.base_path}/{company_id}"
        
        try:
            vectorstore = FAISS.load_local(
                store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vectorstores[company_id] = vectorstore
            return vectorstore
        except Exception as e:
            print(f"Error loading {company_id} store: {e}")
            return None
    
    def query_company(self, company_id: str, query: str, k: int = 5) -> Tuple[List[Document], List[float]]:
        """Query specific company's vector store"""
        if company_id not in self.vectorstores:
            self.load_company_store(company_id)
            
        if company_id in self.vectorstores:
            docs_with_scores = self.vectorstores[company_id].similarity_search_with_score(query, k=k)
            docs = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            return docs, scores
        else:
            raise ValueError(f"Company {company_id} not found")