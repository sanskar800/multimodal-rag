"""
RAG Pipeline Module
Implements the complete RAG pipeline with Llama for generation and output parsing.
"""

import os
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.vector_store import VectorStore

# Load environment variables
load_dotenv()


class RAGResponse(BaseModel):
    """Structured output format for RAG responses."""
    answer: str = Field(description="The main answer to the user's question")
    sources: List[Dict] = Field(description="List of source elements used to generate the answer")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""
    
    def __init__(self, vector_store: VectorStore, model: str = "llama-3.1-8b-instant"):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: VectorStore instance
            model: Groq model to use for generation
        """
        self.vector_store = vector_store
        self.model = model
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        print(f"✅ RAG Pipeline initialized with model: {model}")
    
    def query(self, question: str, k: int = 5) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Structured RAG response
        """
        print(f"\n🔍 Processing query: '{question}'")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.query(question, k=k)
        
        if not retrieved_docs:
            return RAGResponse(
                answer="I couldn't find relevant information to answer this question.",
                sources=[],
                confidence="low"
            )
        
        print(f"  📚 Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 2: Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for idx, doc in enumerate(retrieved_docs):
            element_type = doc.metadata.get('element_type', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            
            context_parts.append(
                f"[Source {idx+1}] (Type: {element_type}, Page: {page})\n{doc.page_content}\n"
            )
            
            sources.append({
                "element_id": doc.metadata.get('element_id', f'doc_{idx}'),
                "element_type": element_type,
                "page": page,
                "content_preview": doc.page_content[:150]
            })
        
        context = "\n".join(context_parts)
        
        # Step 3: Generate response using Llama
        prompt = f"""You are a helpful AI assistant answering questions based on provided context from a document.

Context:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the provided context
2. Be specific and cite which sources you're using (e.g., "According to Source 1...")
3. If the context doesn't contain enough information, say so
4. Keep your answer clear and concise

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Determine confidence based on relevance
            confidence = self._assess_confidence(answer, retrieved_docs)
            
            print(f"  ✅ Generated answer (Confidence: {confidence})\n")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence
            )
        
        except Exception as e:
            print(f"  ❌ Error generating response: {e}\n")
            return RAGResponse(
                answer=f"Error generating response: {str(e)}",
                sources=sources,
                confidence="low"
            )
    
    def _assess_confidence(self, answer: str, docs: List) -> str:
        """Assess confidence level based on answer and retrieved documents."""
        # Simple heuristic: check if answer references sources and isn't too short
        if "source" in answer.lower() and len(answer) > 100:
            return "high"
        elif len(answer) > 50:
            return "medium"
        else:
            return "low"
    
    def batch_query(self, questions: List[str], k: int = 5) -> List[RAGResponse]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions
            k: Number of documents to retrieve per query
            
        Returns:
            List of RAG responses
        """
        responses = []
        for question in questions:
            response = self.query(question, k=k)
            responses.append(response)
        return responses


def format_response(response: RAGResponse) -> str:
    """
    Format RAG response for display.
    
    Args:
        response: RAGResponse object
        
    Returns:
        Formatted string
    """
    output = f"\n{'='*80}\n"
    output += f"ANSWER:\n{response.answer}\n\n"
    output += f"CONFIDENCE: {response.confidence.upper()}\n\n"
    output += f"SOURCES ({len(response.sources)}):\n"
    
    for idx, source in enumerate(response.sources, 1):
        output += f"\n  [{idx}] {source['element_type'].upper()} (Page {source['page']})\n"
        output += f"      ID: {source['element_id']}\n"
        output += f"      Preview: {source['content_preview']}...\n"
    
    output += f"{'='*80}\n"
    return output


if __name__ == "__main__":
    # Test RAG pipeline
    vs = VectorStore()
    rag = RAGPipeline(vs)
    
    test_query = "What is the Transformer architecture?"
    response = rag.query(test_query)
    print(format_response(response))
