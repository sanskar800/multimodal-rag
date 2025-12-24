"""
Multimodal RAG System - Main Application

This script orchestrates the complete workflow:
1. Extract elements from PDF (text, tables, images)
2. Generate summaries using Llama-8B and Gemini (with caching)
3. Store in ChromaDB with embeddings
4. Create RAG pipeline for querying
"""

import os
import sys
import json
import hashlib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extract_elements import extract_elements
from src.generate_summaries import generate_summaries
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline, format_response


def get_pdf_hash(pdf_path: str) -> str:
    """Get hash of PDF file to detect changes."""
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_cached_summaries(cache_dir: str = "data/cache") -> dict:
    """Load cached summaries if they exist."""
    cache_path = Path(cache_dir) / "summaries_cache.json"
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_cached_summaries(summaries: dict, pdf_hash: str, cache_dir: str = "data/cache"):
    """Save summaries to cache."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        "pdf_hash": pdf_hash,
        "summaries": summaries
    }
    
    with open(cache_path / "summaries_cache.json", 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    print("  💾 Cached summaries for future runs\n")


def main():
    """Main application workflow."""
    
    print("\n" + "="*80)
    print("🚀 MULTIMODAL RAG SYSTEM")
    print("="*80 + "\n")
    
    # Configuration
    pdf_path = "docs/attention-is-all-you-need.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return
    
    # Check if we can use cached summaries
    pdf_hash = get_pdf_hash(pdf_path)
    cached_data = load_cached_summaries()
    use_cache = cached_data and cached_data.get("pdf_hash") == pdf_hash
    
    if use_cache:
        print("✨ Found cached summaries! Skipping extraction and summarization.\n")
        summarized_elements = cached_data["summaries"]
    else:
        # Step 1: Extract elements from PDF
        print("STEP 1: EXTRACTING ELEMENTS FROM PDF")
        print("-" * 80)
        elements = extract_elements(pdf_path)
        
        # Step 2: Generate summaries
        print("\nSTEP 2: GENERATING SUMMARIES")
        print("-" * 80)
        summarized_elements = generate_summaries(elements)
        
        # Save to cache
        save_cached_summaries(summarized_elements, pdf_hash)
    
    # Step 3: Initialize vector store and add summaries
    print("STEP 3: STORING IN VECTOR DATABASE")
    print("-" * 80)
    vector_store = VectorStore()
    
    # Check if vector store already has data
    stats = vector_store.get_collection_stats()
    if stats.get("total_documents", 0) > 0 and use_cache:
        print(f"  ℹ️  Vector store already populated with {stats['total_documents']} documents")
        print("  ⏭️  Skipping re-indexing\n")
    else:
        vector_store.add_summaries(summarized_elements)
        stats = vector_store.get_collection_stats()
    
    print(f"📊 Vector Store Stats: {stats}")
    
    # Step 4: Initialize RAG pipeline
    print("\nSTEP 4: INITIALIZING RAG PIPELINE")
    print("-" * 80)
    rag_pipeline = RAGPipeline(vector_store)
    
    # Step 5: Interactive query loop
    print("\nSTEP 5: READY FOR QUERIES")
    print("-" * 80)
    print("\n✨ System ready! You can now ask questions about the document.\n")
    
    # Example queries
    example_queries = [
        "What is the Transformer architecture?",
        "Explain the attention mechanism",
        "What are the key components of the model?",
        "What results are shown in the tables?",
    ]
    
    print("📝 Example queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Type your question, 'clear' to clear cache, or 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if question.lower() == 'clear':
                cache_path = Path("data/cache/summaries_cache.json")
                if cache_path.exists():
                    cache_path.unlink()
                    print("🗑️  Cache cleared! Restart to re-process PDF.\n")
                else:
                    print("ℹ️  No cache to clear.\n")
                continue
            
            # Process query
            response = rag_pipeline.query(question)
            
            # Display formatted response
            print(format_response(response))
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
