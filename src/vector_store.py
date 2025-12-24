"""
Vector Store Module
Manages ChromaDB for storing and retrieving multimodal summaries.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "data/chroma_db", collection_name: str = "multimodal_rag"):
        """
        Initialize vector store.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings using HuggingFace (free, no API key needed)
        # Using a smaller, efficient model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast and efficient
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"✅ Vector store initialized: {collection_name}")
    
    def add_summaries(self, summarized_elements: Dict[str, List[Dict]]) -> None:
        """
        Add summarized elements to the vector store.
        
        Args:
            summarized_elements: Dictionary with 'text', 'tables', and 'images' summaries
        """
        print("\n💾 Adding summaries to vector store...")
        
        documents = []
        
        # Process text summaries
        for elem in summarized_elements['text']:
            doc = Document(
                page_content=elem['summary'],
                metadata={
                    "element_id": elem['element_id'],
                    "element_type": "text",
                    "page": elem.get('page'),
                    "original_text": elem['original_text'][:500]  # Truncate long text
                }
            )
            documents.append(doc)
        
        # Process table summaries
        for elem in summarized_elements['tables']:
            doc = Document(
                page_content=elem['summary'],
                metadata={
                    "element_id": elem['element_id'],
                    "element_type": "table",
                    "page": elem.get('page'),
                    "original_text": elem['original_text'][:500]
                }
            )
            documents.append(doc)
        
        # Process image summaries
        for elem in summarized_elements['images']:
            doc = Document(
                page_content=elem['summary'],
                metadata={
                    "element_id": elem['element_id'],
                    "element_type": "image",
                    "page": elem.get('page'),
                    "image_path": elem['image_path']
                }
            )
            documents.append(doc)
        
        # Add to vector store
        if documents:
            self.vectorstore.add_documents(documents)
            print(f"  ✅ Added {len(documents)} documents to vector store\n")
        else:
            print("  ⚠️ No documents to add\n")
    
    def query(self, query_text: str, k: int = 5, filter_type: Optional[str] = None) -> List[Document]:
        """
        Query the vector store.
        
        Args:
            query_text: Query string
            k: Number of results to return
            filter_type: Optional filter by element type ('text', 'table', 'image')
            
        Returns:
            List of relevant documents
        """
        search_kwargs = {"k": k}
        
        if filter_type:
            search_kwargs["filter"] = {"element_type": filter_type}
        
        results = self.vectorstore.similarity_search(query_text, **search_kwargs)
        return results
    
    def as_retriever(self, k: int = 5):
        """
        Get retriever interface for RAG pipeline.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.vectorstore._collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Test vector store
    vs = VectorStore()
    
    # Test data
    test_summaries = {
        "text": [{
            "summary": "The Transformer uses self-attention mechanisms.",
            "original_text": "The Transformer architecture...",
            "element_id": "test_1",
            "page": 1,
            "element_type": "text"
        }],
        "tables": [],
        "images": []
    }
    
    vs.add_summaries(test_summaries)
    results = vs.query("attention mechanism")
    print(f"Query results: {len(results)}")
