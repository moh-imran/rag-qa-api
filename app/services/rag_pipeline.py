from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time
import random
from app.models.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for querying documents with LLM"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: str = None,
        openai_model: str = "gpt-4o-mini",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model: Sentence transformer model name
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name (e.g., gpt-4o-mini)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection name
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.openai_model_name = openai_model
        
        # Configure OpenAI
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
            logger.info(f"Initialized OpenAI client with model: {openai_model}")
        else:
            logger.warning("No OpenAI API key provided")
            self.client = None
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        logger.info("RAG Pipeline initialized")
    
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query text to embedding vector
        
        Args:
            query: User question
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_embedding_model()
        
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
    
    def retrieve_context(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold
        )
        
        logger.info(f"Retrieved {len(results)} documents")
        return results
    
    def build_prompt(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Build prompt for LLM with context
        
        Args:
            query: User question
            context_docs: Retrieved relevant documents
            system_instruction: Optional system instruction
            
        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source = doc['metadata'].get('filename', 'Unknown')
            content = doc['content']
            score = doc.get('score', 0)
            
            context_parts.append(
                f"[Document {i} - Source: {source} - Relevance: {score:.2f}]\n{content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        if system_instruction:
            prompt = f"""{system_instruction}

Context Documents:
{context}

User Question: {query}

Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents.

If the answer is not in the context, say "I don't have enough information to answer this question."

Context Documents:
{context}

User Question: {query}

Answer:"""
        
        return prompt
    
    def generate_answer(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer using OpenAI LLM
        
        Args:
            prompt: Formatted prompt with context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Returns:
            Generated answer
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")        

        max_retries = 3
        retry_delay = 1.0  # Initial delay in seconds

        for attempt in range(max_retries + 1):
            try:
                # Generate response using OpenAI Chat Completion
                response = self.client.chat.completions.create(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                answer = response.choices[0].message.content
                logger.info(f"Generated answer: {len(answer)} characters")
                
                return answer
                
            except Exception as e:
                # Check for 429 Resource Exhausted
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries:
                        sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limited (429). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                
                logger.error(f"Error generating answer: {e}")
                raise
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG query pipeline
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            system_instruction: Optional system instruction
            max_tokens: Maximum tokens for answer
            temperature: LLM temperature
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Embed query
        query_embedding = self.embed_query(question)
        
        # Step 2: Retrieve relevant documents
        context_docs = self.retrieve_context(
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not context_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context_used": False
            }
        
        # Step 3: Build prompt
        prompt = self.build_prompt(question, context_docs, system_instruction)
        
        # Step 4: Generate answer
        answer = self.generate_answer(prompt, max_tokens, temperature)
        
        # Prepare response
        response = {
            "answer": answer,
            "context_used": True
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc['content'][:200] + "...",
                    "metadata": doc['metadata'],
                    "score": doc.get('score', 0)
                }
                for doc in context_docs
            ]
        
        return response
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat interface with conversation history
        
        Args:
            messages: List of conversation messages [{"role": "user", "content": "..."}]
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for query()
            
        Returns:
            Dictionary with answer and sources
        """
        # Get the latest user message
        latest_message = messages[-1]['content']
        
        # For now, just process the latest message
        # In future, can implement conversation memory
        return self.query(latest_message, top_k=top_k, **kwargs)
