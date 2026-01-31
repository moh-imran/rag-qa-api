from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
import time
import random
from app.models.vector_store import QdrantVectorStore
from app.services.embedding_providers import BaseEmbeddingProvider, create_embedding_provider
from app.services.metrics_logger import metrics_logger
from app.services.collection_router import CollectionRouter
import uuid

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG pipeline for querying documents with LLM"""
    
    def __init__(
        self,
        embedding_provider: str = "huggingface",
        embedding_model: str = "all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        openai_api_key: str = None,
        openai_model: str = "gpt-4o-mini",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents",
        collections: Optional[Dict[str, str]] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_provider: Embedding provider type (huggingface, openai)
            embedding_model: Model name for embeddings
            cross_encoder_model: Cross encoder model name for reranking
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name (e.g., gpt-4o-mini)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection name
            collections: Optional dict of collection names to descriptions for federated search
        """
        self.embedding_provider_type = embedding_provider
        self.embedding_model_name = embedding_model
        self.cross_encoder_model_name = cross_encoder_model
        self.embedding_provider = None
        self.cross_encoder_model = None
        self.openai_model_name = openai_model
        self.openai_api_key = openai_api_key
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        # Configure OpenAI
        if openai_api_key:
            self.client = AsyncOpenAI(api_key=openai_api_key)
            logger.info(f"Initialized AsyncOpenAI client with model: {openai_model}")
        else:
            logger.warning("No OpenAI API key provided")
            self.client = None
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        # Initialize collection router if multiple collections provided
        if collections:
            self.collection_router = CollectionRouter(
                collections=collections,
                openai_client=self.client
            )
            logger.info(f"Initialized with {len(collections)} collections for federated search")
        else:
            self.collection_router = None
        
        logger.info("RAG Pipeline initialized")
    
    def _load_models(self):
        """Lazy load embedding and cross-encoder models"""
        if self.embedding_provider is None:
            logger.info(f"Loading embedding provider: {self.embedding_provider_type}")
            self.embedding_provider = create_embedding_provider(
                provider_type=self.embedding_provider_type,
                model_name=self.embedding_model_name,
                api_key=self.openai_api_key
            )
        
        if self.cross_encoder_model is None:
            logger.info(f"Loading cross-encoder model: {self.cross_encoder_model_name}")
            self.cross_encoder_model = CrossEncoder(self.cross_encoder_model_name)
            
        if not hasattr(self, 'sparse_embedding_model') or self.sparse_embedding_model is None:
            logger.info("Loading sparse embedding model: prithivida/Splade_PP_en_v1")
            self.sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    def embed_query_sparse(self, query: str) -> Dict[str, List]:
        """Convert query to sparse vector"""
        self._load_models()
        # embed returns generator, take first item for single query
        sparse_gen = self.sparse_embedding_model.embed([query])
        sparse_vec = list(sparse_gen)[0]
        
        return {
            "indices": sparse_vec.indices.tolist(),
            "values": sparse_vec.values.tolist()
        }

    def embed_query(self, query: str) -> List[float]:
        """
        Convert query text to embedding vector
        
        Args:
            query: User question
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_models()
        
        embedding = self.embedding_provider.embed(query)
        
        # Handle both single and batch embeddings
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        
        return embedding.tolist()
    
    async def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer using HyDE (Hypothetical Document Embeddings)
        
        Args:
            query: User question
            
        Returns:
            Hypothetical answer that can be embedded for better retrieval
        """
        if not self.client:
            logger.warning("HyDE requires OpenAI client, falling back to original query")
            return query
            
        hyde_prompt = f"""Write a detailed, factual answer to the following question as if you were writing a paragraph from a relevant document. Do not include phrases like "The answer is" or "According to". Just write the content directly.

Question: {query}

Answer:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates hypothetical document passages."},
                    {"role": "user", "content": hyde_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            logger.info(f"Generated HyDE document ({len(hypothetical_doc)} chars)")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"Error generating HyDE: {e}")
            return query
    
    async def retrieve_context(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hyde: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector store and rerank them
        
        Args:
            query: Original user question
            query_embedding: Query embedding vector
            top_k: Number of documents to return after reranking
            score_threshold: Minimum similarity score for initial retrieval
            metadata_filters: Optional metadata filters (e.g., {"filename": "doc.pdf"})
            use_hyde: If True, use HyDE for improved retrieval
            
        Returns:
            List of relevant documents with metadata
        """
        # 0. HyDE: Generate hypothetical document if enabled
        if use_hyde:
            hypothetical_doc = await self._generate_hypothetical_document(query)
            # Re-embed using hypothetical document
            query_embedding = self.embed_query(hypothetical_doc)
            logger.info("Using HyDE embedding for retrieval")
        
        # 1. Generate Sparse Vector for Hybrid Search
        sparse_vector = self.embed_query_sparse(query)
        
        # 2. Initial Retrieval (Fetch more candidates for reranking)
        initial_top_k = top_k * 3
        results = self.vector_store.search(
            query_vector=query_embedding,
            sparse_vector=sparse_vector,
            limit=initial_top_k,
            score_threshold=score_threshold,
            filter_dict=metadata_filters
        )
        
        if not results:
            return []
            
        logger.info(f"Initial retrieval: {len(results)} documents")
        
        # 2. Reranking
        self._load_models()
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['content']] for doc in results]
        
        # Predict scores
        cross_scores = self.cross_encoder_model.predict(pairs)
        
        # Attach new scores and sort
        for doc, score in zip(results, cross_scores):
            doc['score'] = float(score)  # Update score with reranker score
            
        # Sort by reranker score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. Return top_k
        final_results = results[:top_k]
        logger.info(f"Reranked top {len(final_results)} documents")
        
        return final_results
    
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

Instructions:
- Carefully analyze all context documents to find relevant information
- The documents may contain OCR-extracted text with formatting issues - look past these to find the actual content
- In invoices/business documents: bank details (IBAN, BIC) listed in headers/footers typically belong to the company whose letterhead it is (the sender/service provider)
- When you see a company name followed by address/contact info, the bank details in that same document section belong to that company
- Extract and combine information from multiple documents if needed
- Only say "I don't have enough information" if the answer truly cannot be found or inferred from the context

Context Documents:
{context}

User Question: {query}

Answer:"""
        
        return prompt
    
    async def generate_answer(
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
                response = await self.client.chat.completions.create(
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
    
    async def generate_answer_stream(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Generate answer using OpenAI LLM with streaming
        
        Args:
            prompt: Formatted prompt with context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            
        Yields:
            Tokens as they are generated
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        try:
            # Generate response using OpenAI Chat Completion with streaming
            stream = await self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error generating streaming answer: {e}")
            raise
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        return_sources: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hyde: bool = False,
        routing_strategy: Optional[str] = None,
        specific_collections: Optional[List[str]] = None
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
            metadata_filters: Optional metadata filters (e.g., {"filename": "doc.pdf"})
            use_hyde: If True, use HyDE for improved retrieval
            routing_strategy: "auto", "all", or "specific" (for federated search)
            specific_collections: List of collection names (when routing_strategy="specific")
            
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        
        # Federated search if collection router is available
        if self.collection_router and routing_strategy:
            return await self._federated_query(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
                system_instruction=system_instruction,
                max_tokens=max_tokens,
                temperature=temperature,
                return_sources=return_sources,
                metadata_filters=metadata_filters,
                use_hyde=use_hyde,
                routing_strategy=routing_strategy,
                specific_collections=specific_collections
            )
        
        # Step 1: Embed query
        query_embedding = self.embed_query(question)
        
        # Step 2: Retrieve relevant documents
        context_docs = await self.retrieve_context(
            question,
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
            use_hyde=use_hyde
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
        answer = await self.generate_answer(prompt, max_tokens, temperature)
        
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
        
        # Log metrics
        query_id = str(uuid.uuid4())
        response["query_id"] = query_id
        
        try:
            metrics_logger.log_query(
                query_id=query_id,
                question=question,
                retrieved_docs=context_docs,
                answer=answer,
                metadata={
                    "top_k": top_k,
                    "use_hyde": use_hyde,
                    "metadata_filters": metadata_filters
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
        
        return response
    
    async def _federated_query(
        self,
        question: str,
        top_k: int,
        score_threshold: Optional[float],
        system_instruction: Optional[str],
        max_tokens: int,
        temperature: float,
        return_sources: bool,
        metadata_filters: Optional[Dict[str, Any]],
        use_hyde: bool,
        routing_strategy: str,
        specific_collections: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute federated search across multiple collections"""
        # Route query to determine which collections to search
        target_collections = self.collection_router.route_query(
            query=question,
            strategy=routing_strategy,
            specific_collections=specific_collections
        )
        
        logger.info(f"Federated search across collections: {target_collections}")
        
        # Query each collection
        results_by_collection = {}
        original_collection = self.vector_store.collection_name
        
        try:
            for collection_name in target_collections:
                self.vector_store.switch_collection(collection_name)
                query_embedding = self.embed_query(question)
                context_docs = await self.retrieve_context(
                    question, query_embedding, top_k=top_k,
                    score_threshold=score_threshold,
                    metadata_filters=metadata_filters,
                    use_hyde=use_hyde
                )
                results_by_collection[collection_name] = context_docs
            
            merged_docs = self.collection_router.merge_results(results_by_collection, top_k=top_k)
        finally:
            self.vector_store.switch_collection(original_collection)
        
        if not merged_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context_used": False,
                "collections_queried": target_collections
            }
        
        prompt = self.build_prompt(question, merged_docs, system_instruction)
        answer = await self.generate_answer(prompt, max_tokens, temperature)
        
        response = {
            "answer": answer,
            "context_used": True,
            "collections_queried": target_collections
        }
        
        if return_sources:
            response["sources"] = [
                {"content": doc['content'][:200] + "...", "metadata": doc['metadata'], "score": doc.get('score', 0)}
                for doc in merged_docs
            ]
        
        query_id = str(uuid.uuid4())
        response["query_id"] = query_id
        
        try:
            metrics_logger.log_query(query_id=query_id, question=question, retrieved_docs=merged_docs, answer=answer,
                metadata={"top_k": top_k, "use_hyde": use_hyde, "routing_strategy": routing_strategy, "collections_queried": target_collections})
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
        
        return response
    
    async def query_stream(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hyde: bool = False
    ):
        """
        Complete RAG query pipeline with streaming
        
        Yields:
            Dict events with type and data
        """
        import json
        
        logger.info(f"Processing streaming query: {question}")
        
        # Yield retrieval start event
        yield {"type": "retrieval_start", "data": {"question": question}}
        
        # Step 1: Embed query
        query_embedding = self.embed_query(question)
        
        # Step 2: Retrieve relevant documents
        context_docs = await self.retrieve_context(
            question,
            query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filters=metadata_filters,
            use_hyde=use_hyde
        )
        
        # Yield retrieval complete event
        yield {
            "type": "retrieval_complete",
            "data": {
                "num_docs": len(context_docs),
                "sources": [
                    {
                        "content": doc['content'][:200] + "...",
                        "metadata": doc['metadata'],
                        "score": doc.get('score', 0)
                    }
                    for doc in context_docs
                ] if context_docs else []
            }
        }
        
        if not context_docs:
            yield {
                "type": "error",
                "data": {"message": "No relevant documents found"}
            }
            return
        
        # Step 3: Build prompt
        prompt = self.build_prompt(question, context_docs, system_instruction)
        
        # Yield generation start event
        yield {"type": "generation_start", "data": {}}
        
        # Step 4: Generate answer with streaming
        async for token in self.generate_answer_stream(prompt, max_tokens, temperature):
            yield {"type": "token", "data": {"content": token}}
        
        # Yield done event
        yield {"type": "done", "data": {}}
    
    async def _rephrase_question(self, messages: List[Dict[str, str]]) -> str:
        """Rephrase the latest question based on conversation history for optimized retrieval"""
        if len(messages) <= 1:
            return messages[-1]['content']
            
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        latest_question = messages[-1]['content']
        
        rephrase_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be used for document retrieval. 
Do NOT answer the question. Just return the rephrased question.

History:
{history_text}

Follow-up Question: {latest_question}

Standalone Question:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases questions for optimized search."},
                    {"role": "user", "content": rephrase_prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            rephrased = response.choices[0].message.content.strip()
            logger.info(f"Rephrased '{latest_question}' -> '{rephrased}'")
            return rephrased
        except Exception as e:
            logger.error(f"Error rephrasing question: {e}")
            return latest_question

    async def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hyde: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat interface with conversation history
        
        Args:
            messages: Conversation history
            top_k: Number of documents to retrieve
            metadata_filters: Optional metadata filters
            use_hyde: If True, use HyDE for improved retrieval
            **kwargs: Additional parameters
        """
        # 1. Rephrase question for better retrieval
        rephrased_query = await self._rephrase_question(messages)
        
        # 2. Retrieve documents using rephrased query
        query_embedding = self.embed_query(rephrased_query)
        context_docs = await self.retrieve_context(
            rephrased_query,
            query_embedding,
            top_k=top_k,
            metadata_filters=metadata_filters,
            use_hyde=use_hyde
        )
        
        # 3. Build prompt with history and context
        context_text = "\n\n".join([
            f"[Doc {i+1}]: {doc['content']}" for i, doc in enumerate(context_docs)
        ])
        
        system_msg = kwargs.get("system_instruction") or "You are a helpful AI assistant. Answer the user's question using the provided context and history."
        
        llm_messages = [
            {"role": "system", "content": f"{system_msg}\n\nRetrieved Context:\n{context_text}"}
        ]
        
        # Add conversation history
        llm_messages.extend(messages)
        
        # 4. Generate answer
        try:
            response = await self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=llm_messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "context_used": len(context_docs) > 0,
                "sources": [
                    {
                        "content": doc['content'][:200] + "...",
                        "metadata": doc['metadata'],
                        "score": doc.get('score', 0)
                    }
                    for doc in context_docs
                ]
            }
        except Exception as e:
            logger.error(f"Error in chat generation: {e}")
            raise
