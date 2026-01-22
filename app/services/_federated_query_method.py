    def _federated_query(
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
        """
        Execute federated search across multiple collections
        """
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
                # Switch to collection
                self.vector_store.switch_collection(collection_name)
                
                # Embed query
                query_embedding = self.embed_query(question)
                
                # Retrieve from this collection
                context_docs = self.retrieve_context(
                    question,
                    query_embedding,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    metadata_filters=metadata_filters,
                    use_hyde=use_hyde
                )
                
                results_by_collection[collection_name] = context_docs
            
            # Merge results
            merged_docs = self.collection_router.merge_results(
                results_by_collection,
                top_k=top_k
            )
            
        finally:
            # Restore original collection
            self.vector_store.switch_collection(original_collection)
        
        if not merged_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context_used": False,
                "collections_queried": target_collections
            }
        
        # Build prompt and generate answer
        prompt = self.build_prompt(question, merged_docs, system_instruction)
        answer = self.generate_answer(prompt, max_tokens, temperature)
        
        # Prepare response
        response = {
            "answer": answer,
            "context_used": True,
            "collections_queried": target_collections
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc['content'][:200] + "...",
                    "metadata": doc['metadata'],
                    "score": doc.get('score', 0)
                }
                for doc in merged_docs
            ]
        
        # Log metrics
        query_id = str(uuid.uuid4())
        response["query_id"] = query_id
        
        try:
            metrics_logger.log_query(
                query_id=query_id,
                question=question,
                retrieved_docs=merged_docs,
                answer=answer,
                metadata={
                    "top_k": top_k,
                    "use_hyde": use_hyde,
                    "metadata_filters": metadata_filters,
                    "routing_strategy": routing_strategy,
                    "collections_queried": target_collections
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
        
        return response
