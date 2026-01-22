from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class CollectionRouter:
    """Intelligent routing across multiple collections"""
    
    def __init__(
        self,
        collections: Dict[str, str],
        openai_client: Optional[OpenAI] = None,
        default_strategy: str = "auto"
    ):
        """
        Initialize collection router
        
        Args:
            collections: Dict mapping collection names to descriptions
                         e.g., {"docs": "Product documentation", "code": "Source code"}
            openai_client: Optional OpenAI client for LLM-based routing
            default_strategy: Default routing strategy ("auto", "all", "specific")
        """
        self.collections = collections
        self.client = openai_client
        self.default_strategy = default_strategy
        
        logger.info(f"CollectionRouter initialized with {len(collections)} collections")
    
    def route_query(
        self,
        query: str,
        strategy: str = None,
        specific_collections: Optional[List[str]] = None
    ) -> List[str]:
        """
        Determine which collections to query
        
        Args:
            query: User query
            strategy: "auto", "all", or "specific"
            specific_collections: List of collection names (for "specific" strategy)
        
        Returns:
            List of collection names to query
        """
        strategy = strategy or self.default_strategy
        
        if strategy == "all":
            return list(self.collections.keys())
        
        elif strategy == "specific":
            if not specific_collections:
                logger.warning("Specific strategy requested but no collections provided, using all")
                return list(self.collections.keys())
            
            # Validate collections exist
            valid_collections = [c for c in specific_collections if c in self.collections]
            if not valid_collections:
                logger.warning(f"No valid collections in {specific_collections}, using all")
                return list(self.collections.keys())
            
            return valid_collections
        
        elif strategy == "auto":
            return self._auto_route(query)
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', using all collections")
            return list(self.collections.keys())
    
    def _auto_route(self, query: str) -> List[str]:
        """
        Automatically determine relevant collections
        
        Uses LLM if available, otherwise falls back to keyword matching
        """
        if self.client:
            return self._llm_route(query)
        else:
            return self._keyword_route(query)
    
    def _llm_route(self, query: str) -> List[str]:
        """Use LLM to determine relevant collections"""
        
        # Build collection descriptions
        collection_info = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.collections.items()
        ])
        
        prompt = f"""Given the following query and available collections, determine which collection(s) are most relevant to answer the query.

Query: {query}

Available Collections:
{collection_info}

Respond with a JSON array of collection names that should be queried. Include only the most relevant collections (1-3 typically).

Example response: ["docs", "code"]

Response:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query routing expert. Respond only with a JSON array of collection names."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Extract collection names
            if isinstance(result, dict):
                collections = result.get('collections', list(result.values())[0] if result else [])
            else:
                collections = result
            
            # Validate
            valid_collections = [c for c in collections if c in self.collections]
            
            if valid_collections:
                logger.info(f"LLM routed query to: {valid_collections}")
                return valid_collections
            else:
                logger.warning("LLM returned no valid collections, using all")
                return list(self.collections.keys())
        
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}, falling back to keyword routing")
            return self._keyword_route(query)
    
    def _keyword_route(self, query: str) -> List[str]:
        """
        Simple keyword-based routing
        
        Matches query keywords to collection names/descriptions
        """
        query_lower = query.lower()
        matched_collections = []
        
        for name, description in self.collections.items():
            # Check if collection name or description keywords appear in query
            keywords = [name.lower()] + description.lower().split()
            
            if any(keyword in query_lower for keyword in keywords):
                matched_collections.append(name)
        
        # If no matches, use all collections
        if not matched_collections:
            logger.info("No keyword matches, using all collections")
            return list(self.collections.keys())
        
        logger.info(f"Keyword routing matched: {matched_collections}")
        return matched_collections
    
    def merge_results(
        self,
        results_by_collection: Dict[str, List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Merge and rerank results from multiple collections
        
        Args:
            results_by_collection: Dict mapping collection names to result lists
            top_k: Number of results to return
        
        Returns:
            Merged and sorted results
        """
        # Combine all results
        all_results = []
        
        for collection_name, results in results_by_collection.items():
            for result in results:
                # Add collection metadata
                result['metadata']['collection'] = collection_name
                all_results.append(result)
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Return top k
        return all_results[:top_k]
    
    def add_collection(self, name: str, description: str):
        """Add a new collection"""
        self.collections[name] = description
        logger.info(f"Added collection: {name}")
    
    def remove_collection(self, name: str):
        """Remove a collection"""
        if name in self.collections:
            del self.collections[name]
            logger.info(f"Removed collection: {name}")
    
    def list_collections(self) -> Dict[str, str]:
        """Get all collections with descriptions"""
        return self.collections.copy()
