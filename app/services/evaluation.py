from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import ndcg_score
import logging

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate retrieval quality with precision, recall, MRR, NDCG"""
    
    def __init__(self):
        self.name = "RetrievalEvaluator"
    
    def precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Precision@K score (0-1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_retrieved / k
    
    def recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Recall@K score (0-1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_set)
        
        return relevant_retrieved / len(relevant_docs)
    
    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
        
        Returns:
            MRR score (0-1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for idx, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                return 1.0 / idx
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            NDCG@K score (0-1)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_set = set(relevant_docs)
        
        # Create relevance scores (1 if relevant, 0 otherwise)
        relevance_scores = [1 if doc in relevant_set else 0 for doc in top_k]
        
        # Pad to k if needed
        while len(relevance_scores) < k:
            relevance_scores.append(0)
        
        # Calculate NDCG
        try:
            # sklearn expects 2D arrays
            true_relevance = np.array([relevance_scores])
            predicted_relevance = np.array([relevance_scores])
            
            return ndcg_score(true_relevance, predicted_relevance, k=k)
        except:
            return 0.0
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Comprehensive retrieval evaluation
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved_docs, relevant_docs, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved_docs, relevant_docs, k)
        
        metrics['mrr'] = self.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        
        return metrics


class QAEvaluator:
    """Evaluate QA answer quality"""
    
    def __init__(self, openai_client=None):
        self.name = "QAEvaluator"
        self.client = openai_client
    
    def exact_match(self, predicted: str, expected: str) -> float:
        """
        Exact match score (case-insensitive)
        
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0
    
    def contains_match(self, predicted: str, expected: str) -> float:
        """
        Check if expected answer is contained in predicted
        
        Returns:
            1.0 if contained, 0.0 otherwise
        """
        return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0
    
    def llm_as_judge(
        self,
        question: str,
        predicted: str,
        expected: str,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Use LLM to judge answer quality
        
        Args:
            question: Original question
            predicted: Generated answer
            expected: Expected/reference answer
            model: LLM model to use
        
        Returns:
            Dictionary with score and reasoning
        """
        if not self.client:
            logger.warning("No OpenAI client available for LLM-as-judge")
            return {"score": 0.0, "reasoning": "No LLM client"}
        
        prompt = f"""You are an expert evaluator. Compare the predicted answer to the expected answer for the given question.

Question: {question}

Expected Answer: {expected}

Predicted Answer: {predicted}

Rate the predicted answer on a scale of 0-10 where:
- 0-2: Completely wrong or irrelevant
- 3-5: Partially correct but missing key information
- 6-8: Mostly correct with minor issues
- 9-10: Excellent, matches or exceeds expected answer

Respond in JSON format:
{{"score": <0-10>, "reasoning": "<brief explanation>"}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert answer evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Normalize to 0-1
            result['score'] = result.get('score', 0) / 10.0
            
            return result
        
        except Exception as e:
            logger.error(f"Error in LLM-as-judge: {e}")
            return {"score": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def evaluate_answer(
        self,
        question: str,
        predicted: str,
        expected: str,
        use_llm_judge: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive answer evaluation
        
        Args:
            question: Original question
            predicted: Generated answer
            expected: Expected answer
            use_llm_judge: Whether to use LLM for evaluation
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'exact_match': self.exact_match(predicted, expected),
            'contains_match': self.contains_match(predicted, expected)
        }
        
        if use_llm_judge and self.client:
            llm_result = self.llm_as_judge(question, predicted, expected)
            metrics['llm_score'] = llm_result['score']
            metrics['llm_reasoning'] = llm_result.get('reasoning', '')
        
        return metrics
