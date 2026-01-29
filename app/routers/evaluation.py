from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import uuid
from datetime import datetime

from app.services.evaluation import RetrievalEvaluator, QAEvaluator
from app.services.metrics_logger import metrics_logger

router = APIRouter(prefix="/evaluation", tags=["evaluation"])
logger = logging.getLogger(__name__)

# Global evaluators
retrieval_evaluator = RetrievalEvaluator()
qa_evaluator = None  # Will be set with OpenAI client if available


def set_qa_evaluator(openai_client):
    """Set QA evaluator with OpenAI client"""
    global qa_evaluator
    qa_evaluator = QAEvaluator(openai_client)


# Request/Response Models
class BenchmarkItem(BaseModel):
    question: str = Field(..., description="Question to evaluate")
    expected_answer: str = Field(..., description="Expected/reference answer")
    relevant_doc_ids: List[str] = Field(..., description="IDs of relevant documents")


class BenchmarkRequest(BaseModel):
    items: List[BenchmarkItem] = Field(..., description="List of benchmark items")
    use_llm_judge: bool = Field(False, description="Use LLM for answer evaluation")


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query identifier")
    feedback_type: str = Field(..., description="thumbs_up, thumbs_down, or correction")
    rating: Optional[int] = Field(None, description="1-5 rating", ge=1, le=5)
    comment: Optional[str] = Field(None, description="Optional comment")
    correction: Optional[str] = Field(None, description="Corrected answer")


@router.post("/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """
    Run evaluation on benchmark dataset
    
    Returns aggregated metrics across all items
    """
    try:
        from app.routers.chat import rag_pipeline
        
        all_retrieval_metrics = []
        all_qa_metrics = []
        
        for item in request.items:
            # Run query
            result = rag_pipeline.query(
                question=item.question,
                return_sources=True
            )
            
            # Extract retrieved doc IDs
            retrieved_ids = [
                source['metadata'].get('filepath', source['metadata'].get('filename', str(idx)))
                for idx, source in enumerate(result.get('sources', []))
            ]
            
            # Evaluate retrieval
            retrieval_metrics = retrieval_evaluator.evaluate_retrieval(
                retrieved_docs=retrieved_ids,
                relevant_docs=item.relevant_doc_ids
            )
            all_retrieval_metrics.append(retrieval_metrics)
            
            # Evaluate answer
            qa_metrics = await qa_evaluator.evaluate_answer(
                question=item.question,
                predicted=result['answer'],
                expected=item.expected_answer,
                use_llm_judge=request.use_llm_judge
            ) if qa_evaluator else {}
            all_qa_metrics.append(qa_metrics)
        
        # Aggregate metrics
        aggregated_retrieval = _aggregate_metrics(all_retrieval_metrics)
        aggregated_qa = _aggregate_metrics(all_qa_metrics)
        
        # Log metrics
        metrics_logger.log_metrics("retrieval", aggregated_retrieval)
        metrics_logger.log_metrics("qa", aggregated_qa)
        
        return {
            "num_items": len(request.items),
            "retrieval_metrics": aggregated_retrieval,
            "qa_metrics": aggregated_qa
        }
    
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    metric_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get aggregated metrics
    
    Args:
        metric_type: Filter by "retrieval" or "qa"
        start_date: ISO format date string
        end_date: ISO format date string
    """
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None
        
        metrics = metrics_logger.get_aggregated_metrics(
            metric_type=metric_type,
            start_date=start,
            end_date=end
        )
        
        feedback = metrics_logger.get_feedback_summary()
        
        return {
            "metrics": metrics,
            "feedback_summary": feedback
        }
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query
    """
    try:
        metrics_logger.log_feedback(
            query_id=request.query_id,
            feedback_type=request.feedback_type,
            rating=request.rating,
            comment=request.comment,
            correction=request.correction
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded"
        }
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback")
async def list_feedback():
    """
    Get list of all user feedback
    """
    try:
        return metrics_logger.get_all_feedback()
    except Exception as e:
        logger.error(f"Error listing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{log_type}")
async def export_logs(log_type: str):
    """
    Export logs to CSV
    
    Args:
        log_type: "queries", "feedback", or "metrics"
    """
    try:
        output_file = f"logs/exports/{log_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        metrics_logger.export_to_csv(output_file, log_type)
        
        return {
            "status": "success",
            "file": output_file
        }
    
    except Exception as e:
        logger.error(f"Error exporting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate list of metric dictionaries"""
    if not metrics_list:
        return {}
    
    aggregated = {}
    all_keys = set()
    
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
    
    return aggregated
