from typing import Dict, Any, List, Optional
import json
import csv
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Log and track RAG metrics, user feedback, and performance"""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.queries_log = self.log_dir / "queries.jsonl"
        self.feedback_log = self.log_dir / "feedback.jsonl"
        self.metrics_log = self.log_dir / "metrics.jsonl"
        
        logger.info(f"MetricsLogger initialized. Logs: {self.log_dir}")
    
    def log_query(
        self,
        query_id: str,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a query with retrieval and generation details
        
        Args:
            query_id: Unique query identifier
            question: User question
            retrieved_docs: List of retrieved documents with scores
            answer: Generated answer
            metadata: Additional metadata (latency, tokens, etc.)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_id": query_id,
            "question": question,
            "num_retrieved": len(retrieved_docs),
            "retrieval_scores": [doc.get('score', 0) for doc in retrieved_docs],
            "avg_retrieval_score": sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
            "answer_length": len(answer),
            "metadata": metadata or {}
        }
        
        self._append_jsonl(self.queries_log, log_entry)
    
    def log_feedback(
        self,
        query_id: str,
        feedback_type: str,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
        correction: Optional[str] = None
    ):
        """
        Log user feedback
        
        Args:
            query_id: Query identifier
            feedback_type: "thumbs_up", "thumbs_down", "correction"
            rating: Optional 1-5 rating
            comment: Optional text comment
            correction: Optional corrected answer
        """
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query_id": query_id,
            "feedback_type": feedback_type,
            "rating": rating,
            "comment": comment,
            "correction": correction
        }
        
        self._append_jsonl(self.feedback_log, feedback_entry)
        logger.info(f"Logged feedback for query {query_id}: {feedback_type}")
    
    def log_metrics(
        self,
        metric_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log evaluation metrics
        
        Args:
            metric_type: "retrieval" or "qa"
            metrics: Dictionary of metric values
            metadata: Additional context
        """
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_type": metric_type,
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        self._append_jsonl(self.metrics_log, metrics_entry)
    
    def get_aggregated_metrics(
        self,
        metric_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated metrics over time period
        
        Args:
            metric_type: Filter by metric type
            start_date: Start of time range
            end_date: End of time range
        
        Returns:
            Aggregated metrics
        """
        if not self.metrics_log.exists():
            return {}
        
        metrics_data = []
        
        with open(self.metrics_log, 'r') as f:
            for line in f:
                entry = json.loads(line)
                
                # Filter by type
                if metric_type and entry.get('metric_type') != metric_type:
                    continue
                
                # Filter by date
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                
                metrics_data.append(entry)
        
        # Aggregate
        if not metrics_data:
            return {}
        
        aggregated = {
            "count": len(metrics_data),
            "metrics": {}
        }
        
        # Average all numeric metrics
        all_metric_keys = set()
        for entry in metrics_data:
            all_metric_keys.update(entry['metrics'].keys())
        
        for key in all_metric_keys:
            values = [entry['metrics'].get(key, 0) for entry in metrics_data if key in entry['metrics']]
            if values:
                aggregated['metrics'][key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return aggregated
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary of user feedback
        
        Returns:
            Feedback statistics
        """
        if not self.feedback_log.exists():
            return {}
        
        feedback_data = []
        
        with open(self.feedback_log, 'r') as f:
            for line in f:
                feedback_data.append(json.loads(line))
        
        if not feedback_data:
            return {}
        
        summary = {
            "total_feedback": len(feedback_data),
            "by_type": {},
            "avg_rating": None
        }
        
        # Count by type
        for entry in feedback_data:
            ftype = entry.get('feedback_type', 'unknown')
            summary['by_type'][ftype] = summary['by_type'].get(ftype, 0) + 1
        
        # Average rating
        ratings = [entry['rating'] for entry in feedback_data if entry.get('rating')]
        if ratings:
            summary['avg_rating'] = sum(ratings) / len(ratings)
        
        return summary
    
    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all raw feedback entries
        
        Returns:
            List of feedback entries
        """
        if not self.feedback_log.exists():
            return []
        
        feedback_data = []
        
        with open(self.feedback_log, 'r') as f:
            for line in f:
                try:
                    feedback_data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # Sort by timestamp descending
        feedback_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return feedback_data

    def export_to_csv(self, output_file: str, log_type: str = "queries"):
        """
        Export logs to CSV
        
        Args:
            output_file: Output CSV file path
            log_type: "queries", "feedback", or "metrics"
        """
        log_map = {
            "queries": self.queries_log,
            "feedback": self.feedback_log,
            "metrics": self.metrics_log
        }
        
        log_file = log_map.get(log_type)
        if not log_file or not log_file.exists():
            logger.warning(f"No log file found for type: {log_type}")
            return
        
        # Read all entries
        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))
        
        if not entries:
            return
        
        # Get all keys
        all_keys = set()
        for entry in entries:
            all_keys.update(self._flatten_dict(entry).keys())
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for entry in entries:
                writer.writerow(self._flatten_dict(entry))
        
        logger.info(f"Exported {len(entries)} entries to {output_file}")
    
    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Append JSON line to file"""
        with open(file_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)


# Global instance
metrics_logger = MetricsLogger()
