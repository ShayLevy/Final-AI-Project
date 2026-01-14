"""
Response Cache Manager for Code-Based Graders

Generates RAG responses ONCE and caches them for unlimited free validation.
One-time cost: ~$0.40 for 28 queries. Subsequent validations: $0.00.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseCacheManager:
    """Manages cached RAG responses for code-based grader validation."""

    def __init__(self, cache_dir: str = "./evaluation_results/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "rag_response_cache.json"

    def load_cache(self) -> Dict[str, Any]:
        """Load cached responses from disk."""
        if not self.cache_file.exists():
            return {
                "created_at": None,
                "queries_count": 0,
                "responses": {}
            }

        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {
                "created_at": None,
                "queries_count": 0,
                "responses": {}
            }

    def save_cache(self, cache_data: Dict[str, Any]):
        """Save cached responses to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def generate_cache(self, rag_system, test_cases: list, progress_callback=None) -> Dict[str, Any]:
        """
        Generate cache by querying RAG system for all test cases.

        Args:
            rag_system: The RAG system instance with query() method
            test_cases: List of test case dictionaries with 'query' field
            progress_callback: Optional callback function(current, total, query)

        Returns:
            Cache data dictionary
        """
        cache_data = {
            "created_at": datetime.now().isoformat(),
            "queries_count": 0,
            "responses": {}
        }

        # Get unique queries
        unique_queries = list(set(tc.get("query", "") for tc in test_cases if tc.get("query")))
        total = len(unique_queries)

        logger.info(f"Generating cache for {total} unique queries...")

        for i, query in enumerate(unique_queries):
            if progress_callback:
                progress_callback(i + 1, total, query)

            try:
                # Query the RAG system
                result = rag_system.query(query)
                response = result.get("output", "")

                # Store in cache
                cache_data["responses"][query] = {
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                cache_data["queries_count"] += 1

                logger.info(f"Cached query {i+1}/{total}: {query[:50]}...")

            except Exception as e:
                logger.error(f"Failed to cache query '{query}': {e}")
                cache_data["responses"][query] = {
                    "response": "",
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)
                }

        # Save to disk
        self.save_cache(cache_data)

        logger.info(f"Cache generation complete! {cache_data['queries_count']}/{total} successful")
        return cache_data

    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for a query, or None if not found."""
        cache = self.load_cache()

        if query in cache["responses"]:
            cached = cache["responses"][query]
            if cached.get("success"):
                return cached.get("response")

        return None

    def has_cache(self) -> bool:
        """Check if cache exists and has responses."""
        cache = self.load_cache()
        return cache.get("queries_count", 0) > 0

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache metadata."""
        cache = self.load_cache()
        return {
            "exists": self.cache_file.exists(),
            "created_at": cache.get("created_at"),
            "queries_count": cache.get("queries_count", 0),
            "queries": list(cache.get("responses", {}).keys())
        }

    def clear_cache(self):
        """Delete cache file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached responses."""
        cache = self.load_cache()
        responses = cache.get("responses", {})

        successful = sum(1 for r in responses.values() if r.get("success"))
        failed = len(responses) - successful

        avg_length = 0
        if successful > 0:
            lengths = [len(r.get("response", "")) for r in responses.values() if r.get("success")]
            avg_length = sum(lengths) / len(lengths) if lengths else 0

        return {
            "total_queries": len(responses),
            "successful": successful,
            "failed": failed,
            "avg_response_length": int(avg_length),
            "cache_size_kb": self.cache_file.stat().st_size / 1024 if self.cache_file.exists() else 0
        }


def estimate_cache_cost(test_cases: list, cost_per_query: float = 0.04) -> Dict[str, Any]:
    """
    Estimate the one-time cost to generate cache.

    Args:
        test_cases: List of test case dictionaries
        cost_per_query: Estimated cost per RAG query (default $0.04)

    Returns:
        Dictionary with cost breakdown
    """
    unique_queries = list(set(tc.get("query", "") for tc in test_cases if tc.get("query")))
    total_queries = len(unique_queries)
    estimated_cost = total_queries * cost_per_query

    return {
        "total_test_cases": len(test_cases),
        "unique_queries": total_queries,
        "cost_per_query": cost_per_query,
        "estimated_total_cost": estimated_cost,
        "breakdown": {
            "manager_agent_calls": total_queries,
            "retrieval_calls": total_queries,
            "generation_calls": total_queries
        }
    }
