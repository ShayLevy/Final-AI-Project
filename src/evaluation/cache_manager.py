"""
Evaluation Results Cache Manager

Caches evaluation results to avoid redundant API calls and save costs.
Results are stored by query ID and can be reused across sessions.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EvaluationCache:
    """
    Manages caching of evaluation results to disk.

    Cache structure:
    {
        "Q_NEEDLE_01": {
            "result": {...},  # Full EvaluationResult.to_dict()
            "timestamp": "2024-01-15T10:30:00",
            "model": "gpt-4",
            "cache_version": "1.0"
        },
        ...
    }
    """

    CACHE_VERSION = "1.0"

    def __init__(self, cache_dir: str = "./evaluation_cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.needle_cache_file = self.cache_dir / "needle_results_cache.json"
        self.summary_cache_file = self.cache_dir / "summary_results_cache.json"

        # Load existing caches
        self.needle_cache = self._load_cache(self.needle_cache_file)
        self.summary_cache = self._load_cache(self.summary_cache_file)

        logger.info(f"EvaluationCache initialized. Needle: {len(self.needle_cache)} cached, Summary: {len(self.summary_cache)} cached")

    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from file"""
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached results from {cache_file.name}")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache from {cache_file}: {e}")
                return {}
        return {}

    def _save_cache(self, cache: Dict[str, Any], cache_file: Path):
        """Save cache to file"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            logger.info(f"Saved {len(cache)} cached results to {cache_file.name}")
        except Exception as e:
            logger.error(f"Error saving cache to {cache_file}: {e}")

    def get_cached_result(self, query_id: str, query_type: str = "needle") -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Args:
            query_id: Query ID (e.g., "Q_NEEDLE_01")
            query_type: "needle" or "summary"

        Returns:
            Cached result dict if found, None otherwise
        """
        cache = self.needle_cache if query_type == "needle" else self.summary_cache

        if query_id in cache:
            cached_entry = cache[query_id]
            logger.info(f"Cache HIT for {query_id} (cached at {cached_entry.get('timestamp')})")
            return cached_entry.get('result')

        logger.info(f"Cache MISS for {query_id}")
        return None

    def cache_result(
        self,
        query_id: str,
        result: Dict[str, Any],
        query_type: str = "needle",
        model: str = "unknown"
    ):
        """
        Cache an evaluation result.

        Args:
            query_id: Query ID
            result: Full evaluation result from EvaluationResult.to_dict()
            query_type: "needle" or "summary"
            model: Model used for evaluation
        """
        cache = self.needle_cache if query_type == "needle" else self.summary_cache
        cache_file = self.needle_cache_file if query_type == "needle" else self.summary_cache_file

        cache[query_id] = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "cache_version": self.CACHE_VERSION
        }

        # Save to disk
        self._save_cache(cache, cache_file)
        logger.info(f"Cached result for {query_id} (model: {model})")

    def clear_cache(self, query_type: Optional[str] = None):
        """
        Clear cache.

        Args:
            query_type: "needle", "summary", or None (clear both)
        """
        if query_type == "needle" or query_type is None:
            self.needle_cache = {}
            self._save_cache(self.needle_cache, self.needle_cache_file)
            logger.info("Cleared needle cache")

        if query_type == "summary" or query_type is None:
            self.summary_cache = {}
            self._save_cache(self.summary_cache, self.summary_cache_file)
            logger.info("Cleared summary cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "needle_cached": len(self.needle_cache),
            "summary_cached": len(self.summary_cache),
            "total_cached": len(self.needle_cache) + len(self.summary_cache),
            "cache_dir": str(self.cache_dir),
            "needle_cache_file": str(self.needle_cache_file),
            "summary_cache_file": str(self.summary_cache_file)
        }

    def invalidate_query(self, query_id: str, query_type: str = "needle"):
        """
        Invalidate (remove) a specific cached query.

        Args:
            query_id: Query ID to invalidate
            query_type: "needle" or "summary"
        """
        cache = self.needle_cache if query_type == "needle" else self.summary_cache
        cache_file = self.needle_cache_file if query_type == "needle" else self.summary_cache_file

        if query_id in cache:
            del cache[query_id]
            self._save_cache(cache, cache_file)
            logger.info(f"Invalidated cache for {query_id}")
        else:
            logger.warning(f"Cache entry not found for {query_id}")

    def list_cached_queries(self, query_type: str = "needle") -> list:
        """
        List all cached query IDs.

        Args:
            query_type: "needle" or "summary"

        Returns:
            List of cached query IDs
        """
        cache = self.needle_cache if query_type == "needle" else self.summary_cache
        return list(cache.keys())

    def get_cache_info(self, query_id: str, query_type: str = "needle") -> Optional[Dict[str, Any]]:
        """
        Get cache metadata (without full result).

        Args:
            query_id: Query ID
            query_type: "needle" or "summary"

        Returns:
            Cache metadata dict or None
        """
        cache = self.needle_cache if query_type == "needle" else self.summary_cache

        if query_id in cache:
            entry = cache[query_id]
            return {
                "query_id": query_id,
                "timestamp": entry.get("timestamp"),
                "model": entry.get("model"),
                "cache_version": entry.get("cache_version"),
                "has_result": "result" in entry
            }
        return None
