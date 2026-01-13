"""
Regression Tracking System for RAG Evaluation

This module provides tools for tracking evaluation performance over time,
managing baselines, calculating deltas, and detecting regressions.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import uuid


@dataclass
class QueryResult:
    """Per-query result for detailed tracking"""
    query_id: str
    query_text: str
    scores: Dict[str, float]
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """Unified schema for all evaluation types"""
    run_id: str
    timestamp: str
    evaluation_type: str  # "ragas" | "llm_judge" | "code_graders_*"
    grader_subtype: Optional[str]  # For code graders: "exact_match", "numerical", etc.
    aggregate_scores: Dict[str, float]  # Normalized to 0-1 scale
    query_results: List[QueryResult]
    num_queries: int
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Baseline:
    """Baseline configuration and scores"""
    baseline_id: str
    created_at: str
    source_run_id: str
    evaluation_type: str
    aggregate_scores: Dict[str, float]
    per_query_scores: Dict[str, Dict[str, float]]
    thresholds: Dict[str, float]
    description: str
    is_active: bool = True


class RegressionTracker:
    """
    Main regression tracking system.

    Features:
    - Record evaluation runs with normalized scores
    - Set/clear baselines for comparison
    - Calculate deltas (current vs baseline, current vs previous)
    - Check for regressions against configurable thresholds
    - Track trends over time
    """

    DEFAULT_THRESHOLDS = {
        # RAGAS metrics (0-1 scale) - flag if drops more than these amounts
        "faithfulness": 0.05,
        "answer_relevancy": 0.05,
        "context_precision": 0.05,
        "context_recall": 0.05,

        # LLM-as-a-Judge (normalized from 1-5 to 0-1)
        "correctness": 0.10,
        "relevancy": 0.10,
        "recall": 0.10,
        "average": 0.10,

        # Code graders (0-1 pass rate)
        "pass_rate": 0.10,
    }

    def __init__(self, data_dir: str = "./evaluation_results"):
        self.data_dir = Path(data_dir)
        self.baselines_dir = self.data_dir / "baselines"
        self.history_dir = self.data_dir / "history"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.baselines_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)

        # In-memory caches
        self._baselines: Dict[str, Baseline] = {}
        self._history: Dict[str, List[EvaluationRun]] = {}

        self._load_data()

    def _load_data(self):
        """Load baselines and history from disk"""
        # Load baselines
        for baseline_file in self.baselines_dir.glob("*_baseline.json"):
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    baseline = Baseline(**data)
                    self._baselines[baseline.evaluation_type] = baseline
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Warning: Could not load baseline {baseline_file}: {e}")

        # Load history
        for history_file in self.history_dir.glob("*_history.json"):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    eval_type = history_file.stem.replace("_history", "")
                    runs = []
                    for run_data in data:
                        query_results = [
                            QueryResult(**qr) for qr in run_data.pop("query_results", [])
                        ]
                        runs.append(EvaluationRun(**run_data, query_results=query_results))
                    self._history[eval_type] = runs
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"Warning: Could not load history {history_file}: {e}")

    def _save_baseline(self, baseline: Baseline):
        """Save baseline to disk"""
        filepath = self.baselines_dir / f"{baseline.evaluation_type}_baseline.json"
        with open(filepath, 'w') as f:
            json.dump(asdict(baseline), f, indent=2, default=str)

    def _save_history(self, evaluation_type: str):
        """Save history for an evaluation type to disk"""
        filepath = self.history_dir / f"{evaluation_type}_history.json"
        runs = self._history.get(evaluation_type, [])

        # Convert to serializable format
        data = []
        for run in runs:
            run_dict = asdict(run)
            data.append(run_dict)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def normalize_scores(scores: Dict[str, Any], evaluation_type: str) -> Dict[str, float]:
        """Normalize all scores to 0-1 scale for comparison"""
        normalized = {}

        if evaluation_type == "ragas":
            # Already 0-1, handle None values
            for metric, value in scores.items():
                if isinstance(value, list):
                    valid_values = [v for v in value if v is not None]
                    normalized[metric] = sum(valid_values) / len(valid_values) if valid_values else 0
                elif value is not None:
                    normalized[metric] = float(value)

        elif evaluation_type == "llm_judge":
            # 1-5 scale -> 0-1
            for metric, value in scores.items():
                if isinstance(value, (int, float)) and value is not None:
                    normalized[metric] = float(value) / 5.0

        elif evaluation_type.startswith("code_graders"):
            # pass_rate is 0-100 -> 0-1
            if "pass_rate" in scores:
                normalized["pass_rate"] = float(scores["pass_rate"]) / 100.0
            if "passed" in scores:
                normalized["passed"] = float(scores["passed"])
            if "total_tests" in scores:
                normalized["total_tests"] = float(scores["total_tests"])

        return normalized

    def _extract_query_results(self, results: Dict[str, Any], evaluation_type: str) -> List[QueryResult]:
        """Extract per-query results from raw evaluation results"""
        query_results = []

        if evaluation_type == "ragas":
            # RAGAS stores results in details dict with parallel lists
            details = results.get("details", {})
            questions = details.get("question", [])
            answers = details.get("answer", [])

            for i, (q, a) in enumerate(zip(questions, answers)):
                qr = QueryResult(
                    query_id=f"Q{i+1}",
                    query_text=q,
                    scores={},
                    details={"answer": a}
                )
                query_results.append(qr)

        elif evaluation_type == "llm_judge":
            # LLM Judge stores query_results list
            for qr_data in results.get("query_results", []):
                qr = QueryResult(
                    query_id=qr_data.get("question", "")[:20],
                    query_text=qr_data.get("question", ""),
                    scores={
                        "correctness": qr_data.get("correctness", 0) / 5.0,
                        "relevancy": qr_data.get("relevancy", 0) / 5.0,
                    },
                    details=qr_data.get("details", {})
                )
                query_results.append(qr)

        elif evaluation_type.startswith("code_graders"):
            # Code graders store results list
            for r in results.get("results", []):
                qr = QueryResult(
                    query_id=r.get("test_id", "unknown"),
                    query_text=r.get("query", ""),
                    scores={"passed": 1.0 if r.get("passed") else 0.0},
                    passed=r.get("passed"),
                    details={
                        "expected": r.get("expected"),
                        "found": r.get("found"),
                    }
                )
                query_results.append(qr)

        return query_results

    def record_run(self, evaluation_type: str, results: Dict[str, Any],
                   grader_subtype: str = None, config: Dict = None) -> EvaluationRun:
        """Record a new evaluation run to history"""

        run_id = f"{evaluation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Normalize scores based on evaluation type
        if evaluation_type == "ragas":
            raw_scores = results.get("scores", {})
            aggregate_scores = self.normalize_scores(raw_scores, "ragas")
        elif evaluation_type == "llm_judge":
            raw_scores = results.get("scores", {})
            aggregate_scores = self.normalize_scores(raw_scores, "llm_judge")
        elif evaluation_type.startswith("code_graders"):
            raw_scores = results.get("summary", {})
            aggregate_scores = self.normalize_scores(raw_scores, "code_graders")
        else:
            aggregate_scores = {}

        query_results = self._extract_query_results(results, evaluation_type)

        run = EvaluationRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            evaluation_type=evaluation_type,
            grader_subtype=grader_subtype,
            aggregate_scores=aggregate_scores,
            query_results=query_results,
            num_queries=len(query_results),
            config=config or {}
        )

        # Add to history and persist
        if evaluation_type not in self._history:
            self._history[evaluation_type] = []
        self._history[evaluation_type].insert(0, run)  # Most recent first

        # Limit history size
        self._history[evaluation_type] = self._history[evaluation_type][:50]

        self._save_history(evaluation_type)
        return run

    def set_baseline(self, run_id: str, description: str = "",
                     custom_thresholds: Dict[str, float] = None) -> Optional[Baseline]:
        """Set a run as the baseline for its evaluation type"""

        # Find the run
        run = self._get_run_by_id(run_id)
        if not run:
            return None

        # Extract per-query scores
        per_query_scores = {}
        for qr in run.query_results:
            per_query_scores[qr.query_id] = qr.scores.copy()
            if qr.passed is not None:
                per_query_scores[qr.query_id]["passed"] = 1.0 if qr.passed else 0.0

        baseline = Baseline(
            baseline_id=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            source_run_id=run_id,
            evaluation_type=run.evaluation_type,
            aggregate_scores=run.aggregate_scores.copy(),
            per_query_scores=per_query_scores,
            thresholds=custom_thresholds or self.DEFAULT_THRESHOLDS.copy(),
            description=description,
            is_active=True
        )

        self._baselines[run.evaluation_type] = baseline
        self._save_baseline(baseline)

        return baseline

    def clear_baseline(self, evaluation_type: str) -> bool:
        """Clear the baseline for an evaluation type"""
        if evaluation_type in self._baselines:
            del self._baselines[evaluation_type]
            filepath = self.baselines_dir / f"{evaluation_type}_baseline.json"
            if filepath.exists():
                filepath.unlink()
            return True
        return False

    def get_baseline(self, evaluation_type: str) -> Optional[Baseline]:
        """Get the current baseline for an evaluation type"""
        return self._baselines.get(evaluation_type)

    def _get_run_by_id(self, run_id: str) -> Optional[EvaluationRun]:
        """Find a run by its ID across all evaluation types"""
        for runs in self._history.values():
            for run in runs:
                if run.run_id == run_id:
                    return run
        return None

    def _extract_per_query_scores(self, run: EvaluationRun) -> Dict[str, Dict[str, float]]:
        """Extract per-query scores from a run"""
        per_query = {}
        for qr in run.query_results:
            per_query[qr.query_id] = qr.scores.copy()
            if qr.passed is not None:
                per_query[qr.query_id]["passed"] = 1.0 if qr.passed else 0.0
        return per_query

    def calculate_deltas(self, run_id: str,
                         compare_to: str = "baseline") -> Dict[str, Any]:
        """
        Calculate metric deltas between runs.

        Args:
            run_id: ID of the current run
            compare_to: "baseline", "previous", or specific run_id

        Returns:
            {
                "aggregate": {"metric": delta, ...},
                "per_query": {"query_id": {"metric": delta, ...}, ...},
                "comparison_type": "baseline" | "previous" | "run_id"
            }
        """
        current_run = self._get_run_by_id(run_id)
        if not current_run:
            return {"aggregate": {}, "per_query": {}, "error": f"Run {run_id} not found"}

        # Determine comparison target
        if compare_to == "baseline":
            baseline = self.get_baseline(current_run.evaluation_type)
            if not baseline:
                return {"aggregate": {}, "per_query": {}, "error": "No baseline set"}
            target_aggregate = baseline.aggregate_scores
            target_per_query = baseline.per_query_scores
            comparison_type = "baseline"

        elif compare_to == "previous":
            history = self.get_history(current_run.evaluation_type, limit=10)
            # Find current run's position and get the next one
            current_idx = next((i for i, r in enumerate(history) if r.run_id == run_id), -1)
            if current_idx == -1 or current_idx >= len(history) - 1:
                return {"aggregate": {}, "per_query": {}, "error": "No previous run"}
            prev_run = history[current_idx + 1]
            target_aggregate = prev_run.aggregate_scores
            target_per_query = self._extract_per_query_scores(prev_run)
            comparison_type = "previous"

        else:
            target_run = self._get_run_by_id(compare_to)
            if not target_run:
                return {"aggregate": {}, "per_query": {}, "error": f"Run {compare_to} not found"}
            target_aggregate = target_run.aggregate_scores
            target_per_query = self._extract_per_query_scores(target_run)
            comparison_type = "run_id"

        # Calculate aggregate deltas
        aggregate_deltas = {}
        for metric, current_val in current_run.aggregate_scores.items():
            target_val = target_aggregate.get(metric, 0)
            aggregate_deltas[metric] = current_val - target_val

        # Calculate per-query deltas
        per_query_deltas = {}
        current_per_query = self._extract_per_query_scores(current_run)
        for query_id, current_scores in current_per_query.items():
            target_scores = target_per_query.get(query_id, {})
            per_query_deltas[query_id] = {}
            for metric, current_val in current_scores.items():
                target_val = target_scores.get(metric, 0)
                per_query_deltas[query_id][metric] = current_val - target_val

        return {
            "aggregate": aggregate_deltas,
            "per_query": per_query_deltas,
            "comparison_type": comparison_type
        }

    def check_regressions(self, run_id: str,
                          thresholds: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Check for regressions and generate alerts.

        Returns list of alerts:
            [
                {
                    "type": "aggregate" | "per_query",
                    "comparison": "vs_baseline" | "vs_previous",
                    "metric": "faithfulness",
                    "current": 0.75,
                    "target": 0.85,
                    "delta": -0.10,
                    "threshold": 0.05,
                    "severity": "warning" | "critical",
                    "message": "..."
                }
            ]
        """
        alerts = []
        effective_thresholds = thresholds or self.DEFAULT_THRESHOLDS
        current_run = self._get_run_by_id(run_id)

        if not current_run:
            return alerts

        # Check vs baseline
        baseline = self.get_baseline(current_run.evaluation_type)
        if baseline:
            baseline_deltas = self.calculate_deltas(run_id, "baseline")

            for metric, delta in baseline_deltas.get("aggregate", {}).items():
                threshold = effective_thresholds.get(metric, 0.05)

                if delta < -threshold:
                    # Determine severity
                    severity = "critical" if delta < -threshold * 2 else "warning"

                    alerts.append({
                        "type": "aggregate",
                        "comparison": "vs_baseline",
                        "metric": metric,
                        "current": current_run.aggregate_scores.get(metric, 0),
                        "target": baseline.aggregate_scores.get(metric, 0),
                        "delta": delta,
                        "threshold": threshold,
                        "severity": severity,
                        "message": f"{metric.replace('_', ' ').title()} dropped by {abs(delta):.1%} vs baseline (threshold: {threshold:.1%})"
                    })

        # Check vs previous run
        prev_deltas = self.calculate_deltas(run_id, "previous")
        if "error" not in prev_deltas:
            for metric, delta in prev_deltas.get("aggregate", {}).items():
                threshold = effective_thresholds.get(metric, 0.05) * 1.5  # Slightly higher for previous

                if delta < -threshold:
                    alerts.append({
                        "type": "aggregate",
                        "comparison": "vs_previous",
                        "metric": metric,
                        "current": current_run.aggregate_scores.get(metric, 0),
                        "delta": delta,
                        "threshold": threshold,
                        "severity": "warning",
                        "message": f"{metric.replace('_', ' ').title()} dropped by {abs(delta):.1%} vs previous run"
                    })

        # Sort by severity (critical first) then by delta magnitude
        return sorted(alerts, key=lambda x: (x["severity"] == "warning", x["delta"]))

    def get_history(self, evaluation_type: str, limit: int = 20) -> List[EvaluationRun]:
        """Get evaluation history for an evaluation type"""
        runs = self._history.get(evaluation_type, [])
        return runs[:limit]

    def get_trends(self, evaluation_type: str, metric: str,
                   limit: int = 10) -> List[Tuple[str, float]]:
        """Get metric trends over time"""
        runs = self.get_history(evaluation_type, limit)
        trends = []
        for run in reversed(runs):  # Oldest first
            value = run.aggregate_scores.get(metric)
            if value is not None:
                trends.append((run.timestamp[:16], value))
        return trends

    def get_latest_run(self, evaluation_type: str) -> Optional[EvaluationRun]:
        """Get the most recent run for an evaluation type"""
        runs = self._history.get(evaluation_type, [])
        return runs[0] if runs else None

    @staticmethod
    def run_self_tests():
        """Run self-tests to verify functionality"""
        import tempfile
        import shutil

        print("Running RegressionTracker self-tests...")

        # Create temporary directory for tests
        temp_dir = tempfile.mkdtemp()

        try:
            tracker = RegressionTracker(data_dir=temp_dir)

            # Test 1: Record a run
            mock_results = {
                "scores": {"faithfulness": 0.85, "answer_relevancy": 0.90},
                "details": {"question": ["Q1", "Q2"], "answer": ["A1", "A2"]}
            }
            run = tracker.record_run("ragas", mock_results)
            assert run.run_id is not None
            assert len(run.aggregate_scores) > 0
            print("PASS: Test 1 - Record run")

            # Test 2: Set baseline
            baseline = tracker.set_baseline(run.run_id, description="Test baseline")
            assert baseline is not None
            assert baseline.evaluation_type == "ragas"
            print("PASS: Test 2 - Set baseline")

            # Test 3: Get baseline
            retrieved_baseline = tracker.get_baseline("ragas")
            assert retrieved_baseline is not None
            assert retrieved_baseline.baseline_id == baseline.baseline_id
            print("PASS: Test 3 - Get baseline")

            # Test 4: Record another run and calculate deltas
            mock_results_2 = {
                "scores": {"faithfulness": 0.75, "answer_relevancy": 0.92},  # Dropped by 0.10
                "details": {"question": ["Q1", "Q2"], "answer": ["A1", "A2"]}
            }
            run2 = tracker.record_run("ragas", mock_results_2)
            deltas = tracker.calculate_deltas(run2.run_id, "baseline")
            assert "aggregate" in deltas
            assert "faithfulness" in deltas["aggregate"]
            assert deltas["aggregate"]["faithfulness"] < 0  # Regression
            print("PASS: Test 4 - Calculate deltas")

            # Test 5: Check regressions
            alerts = tracker.check_regressions(run2.run_id)
            assert len(alerts) >= 1, f"Expected regression alert, got {alerts}"  # Should detect faithfulness regression
            print("PASS: Test 5 - Check regressions")

            # Test 6: Get history
            history = tracker.get_history("ragas")
            assert len(history) == 2
            print("PASS: Test 6 - Get history")

            # Test 7: Clear baseline
            tracker.clear_baseline("ragas")
            assert tracker.get_baseline("ragas") is None
            print("PASS: Test 7 - Clear baseline")

            # Test 8: Code graders support
            mock_code_grader_results = {
                "summary": {"pass_rate": 80, "passed": 8, "total_tests": 10},
                "results": [
                    {"test_id": "CBG_RAG_01", "passed": True, "query": "Q1"},
                    {"test_id": "CBG_RAG_02", "passed": False, "query": "Q2"},
                ]
            }
            code_run = tracker.record_run("code_graders_exact_match", mock_code_grader_results)
            assert code_run.aggregate_scores.get("pass_rate") == 0.8
            print("PASS: Test 8 - Code graders support")

            # Test 9: Persistence
            tracker2 = RegressionTracker(data_dir=temp_dir)
            history2 = tracker2.get_history("ragas")
            assert len(history2) == 2
            print("PASS: Test 9 - Persistence")

            print("\nSelf-test result: ALL PASSED")

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    RegressionTracker.run_self_tests()
