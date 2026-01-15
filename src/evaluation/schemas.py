"""
Unified result schemas for evaluation system.

This module defines dataclasses for structured evaluation results following
Anthropic's best practices for AI agent evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class GraderResult:
    """
    Result from a single grader (code-based or model-based).

    Attributes:
        grader_name: Name of the grader (e.g., "exact_match", "correctness")
        grader_type: Type of grader ("code" or "model")
        score: Normalized score from 0.0 to 1.0
        raw_score: Original score before normalization (may be different scale)
        passed: Binary pass/fail based on threshold
        partial_credit_awarded: Whether partial credit was given
        partial_credit_breakdown: Component scores for partial credit
        reasoning: Explanation of the score
        matched_items: Items that were correctly matched/found
        missed_items: Items that were missed or incorrect
        violations: List of constraint violations (for consistency checks, etc.)
        confidence: Confidence level (model-based graders only)
        escape_clause_triggered: Whether escape clause was activated (e.g., N/A)
        execution_time_ms: Time taken to execute grader
    """
    grader_name: str
    grader_type: str  # "code" | "model"
    score: float  # 0.0 to 1.0 (normalized)
    raw_score: Any  # Original score (may be different scale)
    passed: bool  # Binary pass/fail

    # Partial Credit
    partial_credit_awarded: bool = False
    partial_credit_breakdown: Dict[str, float] = field(default_factory=dict)

    # Explanation
    reasoning: str = ""
    matched_items: List[str] = field(default_factory=list)
    missed_items: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    confidence: Optional[float] = None  # Model-based graders only
    escape_clause_triggered: bool = False
    execution_time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "grader_name": self.grader_name,
            "grader_type": self.grader_type,
            "score": self.score,
            "raw_score": self.raw_score,
            "passed": self.passed,
            "partial_credit_awarded": self.partial_credit_awarded,
            "partial_credit_breakdown": self.partial_credit_breakdown,
            "reasoning": self.reasoning,
            "matched_items": self.matched_items,
            "missed_items": self.missed_items,
            "violations": self.violations,
            "confidence": self.confidence,
            "escape_clause_triggered": self.escape_clause_triggered,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class EvaluationResult:
    """
    Unified schema for evaluation of a single query.

    This dataclass combines results from all graders (code-based and model-based)
    for a single test query, along with metadata and performance metrics.

    Attributes:
        evaluation_id: Unique ID for this evaluation
        timestamp: ISO format timestamp
        agent_type: Which agent answered ("needle" | "summarization" | "manager")
        query_id: ID of the query (e.g., "Q_NEEDLE_01")
        query_text: The actual query text
        query_difficulty: Difficulty level ("easy" | "medium" | "hard")
        query_category: Category (e.g., "financial", "medical", "timeline")
        ground_truth: Expected correct answer
        expected_facts: List of facts that should be in the answer
        expected_sections: Document sections that should be retrieved
        answer: The agent's answer
        retrieved_context: Context chunks retrieved
        tool_trace: List of tools used during answer generation
        code_grader_scores: Results from all code-based graders
        code_grader_aggregate: Weighted average of code grader scores
        model_grader_scores: Results from all model-based graders
        model_grader_aggregate: Weighted average of model grader scores
        overall_score: Weighted combination of code + model graders
        passed: Overall pass/fail (threshold-based)
        partial_credit_details: Detailed breakdown of partial credit
        latency_ms: Time taken to generate answer
        cost_usd: Estimated cost for this query
        tokens_used: Token usage breakdown
    """
    # Metadata
    evaluation_id: str
    timestamp: str
    agent_type: str  # "needle" | "summarization" | "manager"

    # Query Info
    query_id: str
    query_text: str
    query_difficulty: str  # "easy" | "medium" | "hard"
    query_category: str  # "financial" | "medical" | "timeline" | etc.

    # Ground Truth
    ground_truth: str
    expected_facts: List[str]
    expected_sections: List[str] = field(default_factory=list)

    # System Response
    answer: str = ""
    retrieved_context: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Code-Based Grader Results (18 graders)
    code_grader_scores: Dict[str, GraderResult] = field(default_factory=dict)
    code_grader_aggregate: float = 0.0  # Weighted average (0-1)

    # Model-Based Grader Results (10 graders)
    model_grader_scores: Dict[str, GraderResult] = field(default_factory=dict)
    model_grader_aggregate: float = 0.0  # Weighted average (0-1)

    # Overall Scores
    overall_score: float = 0.0  # Weighted combination of code + model graders
    passed: bool = False  # Overall pass/fail (threshold-based)

    # Partial Credit Breakdown
    partial_credit_details: Dict[str, Any] = field(default_factory=dict)

    # Performance Metadata
    latency_ms: int = 0
    cost_usd: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)  # {prompt, completion, total}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "agent_type": self.agent_type,
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_difficulty": self.query_difficulty,
            "query_category": self.query_category,
            "ground_truth": self.ground_truth,
            "expected_facts": self.expected_facts,
            "expected_sections": self.expected_sections,
            "answer": self.answer,
            "retrieved_context": self.retrieved_context,
            "tool_trace": self.tool_trace,
            "code_grader_scores": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.code_grader_scores.items()},
            "code_grader_aggregate": self.code_grader_aggregate,
            "model_grader_scores": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.model_grader_scores.items()},
            "model_grader_aggregate": self.model_grader_aggregate,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "partial_credit_details": self.partial_credit_details,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tokens_used": self.tokens_used
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create EvaluationResult from dictionary (for cache deserialization)"""
        # Convert grader score dicts back to GraderResult objects if needed
        code_grader_scores = {}
        for k, v in data.get('code_grader_scores', {}).items():
            if isinstance(v, dict):
                code_grader_scores[k] = GraderResult(**v)
            else:
                code_grader_scores[k] = v

        model_grader_scores = {}
        for k, v in data.get('model_grader_scores', {}).items():
            if isinstance(v, dict):
                model_grader_scores[k] = GraderResult(**v)
            else:
                model_grader_scores[k] = v

        return cls(
            evaluation_id=data.get('evaluation_id', ''),
            timestamp=data.get('timestamp', ''),
            agent_type=data.get('agent_type', ''),
            query_id=data.get('query_id', ''),
            query_text=data.get('query_text', ''),
            query_difficulty=data.get('query_difficulty', ''),
            query_category=data.get('query_category', ''),
            ground_truth=data.get('ground_truth', ''),
            expected_facts=data.get('expected_facts', []),
            expected_sections=data.get('expected_sections', []),
            answer=data.get('answer', ''),
            retrieved_context=data.get('retrieved_context', ''),
            tool_trace=data.get('tool_trace', []),
            code_grader_scores=code_grader_scores,
            code_grader_aggregate=data.get('code_grader_aggregate', 0.0),
            model_grader_scores=model_grader_scores,
            model_grader_aggregate=data.get('model_grader_aggregate', 0.0),
            overall_score=data.get('overall_score', 0.0),
            passed=data.get('passed', False),
            partial_credit_details=data.get('partial_credit_details', {}),
            latency_ms=data.get('latency_ms', 0),
            cost_usd=data.get('cost_usd', 0.0),
            tokens_used=data.get('tokens_used', {})
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the evaluation"""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"{status} [{self.query_id}] {self.query_difficulty.upper()}\n"
            f"Score: {self.overall_score:.2f} "
            f"(Code: {self.code_grader_aggregate:.2f}, Model: {self.model_grader_aggregate:.2f})\n"
            f"Query: {self.query_text}\n"
            f"Answer: {self.answer[:100]}{'...' if len(self.answer) > 100 else ''}"
        )


@dataclass
class AggregateResults:
    """
    Aggregate results for a full evaluation run across all queries.

    Attributes:
        run_id: Unique ID for this evaluation run
        timestamp: ISO format timestamp
        individual_results: List of results for each query
        total_queries: Total number of queries evaluated
        code_grader_aggregates: Average score per code grader
        code_grader_pass_rates: Pass rate per code grader
        model_grader_aggregates: Average score per model grader
        model_grader_score_distributions: Score distribution per model grader
        overall_pass_rate: Overall pass rate across all queries
        overall_average_score: Average overall score
        easy_pass_rate: Pass rate for easy queries
        medium_pass_rate: Pass rate for medium queries
        hard_pass_rate: Pass rate for hard queries
        category_performance: Performance breakdown by category
        queries_with_partial_credit: Number of queries awarded partial credit
        avg_partial_credit_score: Average score when partial credit awarded
        common_failure_patterns: Analysis of common failure modes
        graders_with_low_scores: Graders with consistently low scores
        total_cost_usd: Total cost for evaluation run
        avg_latency_ms: Average latency per query
        total_tokens: Total tokens used
    """
    run_id: str
    timestamp: str

    # Query-Level Results
    individual_results: List[EvaluationResult] = field(default_factory=list)
    total_queries: int = 0

    # Code-Based Grader Aggregates
    code_grader_aggregates: Dict[str, float] = field(default_factory=dict)  # {grader_name: avg_score}
    code_grader_pass_rates: Dict[str, float] = field(default_factory=dict)  # {grader_name: pass_rate}

    # Model-Based Grader Aggregates
    model_grader_aggregates: Dict[str, float] = field(default_factory=dict)  # {metric_name: avg_score}
    model_grader_score_distributions: Dict[str, List[int]] = field(default_factory=dict)  # Histogram

    # Overall Performance
    overall_pass_rate: float = 0.0
    overall_average_score: float = 0.0

    # Breakdown by Difficulty
    easy_pass_rate: float = 0.0
    medium_pass_rate: float = 0.0
    hard_pass_rate: float = 0.0

    # Breakdown by Category
    category_performance: Dict[str, float] = field(default_factory=dict)  # {category: pass_rate}

    # Partial Credit Summary
    queries_with_partial_credit: int = 0
    avg_partial_credit_score: float = 0.0

    # Failure Analysis
    common_failure_patterns: List[Dict[str, Any]] = field(default_factory=list)
    graders_with_low_scores: List[str] = field(default_factory=list)

    # Cost & Performance
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "individual_results": [r.to_dict() for r in self.individual_results],
            "total_queries": self.total_queries,
            "code_grader_aggregates": self.code_grader_aggregates,
            "code_grader_pass_rates": self.code_grader_pass_rates,
            "model_grader_aggregates": self.model_grader_aggregates,
            "model_grader_score_distributions": self.model_grader_score_distributions,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_average_score": self.overall_average_score,
            "easy_pass_rate": self.easy_pass_rate,
            "medium_pass_rate": self.medium_pass_rate,
            "hard_pass_rate": self.hard_pass_rate,
            "category_performance": self.category_performance,
            "queries_with_partial_credit": self.queries_with_partial_credit,
            "avg_partial_credit_score": self.avg_partial_credit_score,
            "common_failure_patterns": self.common_failure_patterns,
            "graders_with_low_scores": self.graders_with_low_scores,
            "total_cost_usd": self.total_cost_usd,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens": self.total_tokens
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the aggregate results"""
        return (
            f"=== Evaluation Run: {self.run_id} ===\n"
            f"Total Queries: {self.total_queries}\n"
            f"Overall Pass Rate: {self.overall_pass_rate:.1%}\n"
            f"Overall Average Score: {self.overall_average_score:.2f}\n"
            f"\nBy Difficulty:\n"
            f"  Easy: {self.easy_pass_rate:.1%}\n"
            f"  Medium: {self.medium_pass_rate:.1%}\n"
            f"  Hard: {self.hard_pass_rate:.1%}\n"
            f"\nPartial Credit:\n"
            f"  Queries: {self.queries_with_partial_credit}\n"
            f"  Avg Score: {self.avg_partial_credit_score:.2f}\n"
            f"\nPerformance:\n"
            f"  Total Cost: ${self.total_cost_usd:.2f}\n"
            f"  Avg Latency: {self.avg_latency_ms:.0f}ms\n"
            f"  Total Tokens: {self.total_tokens:,}"
        )


@dataclass
class ComparisonResult:
    """
    Comparison between current run and baseline.

    Attributes:
        current_run_id: ID of current evaluation run
        baseline_run_id: ID of baseline run
        comparison_type: Type of comparison ("vs_baseline" | "vs_previous" | "vs_specific")
        overall_delta: Change in overall score
        code_grader_deltas: Delta per code grader
        model_grader_deltas: Delta per model grader
        pass_rate_delta: Change in pass rate
        pass_rate_by_difficulty: Pass rate change by difficulty
        improved_queries: List of query IDs that improved
        regressed_queries: List of query IDs that regressed
        unchanged_queries: List of query IDs unchanged
        regression_alerts: List of regression alerts
        critical_regressions: Count of critical regressions
        warning_regressions: Count of warning regressions
        improvements: List of improvements
        most_improved_grader: Grader with biggest improvement
        most_regressed_grader: Grader with biggest regression
    """
    current_run_id: str
    baseline_run_id: str
    comparison_type: str  # "vs_baseline" | "vs_previous" | "vs_specific"

    # Aggregate Deltas
    overall_delta: float = 0.0  # Change in overall score
    code_grader_deltas: Dict[str, float] = field(default_factory=dict)  # Per grader
    model_grader_deltas: Dict[str, float] = field(default_factory=dict)  # Per metric

    # Pass Rate Changes
    pass_rate_delta: float = 0.0
    pass_rate_by_difficulty: Dict[str, float] = field(default_factory=dict)  # {easy/medium/hard: delta}

    # Per-Query Comparisons
    improved_queries: List[str] = field(default_factory=list)  # Query IDs
    regressed_queries: List[str] = field(default_factory=list)
    unchanged_queries: List[str] = field(default_factory=list)

    # Regression Alerts
    regression_alerts: List[Dict[str, Any]] = field(default_factory=list)
    critical_regressions: int = 0
    warning_regressions: int = 0

    # Improvement Summary
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    most_improved_grader: str = ""
    most_regressed_grader: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "current_run_id": self.current_run_id,
            "baseline_run_id": self.baseline_run_id,
            "comparison_type": self.comparison_type,
            "overall_delta": self.overall_delta,
            "code_grader_deltas": self.code_grader_deltas,
            "model_grader_deltas": self.model_grader_deltas,
            "pass_rate_delta": self.pass_rate_delta,
            "pass_rate_by_difficulty": self.pass_rate_by_difficulty,
            "improved_queries": self.improved_queries,
            "regressed_queries": self.regressed_queries,
            "unchanged_queries": self.unchanged_queries,
            "regression_alerts": self.regression_alerts,
            "critical_regressions": self.critical_regressions,
            "warning_regressions": self.warning_regressions,
            "improvements": self.improvements,
            "most_improved_grader": self.most_improved_grader,
            "most_regressed_grader": self.most_regressed_grader
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the comparison"""
        delta_symbol = "↑" if self.overall_delta >= 0 else "↓"
        return (
            f"=== Comparison: {self.current_run_id} vs {self.baseline_run_id} ===\n"
            f"Overall Delta: {delta_symbol} {abs(self.overall_delta):.3f}\n"
            f"Pass Rate Delta: {self.pass_rate_delta:+.1%}\n"
            f"\nQuery Changes:\n"
            f"  Improved: {len(self.improved_queries)}\n"
            f"  Regressed: {len(self.regressed_queries)}\n"
            f"  Unchanged: {len(self.unchanged_queries)}\n"
            f"\nRegressions:\n"
            f"  Critical: {self.critical_regressions}\n"
            f"  Warning: {self.warning_regressions}\n"
            f"\nGrader Performance:\n"
            f"  Most Improved: {self.most_improved_grader}\n"
            f"  Most Regressed: {self.most_regressed_grader}"
        )


def create_evaluation_result(
    query_id: str,
    query_text: str,
    ground_truth: str,
    expected_facts: List[str],
    difficulty: str = "medium",
    category: str = "general",
    agent_type: str = "needle"
) -> EvaluationResult:
    """
    Factory function to create a new EvaluationResult with default values.

    Args:
        query_id: ID of the query (e.g., "Q_NEEDLE_01")
        query_text: The actual query text
        ground_truth: Expected correct answer
        expected_facts: List of facts that should be in answer
        difficulty: Difficulty level ("easy", "medium", "hard")
        category: Category (e.g., "financial", "medical")
        agent_type: Agent type ("needle", "summarization", "manager")

    Returns:
        EvaluationResult with metadata populated
    """
    return EvaluationResult(
        evaluation_id=f"{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        agent_type=agent_type,
        query_id=query_id,
        query_text=query_text,
        query_difficulty=difficulty,
        query_category=category,
        ground_truth=ground_truth,
        expected_facts=expected_facts
    )
