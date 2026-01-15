"""
Unified Evaluation Orchestrator

Coordinates evaluation using:
- 18 code-based graders (with partial credit)
- 10 model-based graders (with rubrics)
- 40 test queries (20 needle + 20 summary)
- Comprehensive result schemas

Follows Anthropic's best practices:
- Evaluate outcomes over processes
- Implement partial credit
- Dimensional separation
- Exploitation resistance
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from src.evaluation.schemas import EvaluationResult, GraderResult, AggregateResults
from src.evaluation.code_graders import CodeBasedGraders
from src.evaluation.judge import LLMJudge
from src.evaluation.test_queries_expanded import ExpandedTestSuite

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Unified orchestrator for running comprehensive evaluations

    Integrates:
    - Code-based graders (18 total)
    - Model-based graders (10 total)
    - Expanded test suite (40 queries)
    - Comprehensive result tracking
    """

    def __init__(
        self,
        system,
        judge: Optional[LLMJudge] = None,
        output_dir: str = "./evaluation_results",
        enable_model_graders: bool = True,
        use_cache: bool = True
    ):
        """
        Initialize evaluation orchestrator

        Args:
            system: InsuranceClaimSystem instance
            judge: LLMJudge instance (created if None)
            output_dir: Directory for saving results
            enable_model_graders: Whether to run model-based graders (requires API key)
            use_cache: Whether to use cached results (saves API costs)
        """
        self.system = system
        self.enable_model_graders = enable_model_graders
        self.use_cache = use_cache

        # Initialize cache
        from src.evaluation.cache_manager import EvaluationCache
        self.cache = EvaluationCache()
        logger.info(f"Cache initialized: {self.cache.get_cache_stats()['total_cached']} results cached")

        # Initialize graders
        self.code_graders = CodeBasedGraders()

        if enable_model_graders:
            self.judge = judge if judge else LLMJudge(temperature=0)
            logger.info("Model-based graders enabled (using Claude)")
        else:
            self.judge = None
            logger.info("Model-based graders disabled")

        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("EvaluationOrchestrator initialized")

    def run_full_evaluation(
        self,
        query_filter: Optional[Dict[str, Any]] = None,
        save_results: bool = True
    ) -> AggregateResults:
        """
        Run full evaluation on all or filtered test queries

        Args:
            query_filter: Optional filter dict (e.g., {"difficulty": "easy"})
            save_results: Whether to save results to disk

        Returns:
            AggregateResults with comprehensive metrics
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE EVALUATION")
        logger.info("=" * 80)

        # Get test queries
        all_queries = ExpandedTestSuite.get_all_queries()

        # Apply filter if provided
        if query_filter:
            queries = [q for q in all_queries if self._matches_filter(q, query_filter)]
            logger.info(f"Filtered to {len(queries)}/{len(all_queries)} queries")
        else:
            queries = all_queries
            logger.info(f"Evaluating all {len(queries)} queries")

        # Track results
        evaluation_results = []
        start_time = time.time()

        # Evaluate each query
        for i, query_spec in enumerate(queries, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Query {i}/{len(queries)}: {query_spec['id']} ({query_spec['difficulty']})")
            logger.info(f"{'=' * 80}")

            try:
                result = self.evaluate_single_query(query_spec)
                evaluation_results.append(result)

                # Print summary
                self._print_query_summary(result)

            except Exception as e:
                logger.error(f"Error evaluating {query_spec['id']}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                    timestamp=datetime.now().isoformat(),
                    agent_type=query_spec.get('expected_agent', 'unknown'),
                    query_id=query_spec['id'],
                    query_text=query_spec['query'],
                    query_difficulty=query_spec.get('difficulty', 'unknown'),
                    query_category=query_spec.get('category', 'unknown'),
                    ground_truth=query_spec.get('ground_truth', ''),
                    expected_facts=query_spec.get('expected_facts', []),
                    answer=f"ERROR: {str(e)}",
                    overall_score=0.0,
                    passed=False
                )
                evaluation_results.append(error_result)

        # Calculate aggregate results
        elapsed_time = time.time() - start_time
        aggregate = self._calculate_aggregate_results(evaluation_results, elapsed_time)

        # Save if requested
        if save_results:
            self._save_results(aggregate, evaluation_results)

        # Print final summary
        self._print_final_summary(aggregate)

        return aggregate

    def evaluate_single_query(self, query_spec: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single test query with all applicable graders

        Args:
            query_spec: Query specification from test suite

        Returns:
            Comprehensive EvaluationResult
        """
        query_text = query_spec['query']
        query_id = query_spec['id']
        query_type = query_spec.get('type', 'needle')

        # Check cache first
        if self.use_cache:
            cached_result = self.cache.get_cached_result(query_id, query_type)
            if cached_result is not None:
                logger.info(f"✅ Using cached result for {query_id}")
                return EvaluationResult.from_dict(cached_result)

        logger.info(f"🔄 Evaluating query: {query_text}")

        # Run query through system
        start_time = time.time()
        system_response = self.system.query(query_text, use_manager=True)
        latency_ms = int((time.time() - start_time) * 1000)

        answer = system_response.get("output", "")
        success = system_response.get("success", False)

        # Extract retrieved context and tool trace
        retrieved_context = self._extract_context(system_response)
        tool_trace = self._extract_tool_trace(system_response)

        # Initialize result
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query_id}"
        result = EvaluationResult(
            evaluation_id=eval_id,
            timestamp=datetime.now().isoformat(),
            agent_type=query_spec.get('expected_agent', 'unknown'),
            query_id=query_id,
            query_text=query_text,
            query_difficulty=query_spec.get('difficulty', 'unknown'),
            query_category=query_spec.get('category', 'unknown'),
            ground_truth=query_spec.get('ground_truth', ''),
            expected_facts=query_spec.get('expected_facts', []),
            expected_sections=query_spec.get('expected_sections', []),
            answer=answer,
            retrieved_context=retrieved_context,
            tool_trace=tool_trace,
            latency_ms=latency_ms
        )

        # Apply code-based graders
        logger.info("Applying code-based graders...")
        result.code_grader_scores = self._apply_code_graders(
            query_spec, answer, retrieved_context, tool_trace
        )

        # Calculate code grader aggregate
        code_scores = [
            g.score for g in result.code_grader_scores.values()
            if g.score is not None and isinstance(g.score, (int, float))
        ]
        result.code_grader_aggregate = float(sum(code_scores) / len(code_scores)) if code_scores else 0.0

        # Apply model-based graders
        if self.enable_model_graders and self.judge:
            logger.info("Applying model-based graders...")
            result.model_grader_scores = self._apply_model_graders(
                query_spec, answer, retrieved_context, tool_trace
            )

            # Calculate model grader aggregate
            model_scores = [
                g.score for g in result.model_grader_scores.values()
                if g.score is not None and isinstance(g.score, (int, float))
            ]
            result.model_grader_aggregate = float(sum(model_scores) / len(model_scores)) if model_scores else 0.0
        else:
            result.model_grader_scores = {}
            result.model_grader_aggregate = 0.0

        # Calculate overall score (weighted: 60% code, 40% model)
        # Ensure we have valid numbers
        code_agg = result.code_grader_aggregate if result.code_grader_aggregate is not None else 0.0
        model_agg = result.model_grader_aggregate if result.model_grader_aggregate is not None else 0.0

        if self.enable_model_graders:
            result.overall_score = float(0.6 * code_agg + 0.4 * model_agg)
        else:
            result.overall_score = float(code_agg)

        # Determine pass/fail - ensure overall_score is not None
        result.passed = bool(result.overall_score is not None and result.overall_score >= 0.7)

        # Cache the result
        if self.use_cache:
            self.cache.cache_result(
                query_id=query_id,
                result=result.to_dict(),
                query_type=query_type,
                model="gpt-4o-mini"  # TODO: Get actual model from system
            )
            logger.info(f"💾 Cached result for {query_id}")

        return result

    def _apply_code_graders(
        self,
        query_spec: Dict[str, Any],
        answer: str,
        retrieved_context: str,
        tool_trace: List[Dict]
    ) -> Dict[str, GraderResult]:
        """Apply all relevant code-based graders"""
        graders_to_run = query_spec.get('graders', [])
        results = {}

        # Map grader names to methods
        grader_map = {
            'exact_match': lambda: self.code_graders.exact_match_grade(
                answer, query_spec.get('ground_truth', '')
            ),
            'regex': lambda: self.code_graders.regex_grade(
                answer, query_spec.get('regex_pattern', r'.*')
            ),
            'numerical_validation': lambda: self.code_graders.numerical_validation_grade(
                answer, query_spec.get('expected_number')
            ),
            'consistency_check': lambda: self.code_graders.consistency_check_grade(
                answer, query_spec.get('consistency_rules', [])
            ),
            'key_fact_coverage': lambda: self.code_graders.key_fact_coverage_grade(
                answer, query_spec.get('expected_facts', [])
            ),
            'fuzzy_match': lambda: self.code_graders.fuzzy_match_grade(
                answer, query_spec.get('ground_truth', '')
            ),
            'multi_fact_extraction': lambda: self.code_graders.multi_fact_extraction_grade(
                answer, query_spec.get('required_facts', [])
            ),
            'tool_usage_correctness': lambda: self.code_graders.tool_usage_correctness_grade(
                query_spec['query'], answer, query_spec.get('expected_outcome'), tool_trace
            ),
            'timeline_ordering': lambda: self.code_graders.timeline_ordering_grade(
                answer, query_spec.get('expected_events', [])
            ),
            'financial_constraint': lambda: self.code_graders.financial_constraint_grade(
                answer, query_spec.get('financial_constraints', [])
            ),
            'entity_relationship': lambda: self.code_graders.entity_relationship_grade(
                answer, query_spec.get('entity_relationships', [])
            ),
            'missing_information_detection': lambda: self.code_graders.missing_information_detection_grade(
                query_spec['query'], answer, query_spec.get('info_availability', {})
            ),
            'cross_section_inference': lambda: self.code_graders.cross_section_inference_grade(
                answer, query_spec.get('required_sections', []),
                query_spec.get('inference_check')
            ),
            'ambiguity_resolution': lambda: self.code_graders.ambiguity_resolution_grade(
                query_spec['query'], answer, query_spec.get('expected_behavior', {})
            ),
            'date_arithmetic': lambda: self.code_graders.date_arithmetic_grade(
                query_spec['query'], answer, query_spec.get('expected_calculation')
            ),
            'confidence_calibration': lambda: self.code_graders.confidence_calibration_grade(
                query_spec['query'], answer, query_spec.get('expected_confidence', 'high')
            ),
            'retrieval_coverage': lambda: self.code_graders.retrieval_coverage_grade(
                query_spec['query'], [retrieved_context],
                query_spec.get('required_information', [])
            ),
            'answer_completeness': lambda: self.code_graders.answer_completeness_grade(
                query_spec['query'], answer, query_spec.get('required_components', [])
            ),
        }

        # Run each requested grader
        for grader_name in graders_to_run:
            if grader_name in grader_map:
                try:
                    grader_result = grader_map[grader_name]()

                    # Ensure grader_result is a dict with a score
                    if not isinstance(grader_result, dict):
                        logger.warning(f"Grader '{grader_name}' returned non-dict: {type(grader_result)}")
                        grader_result = {'score': 0.0, 'passed': False, 'reasoning': 'Invalid grader result type'}

                    if 'score' not in grader_result or grader_result['score'] is None:
                        logger.warning(f"Grader '{grader_name}' returned None score")
                        grader_result['score'] = 0.0
                        grader_result['passed'] = False
                        grader_result['reasoning'] = grader_result.get('reasoning', 'No score returned')

                    results[grader_name] = self._convert_to_grader_result(
                        grader_name, 'code', grader_result
                    )
                except Exception as e:
                    logger.error(f"Error in code grader '{grader_name}': {e}", exc_info=True)
                    results[grader_name] = GraderResult(
                        grader_name=grader_name,
                        grader_type='code',
                        score=0.0,
                        raw_score=0.0,
                        passed=False,
                        reasoning=f"Error: {str(e)}"
                    )
            else:
                logger.warning(f"Grader '{grader_name}' not found in code graders (might be model-based)")

        return results

    def _apply_model_graders(
        self,
        query_spec: Dict[str, Any],
        answer: str,
        retrieved_context: str,
        tool_trace: List[Dict]
    ) -> Dict[str, GraderResult]:
        """Apply all relevant model-based graders"""
        results = {}

        # Always run core model graders
        model_graders = {
            'correctness': lambda: self.judge.evaluate_answer_correctness(
                query_spec['query'], answer, query_spec.get('ground_truth', '')
            ),
            'faithfulness': lambda: self.judge.evaluate_faithfulness(answer, retrieved_context),
            'helpfulness': lambda: self.judge.evaluate_helpfulness(
                query_spec['query'], answer, retrieved_context
            ),
            'coherence': lambda: self.judge.evaluate_coherence(answer),
            'conciseness': lambda: self.judge.evaluate_conciseness(
                query_spec['query'], answer, query_spec.get('type', 'needle')
            ),
            'relevancy': lambda: self.judge.evaluate_context_relevancy(
                query_spec['query'], retrieved_context
            ),
            'recall': lambda: self.judge.evaluate_context_recall(
                query_spec.get('ground_truth', ''), retrieved_context
            ),
            'citation_quality': lambda: self.judge.evaluate_citation_quality(answer, retrieved_context),
            'query_understanding': lambda: self.judge.evaluate_query_understanding(
                query_spec['query'], answer
            ),
            'efficiency': lambda: self.judge.evaluate_efficiency(
                query_spec['query'], tool_trace
            ),
        }

        # Run each model grader
        for grader_name, grader_func in model_graders.items():
            try:
                grader_result = grader_func()
                results[grader_name] = self._convert_to_grader_result(
                    grader_name, 'model', grader_result
                )
            except Exception as e:
                logger.error(f"Error in model grader '{grader_name}': {e}")
                results[grader_name] = GraderResult(
                    grader_name=grader_name,
                    grader_type='model',
                    score=0.0,
                    raw_score=0.0,
                    passed=False,
                    reasoning=f"Error: {str(e)}"
                )

        return results

    def _convert_to_grader_result(
        self,
        grader_name: str,
        grader_type: str,
        result_dict: Dict[str, Any]
    ) -> GraderResult:
        """Convert grader output dict to GraderResult object"""
        # Get raw score
        raw_score = result_dict.get('score', 0)

        # Normalize score to 0-1 range
        score = raw_score

        # Handle N/A scores from escape clauses
        if score == "N/A":
            score = None
        elif isinstance(score, (int, float)):
            # Model graders use 1-5 scale
            if grader_type == 'model' and score > 1:
                score = score / 5.0
        elif score is None:
            # Explicitly set None scores to 0.0 for safety
            score = 0.0
            raw_score = 0.0

        # Determine passed status safely
        if score is not None and isinstance(score, (int, float)):
            passed = bool(score >= 0.7)
        else:
            passed = result_dict.get('passed', False)

        return GraderResult(
            grader_name=grader_name,
            grader_type=grader_type,
            score=score if score is not None else 0.0,
            raw_score=raw_score if raw_score is not None else 0.0,
            passed=passed,
            partial_credit_awarded=result_dict.get('partial_credit_awarded', False),
            partial_credit_breakdown=result_dict.get('partial_credit_breakdown', {}),
            reasoning=result_dict.get('reasoning', result_dict.get('explanation', '')),
            matched_items=result_dict.get('matched_items', result_dict.get('found_facts', [])),
            missed_items=result_dict.get('missed_items', result_dict.get('missing_facts', [])),
            confidence=result_dict.get('confidence')
        )

    def _extract_context(self, system_response: Dict[str, Any]) -> str:
        """Extract retrieved context from system response"""
        context_parts = []

        # Try to extract from messages (LangGraph format)
        messages = system_response.get('messages', [])
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'tool':
                if hasattr(msg, 'content'):
                    context_parts.append(str(msg.content))

        # Try intermediate steps (legacy format)
        if system_response.get('intermediate_steps'):
            for step in system_response['intermediate_steps']:
                if len(step) >= 2:
                    context_parts.append(str(step[1]))

        return '\n\n'.join(context_parts) if context_parts else ""

    def _extract_tool_trace(self, system_response: Dict[str, Any]) -> List[Dict]:
        """Extract tool usage trace from system response"""
        trace = []

        # Extract from messages
        messages = system_response.get('messages', [])
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    trace.append({
                        'tool': tool_call.get('name', 'unknown'),
                        'input': tool_call.get('args', {})
                    })

        return trace

    def _calculate_aggregate_results(
        self,
        evaluation_results: List[EvaluationResult],
        elapsed_time: float
    ) -> AggregateResults:
        """Calculate aggregate metrics across all evaluation results"""

        # Overall metrics
        total_queries = len(evaluation_results)
        passed_queries = sum(1 for r in evaluation_results if r.passed)
        overall_pass_rate = passed_queries / total_queries if total_queries > 0 else 0.0

        # Average scores
        overall_scores = [r.overall_score for r in evaluation_results]
        overall_average_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        # By difficulty
        difficulty_stats = {}
        for difficulty in ['easy', 'medium', 'hard']:
            diff_results = [r for r in evaluation_results if r.query_difficulty == difficulty]
            if diff_results:
                difficulty_stats[f"{difficulty}_pass_rate"] = sum(
                    1 for r in diff_results if r.passed
                ) / len(diff_results)
                difficulty_stats[f"{difficulty}_avg_score"] = sum(
                    r.overall_score for r in diff_results
                ) / len(diff_results)

        # By category
        category_performance = {}
        categories = set(r.query_category for r in evaluation_results)
        for category in categories:
            cat_results = [r for r in evaluation_results if r.query_category == category]
            if cat_results:
                category_performance[category] = sum(
                    r.overall_score for r in cat_results
                ) / len(cat_results)

        # Grader aggregates
        code_grader_aggregates = self._aggregate_grader_scores(
            evaluation_results, 'code'
        )
        model_grader_aggregates = self._aggregate_grader_scores(
            evaluation_results, 'model'
        )

        # Cost and latency
        total_latency = sum(r.latency_ms for r in evaluation_results)
        avg_latency_ms = total_latency / total_queries if total_queries > 0 else 0

        # Estimated cost (rough estimate)
        # ~$0.01 per model grader call * 10 graders * queries
        estimated_cost = (total_queries * 10 * 0.01) if self.enable_model_graders else 0.0

        # Calculate partial credit stats
        queries_with_partial = sum(
            1 for r in evaluation_results
            if any(g.partial_credit_awarded for g in r.code_grader_scores.values())
        )
        partial_scores = [
            r.overall_score for r in evaluation_results
            if any(g.partial_credit_awarded for g in r.code_grader_scores.values())
        ]
        avg_partial_score = sum(partial_scores) / len(partial_scores) if partial_scores else 0.0

        return AggregateResults(
            run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            individual_results=evaluation_results,
            total_queries=total_queries,
            overall_pass_rate=overall_pass_rate,
            overall_average_score=overall_average_score,
            easy_pass_rate=difficulty_stats.get('easy_pass_rate', 0.0),
            medium_pass_rate=difficulty_stats.get('medium_pass_rate', 0.0),
            hard_pass_rate=difficulty_stats.get('hard_pass_rate', 0.0),
            category_performance=category_performance,
            code_grader_aggregates=code_grader_aggregates,
            model_grader_aggregates=model_grader_aggregates,
            queries_with_partial_credit=queries_with_partial,
            avg_partial_credit_score=avg_partial_score,
            total_cost_usd=estimated_cost,
            avg_latency_ms=avg_latency_ms
        )

    def _aggregate_grader_scores(
        self,
        evaluation_results: List[EvaluationResult],
        grader_type: str
    ) -> Dict[str, float]:
        """Calculate average score for each grader across all evaluations"""
        grader_scores = {}

        for result in evaluation_results:
            scores_dict = (result.code_grader_scores if grader_type == 'code'
                          else result.model_grader_scores)

            for grader_name, grader_result in scores_dict.items():
                if grader_result.score is not None:
                    if grader_name not in grader_scores:
                        grader_scores[grader_name] = []
                    grader_scores[grader_name].append(grader_result.score)

        # Calculate averages
        return {
            name: sum(scores) / len(scores)
            for name, scores in grader_scores.items()
            if scores
        }

    def _matches_filter(self, query_spec: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if query matches filter criteria"""
        for key, value in filter_dict.items():
            if query_spec.get(key) != value:
                return False
        return True

    def _save_results(
        self,
        aggregate: AggregateResults,
        evaluation_results: List[EvaluationResult]
    ):
        """Save results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save aggregate results
        agg_file = self.output_dir / f"aggregate_{timestamp}.json"
        with open(agg_file, 'w') as f:
            json.dump(aggregate.to_dict(), f, indent=2)
        logger.info(f"Saved aggregate results to: {agg_file}")

        # Save detailed results
        detailed_file = self.output_dir / f"detailed_{timestamp}.json"
        detailed_data = {
            'aggregate': aggregate.to_dict(),
            'evaluations': [r.to_dict() for r in evaluation_results]
        }
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        logger.info(f"Saved detailed results to: {detailed_file}")

    def _print_query_summary(self, result: EvaluationResult):
        """Print summary for a single query evaluation"""
        print(f"\n{'-' * 80}")
        print(f"Query ID: {result.query_id}")
        print(f"Difficulty: {result.query_difficulty} | Category: {result.query_category}")
        print(f"{'-' * 80}")
        print(f"Code Graders:  {result.code_grader_aggregate:.2f}")
        print(f"Model Graders: {result.model_grader_aggregate:.2f}")
        print(f"Overall Score: {result.overall_score:.2f} | {'PASS' if result.passed else 'FAIL'}")

    def _print_final_summary(self, aggregate: AggregateResults):
        """Print final evaluation summary"""
        print("\n" + "=" * 80)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 80)
        passed_count = sum(1 for r in aggregate.individual_results if r.passed)
        print(f"\nTotal Queries: {aggregate.total_queries}")
        print(f"Passed: {passed_count} ({aggregate.overall_pass_rate*100:.1f}%)")
        print(f"Overall Average Score: {aggregate.overall_average_score:.2f}")
        print(f"\n{'-' * 80}")
        print("BY DIFFICULTY")
        print(f"{'-' * 80}")
        print(f"Easy:   Pass Rate: {aggregate.easy_pass_rate*100:5.1f}%")
        print(f"Medium: Pass Rate: {aggregate.medium_pass_rate*100:5.1f}%")
        print(f"Hard:   Pass Rate: {aggregate.hard_pass_rate*100:5.1f}%")
        print(f"\n{'-' * 80}")
        print(f"Partial Credit: {aggregate.queries_with_partial_credit} queries | Avg: {aggregate.avg_partial_credit_score:.2f}")
        print(f"Avg Latency: {aggregate.avg_latency_ms:.0f}ms | Cost: ${aggregate.total_cost_usd:.2f}")
        print("=" * 80 + "\n")
