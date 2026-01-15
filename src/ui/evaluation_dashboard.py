"""
Comprehensive Evaluation Dashboard for Streamlit UI

Displays results from all 28 graders (18 code-based + 10 model-based)
with visualizations, partial credit breakdowns, and detailed metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional


def render_grader_result_card(grader_result: Dict[str, Any], grader_name: str, grader_type: str = "", key_prefix: str = ""):
    """
    Render a single grader result as a card with details.

    Args:
        grader_result: Result dictionary from a grader
        grader_name: Name of the grader
        grader_type: Type prefix for unique keys (e.g., "code", "model")
        key_prefix: Unique prefix to avoid key collisions when rendering multiple dashboards
    """
    passed = grader_result.get('passed', False)
    score = grader_result.get('score', 0)

    # Color based on pass/fail
    if passed:
        border_color = "#28a745"  # Green
        icon = "✅"
    else:
        border_color = "#dc3545"  # Red
        icon = "❌"

    # Determine if partial credit was awarded
    partial_credit = score > 0 and score < 1 and not passed
    if partial_credit:
        border_color = "#ffc107"  # Yellow
        icon = "⚠️"

    with st.container():
        st.markdown(f"""
        <div style="border-left: 4px solid {border_color}; padding: 10px; margin: 10px 0; background-color: rgba(255,255,255,0.05);">
            <h4 style="margin: 0 0 10px 0;">{icon} {grader_name.replace('_', ' ').title()}</h4>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Score", f"{score:.3f}" if isinstance(score, float) else str(score))
            st.metric("Status", "PASS" if passed else ("PARTIAL" if partial_credit else "FAIL"))

        with col2:
            # Show details
            details = grader_result.get('details', grader_result.get('reasoning', 'No details available'))
            unique_key = f"{key_prefix}{grader_type}_{grader_name}_details" if grader_type else f"{key_prefix}{grader_name}_details"
            st.text_area("Details", details, height=80, key=unique_key, disabled=True)

        # Partial credit breakdown if available
        if 'partial_credit_breakdown' in grader_result and grader_result['partial_credit_breakdown']:
            with st.expander("📊 Partial Credit Breakdown"):
                breakdown = grader_result['partial_credit_breakdown']
                for component, component_score in breakdown.items():
                    st.write(f"- **{component}**: {component_score}")

        # Matched/missed items if available
        if 'matched_items' in grader_result or 'facts_found' in grader_result:
            with st.expander("🔍 Found/Missing Items"):
                matched = grader_result.get('matched_items', grader_result.get('facts_found', []))
                missed = grader_result.get('missed_items', grader_result.get('facts_missing', []))

                if matched:
                    st.success(f"**Found ({len(matched)})**: {', '.join(matched)}")
                if missed:
                    st.error(f"**Missing ({len(missed)})**: {', '.join(missed)}")


def render_code_graders_section(code_grader_scores: Dict[str, Dict[str, Any]], key_prefix: str = ""):
    """
    Render all code-based grader results.

    Args:
        code_grader_scores: Dictionary of code grader results
        key_prefix: Unique prefix to avoid key collisions
    """
    st.markdown("### 📋 Code-Based Graders (18 Total)")

    if not code_grader_scores:
        st.info("No code-based grader results available. Run an evaluation first.")
        return

    # Calculate summary statistics
    total_graders = len(code_grader_scores)
    passed_graders = sum(1 for r in code_grader_scores.values() if r.get('passed', False))
    avg_score = sum(r.get('score', 0) for r in code_grader_scores.values()) / total_graders if total_graders > 0 else 0

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Graders", total_graders)
    with col2:
        st.metric("Passed", f"{passed_graders}/{total_graders}")
    with col3:
        st.metric("Average Score", f"{avg_score:.3f}")

    # Group graders by category
    categories = {
        "Fact Extraction": ["multi_fact_extraction", "exact_match", "regex"],
        "Validation": ["numerical_validation", "tool_usage_correctness", "date_arithmetic"],
        "Relationships": ["entity_relationship", "timeline_ordering", "financial_constraint"],
        "Quality Checks": ["missing_information_detection", "cross_section_inference", "ambiguity_resolution"],
        "Completeness": ["retrieval_coverage", "answer_completeness", "key_fact_coverage"],
        "Consistency": ["consistency_check", "confidence_calibration", "fuzzy_match"]
    }

    # Render by category
    for category, grader_names in categories.items():
        category_results = {name: code_grader_scores[name] for name in grader_names if name in code_grader_scores}

        if category_results:
            with st.expander(f"**{category}** ({len(category_results)} graders)", expanded=False):
                for grader_name, result in category_results.items():
                    render_grader_result_card(result, grader_name, grader_type="code", key_prefix=key_prefix)


def render_model_graders_section(model_grader_scores: Dict[str, Dict[str, Any]], key_prefix: str = ""):
    """
    Render all model-based grader results.

    Args:
        model_grader_scores: Dictionary of model grader results
        key_prefix: Unique prefix to avoid key collisions
    """
    st.markdown("### 🤖 Model-Based Graders (10 Total)")

    if not model_grader_scores:
        st.info("No model-based grader results available. Run an evaluation first.")
        return

    # Calculate summary statistics (handle N/A scores)
    total_graders = len(model_grader_scores)
    numeric_scores = [r.get('score', 0) for r in model_grader_scores.values() if isinstance(r.get('score'), (int, float))]
    avg_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0

    # Convert to 0-1 scale (model graders use 1-5)
    normalized_scores = [s / 5.0 for s in numeric_scores]
    normalized_avg = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Graders", total_graders)
    with col2:
        st.metric("Avg Score (1-5)", f"{avg_score:.2f}")
    with col3:
        st.metric("Normalized (0-1)", f"{normalized_avg:.3f}")

    # Score distribution chart
    if numeric_scores:
        fig = go.Figure(data=[go.Bar(
            x=list(model_grader_scores.keys()),
            y=[r.get('score', 0) for r in model_grader_scores.values()],
            marker_color=['green' if s >= 4 else 'orange' if s >= 3 else 'red'
                         for s in [r.get('score', 0) for r in model_grader_scores.values()]]
        )])
        fig.update_layout(
            title="Model Grader Scores (1-5 scale)",
            xaxis_title="Grader",
            yaxis_title="Score",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}model_grader_scores_chart")

    # Group model graders
    categories = {
        "Accuracy": ["correctness", "faithfulness", "query_understanding"],
        "Quality": ["helpfulness", "coherence", "conciseness"],
        "Sources": ["relevancy", "recall", "citation_quality"],
        "Efficiency": ["efficiency"]
    }

    # Render by category
    for category, grader_names in categories.items():
        category_results = {name: model_grader_scores[name] for name in grader_names if name in model_grader_scores}

        if category_results:
            with st.expander(f"**{category}** ({len(category_results)} metrics)", expanded=False):
                for grader_name, result in category_results.items():
                    render_grader_result_card(result, grader_name, grader_type="model", key_prefix=key_prefix)


def render_overall_summary(
    code_grader_aggregate: float,
    model_grader_aggregate: float,
    overall_score: float,
    passed: bool,
    key_prefix: str = ""
):
    """
    Render overall evaluation summary with combined scores.

    Args:
        code_grader_aggregate: Aggregate code grader score (0-1)
        model_grader_aggregate: Aggregate model grader score (0-1)
        overall_score: Overall weighted score (0-1)
        passed: Whether evaluation passed overall
    """
    st.markdown("## 📊 Overall Evaluation Summary")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Score",
            f"{overall_score:.3f}",
            delta="PASS" if passed else "FAIL",
            delta_color="normal" if passed else "inverse"
        )

    with col2:
        st.metric("Code Graders", f"{code_grader_aggregate:.3f}")

    with col3:
        st.metric("Model Graders", f"{model_grader_aggregate:.3f}")

    with col4:
        status_color = "🟢" if passed else "🔴"
        st.markdown(f"### {status_color} {'PASSED' if passed else 'FAILED'}")

    # Score breakdown chart
    fig = go.Figure(data=[
        go.Bar(
            name='Code Graders',
            x=['Aggregate Scores'],
            y=[code_grader_aggregate],
            marker_color='#3498db'
        ),
        go.Bar(
            name='Model Graders',
            x=['Aggregate Scores'],
            y=[model_grader_aggregate],
            marker_color='#9b59b6'
        ),
        go.Bar(
            name='Overall (Weighted)',
            x=['Aggregate Scores'],
            y=[overall_score],
            marker_color='#2ecc71' if passed else '#e74c3c'
        )
    ])

    fig.update_layout(
        title="Score Breakdown",
        yaxis_title="Score (0-1)",
        barmode='group',
        height=300
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}overall_summary_chart")


def render_query_metadata(evaluation_result: Dict[str, Any], key_prefix: str = ""):
    """
    Render query metadata and context.

    Args:
        evaluation_result: Full evaluation result dictionary
        key_prefix: Unique prefix to avoid key collisions
    """
    with st.expander("📝 Query Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Query ID**: {evaluation_result.get('query_id', 'N/A')}")
            st.write(f"**Difficulty**: {evaluation_result.get('query_difficulty', 'N/A')}")
            st.write(f"**Category**: {evaluation_result.get('query_category', 'N/A')}")

        with col2:
            st.write(f"**Agent Type**: {evaluation_result.get('agent_type', 'N/A')}")
            st.write(f"**Latency**: {evaluation_result.get('latency_ms', 0)}ms")
            st.write(f"**Cost**: ${evaluation_result.get('cost_usd', 0.0):.4f}")

        st.text_area("Query", evaluation_result.get('query_text', ''), height=80, disabled=True, key=f"{key_prefix}query_text")
        st.text_area("Answer", evaluation_result.get('answer', ''), height=150, disabled=True, key=f"{key_prefix}answer_text")
        st.text_area("Ground Truth", evaluation_result.get('ground_truth', ''), height=80, disabled=True, key=f"{key_prefix}ground_truth_text")


def render_evaluation_dashboard(evaluation_result: Dict[str, Any], key_prefix: str = ""):
    """
    Main entry point: Render complete evaluation dashboard.

    Args:
        evaluation_result: Full evaluation result from EvaluationResult.to_dict()
        key_prefix: Unique prefix to avoid key collisions when rendering multiple dashboards
    """
    st.title("🎯 Comprehensive Evaluation Dashboard")

    # Overall summary
    render_overall_summary(
        code_grader_aggregate=evaluation_result.get('code_grader_aggregate', 0.0),
        model_grader_aggregate=evaluation_result.get('model_grader_aggregate', 0.0),
        overall_score=evaluation_result.get('overall_score', 0.0),
        passed=evaluation_result.get('passed', False),
        key_prefix=key_prefix
    )

    st.divider()

    # Query metadata
    render_query_metadata(evaluation_result, key_prefix=key_prefix)

    st.divider()

    # Code graders section
    code_grader_scores = evaluation_result.get('code_grader_scores', {})
    render_code_graders_section(code_grader_scores, key_prefix=key_prefix)

    st.divider()

    # Model graders section
    model_grader_scores = evaluation_result.get('model_grader_scores', {})
    render_model_graders_section(model_grader_scores, key_prefix=key_prefix)

    # Partial credit summary
    if evaluation_result.get('partial_credit_details'):
        st.divider()
        with st.expander("💯 Partial Credit Summary"):
            st.json(evaluation_result['partial_credit_details'])


def render_aggregate_dashboard(aggregate_results: Dict[str, Any], key_prefix: str = ""):
    """
    Render dashboard for aggregate results across multiple queries.

    Args:
        aggregate_results: Full aggregate results from AggregateResults.to_dict()
        key_prefix: Unique prefix to avoid key collisions
    """
    st.title("📈 Aggregate Evaluation Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", aggregate_results.get('total_queries', 0))

    with col2:
        st.metric("Pass Rate", f"{aggregate_results.get('overall_pass_rate', 0):.1%}")

    with col3:
        st.metric("Avg Score", f"{aggregate_results.get('overall_average_score', 0):.3f}")

    with col4:
        st.metric("Partial Credit Queries", aggregate_results.get('queries_with_partial_credit', 0))

    st.divider()

    # Pass rate by difficulty
    st.subheader("📊 Pass Rate by Difficulty")

    difficulty_data = {
        "Easy": aggregate_results.get('easy_pass_rate', 0),
        "Medium": aggregate_results.get('medium_pass_rate', 0),
        "Hard": aggregate_results.get('hard_pass_rate', 0)
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(difficulty_data.keys()),
            y=list(difficulty_data.values()),
            marker_color=['green', 'orange', 'red']
        )
    ])
    fig.update_layout(
        yaxis_title="Pass Rate",
        yaxis=dict(tickformat='.0%'),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}difficulty_pass_rate_chart")

    st.divider()

    # Grader performance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Code Grader Performance")
        code_graders = aggregate_results.get('code_grader_aggregates', {})
        if code_graders:
            df = pd.DataFrame([
                {"Grader": k, "Avg Score": v}
                for k, v in code_graders.items()
            ]).sort_values('Avg Score', ascending=False)
            st.dataframe(df, width="stretch")

    with col2:
        st.subheader("🤖 Model Grader Performance")
        model_graders = aggregate_results.get('model_grader_aggregates', {})
        if model_graders:
            df = pd.DataFrame([
                {"Metric": k, "Avg Score": v}
                for k, v in model_graders.items()
            ]).sort_values('Avg Score', ascending=False)
            st.dataframe(df, width="stretch")

    st.divider()

    # Category performance
    if aggregate_results.get('category_performance'):
        st.subheader("📂 Performance by Category")
        category_perf = aggregate_results['category_performance']

        fig = px.bar(
            x=list(category_perf.keys()),
            y=list(category_perf.values()),
            labels={'x': 'Category', 'y': 'Pass Rate'},
            title="Pass Rate by Query Category"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}category_performance_chart")

    # Cost and performance stats
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Cost", f"${aggregate_results.get('total_cost_usd', 0):.2f}")

    with col2:
        st.metric("Avg Latency", f"{aggregate_results.get('avg_latency_ms', 0):.0f}ms")

    with col3:
        st.metric("Total Tokens", f"{aggregate_results.get('total_tokens', 0):,}")
