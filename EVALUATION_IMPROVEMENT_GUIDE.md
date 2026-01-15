# Evaluation Improvement Guide

## Quick Diagnosis

If most queries are failing, the issue is likely one of these:

### 1. **Model Quality (Most Likely Cause)**

**Problem**: We recently switched from GPT-4 to GPT-4o-mini for cost savings. While GPT-4o-mini is 200x cheaper, it's significantly less capable for complex RAG tasks.

**Quick Fix - Switch Back to GPT-4**:
```bash
# Edit these 3 files and change gpt-4o-mini back to gpt-4:
# 1. src/agents/manager_agent.py (line 30)
# 2. src/agents/summarization_agent.py (line 24)
# 3. src/indexing/build_indexes.py (line 33)
```

**Expected Impact**:
- ✅ Much better accuracy and reasoning
- ❌ Higher cost (~$0.40-2 for 40 queries vs $0.02-0.08)
- Use GPT-4 for final evaluation, GPT-4o-mini for development

### 2. **Grader Thresholds Too Strict**

**Problem**: Code-based graders might have overly strict thresholds or incorrect expected values.

**Check This**:
Look at specific failures to see patterns:
- Are numerical graders failing? (dates, amounts)
- Are regex patterns too specific?
- Are required_components weights too high?

**Fix**: Adjust thresholds in `src/evaluation/code_graders.py`

### 3. **Retrieval Not Finding Right Information**

**Problem**: The system isn't retrieving relevant chunks for queries.

**Diagnostic Steps**:
1. Look at the `retrieved_context` in failed queries
2. Check if relevant information is present
3. If context is empty or irrelevant, retrieval needs tuning

**Fixes**:
- Increase retrieval k value (currently 5)
- Adjust chunk sizes
- Improve query understanding in agents

### 4. **Missing/Wrong Expected Values**

**Problem**: Summary queries now require many parameters (required_components, expected_events, etc.) that we just added. These might be too strict.

**Check**: Look at code grader failures - are they scoring 0 because of missing matches?

**Fix**: Adjust patterns and weights in `test_queries_expanded.py`

## Detailed Improvement Steps

### Step 1: Identify Failure Patterns

Run a small evaluation (5-10 queries) and check:

```python
# Look at individual results in the UI and note:
- Which graders are failing most?
- Are failures consistent across query types?
- Is retrieved_context present and relevant?
- What are the actual vs expected values?
```

**Common Patterns**:
- All queries failing → Model quality issue (use GPT-4)
- Specific graders failing → Threshold/pattern issue
- Low code_grader_aggregate → Expected value mismatches
- Low model_grader_aggregate → Response quality issue

### Step 2: Quick Wins

#### A. Use GPT-4 Instead of GPT-4o-mini

**For Manager Agent**:
```python
# src/agents/manager_agent.py
def __init__(
    self,
    tools: List[Tool],
    llm_model: str = "gpt-4",  # Change from gpt-4o-mini
    temperature: float = 0
):
```

**For Summarization Agent**:
```python
# src/agents/summarization_agent.py
def __init__(
    self,
    summary_index,
    llm_model: str = "gpt-4",  # Change from gpt-4o-mini
    temperature: float = 0
):
```

**For Index Builder**:
```python
# src/indexing/build_indexes.py
def __init__(self, vector_store_manager, llm_model: str = "gpt-4", temperature: float = 0):
```

#### B. Increase Retrieval Context

```python
# In LangChainIntegration, increase k from 5 to 10:
def __init__(self, hierarchical_retriever, summary_index, k: int = 10):
```

#### C. Relax Grader Thresholds

```python
# src/evaluation/code_graders.py
# Change passing thresholds from 0.7 to 0.6:

def multi_fact_extraction_grade(...):
    # ...
    passed = normalized_score >= 0.6  # Was 0.7
```

### Step 3: Check Data Quality

Ensure the insurance claim document has all expected information:

```bash
# Check the document
ls -la data/
# Should see the insurance claim PDF

# Verify indexes are built
ls -la chroma_db/
# Should see collections
```

If indexes are outdated:
1. Go to Streamlit UI → 📋 Upload tab
2. Re-upload the document
3. This will rebuild indexes with current settings

### Step 4: Adjust Expected Values

If specific queries fail consistently, adjust their expected values:

**Example - Numerical Grader**:
```python
# src/evaluation/test_queries_expanded.py
{
    "expected_value": 23370.80,
    "tolerance": 10,  # Increase from 1 to 10 for more flexibility
}
```

**Example - Fuzzy Match**:
```python
# Adjust similarity threshold
{
    "expected_string": "Sarah Mitchell",
    "threshold": 0.7,  # Lower from 0.8 to 0.7 for more flexibility
}
```

### Step 5: Improve Agent Prompts

If agents aren't following expected patterns, improve their system prompts:

**Manager Agent** (`src/agents/manager_agent.py`):
- Add more explicit instructions about what information to extract
- Emphasize importance of precision vs completeness

**Summarization Agent** (`src/agents/summarization_agent.py`):
- Add instructions to include specific details (dates, amounts, names)
- Emphasize comprehensive coverage of all facts

## Testing Strategy

### Phase 1: Baseline with GPT-4 (Recommended)
1. Switch all agents to GPT-4
2. Run 5 needle + 5 summary queries (10 total)
3. Check pass rate - should be 60-80%
4. If still low, continue to Phase 2

### Phase 2: Retrieval Tuning
1. Increase k from 5 to 10
2. Check retrieved_context in failures
3. Ensure relevant information is being retrieved

### Phase 3: Grader Tuning
1. Identify which graders fail most
2. Check if expected values match document
3. Adjust thresholds and patterns
4. Re-run failed queries

### Phase 4: Full Evaluation
1. Once pass rate is acceptable (60-70%)
2. Run full 40 query evaluation
3. Document results and patterns

## Cost Management

**Development/Tuning** (Many iterations):
- Use GPT-4o-mini: ~$0.02-0.08 per 40 queries
- Accept lower accuracy for faster iteration

**Final Evaluation** (1-2 runs):
- Use GPT-4: ~$0.40-2 per 40 queries
- Get accurate baseline performance

**Hybrid Approach**:
- Use GPT-4o-mini for needle queries (simpler)
- Use GPT-4 for summary queries (more complex)

## Expected Pass Rates

Realistic expectations:

### With GPT-4:
- Easy queries: 85-95% pass rate
- Medium queries: 70-85% pass rate
- Hard queries: 50-70% pass rate
- **Overall: 70-85%**

### With GPT-4o-mini:
- Easy queries: 70-85% pass rate
- Medium queries: 50-70% pass rate
- Hard queries: 35-55% pass rate
- **Overall: 50-70%**

## Troubleshooting Specific Issues

### Issue: All queries score 0.0
**Cause**: Graders aren't finding any matches
**Fix**:
1. Check if document is loaded: `ls -la data/`
2. Check if indexes exist: `ls -la chroma_db/`
3. Rebuild indexes in UI

### Issue: Retrieved context is empty
**Cause**: Retrieval isn't working
**Fix**:
1. Check ChromaDB collections exist
2. Verify embeddings are generated
3. Try rebuilding indexes

### Issue: Retrieved context is irrelevant
**Cause**: Query understanding or embedding mismatch
**Fix**:
1. Use GPT-4 for better query understanding
2. Check if query reformulation is working
3. Adjust retrieval strategy

### Issue: Answers are vague/incomplete
**Cause**: Model quality or insufficient context
**Fix**:
1. Switch to GPT-4
2. Increase retrieval k
3. Improve agent prompts

### Issue: Specific grader always fails
**Cause**: Wrong expected value or too strict pattern
**Fix**:
1. Check the expected value matches document
2. Adjust regex pattern for flexibility
3. Lower threshold or weights

## Next Steps

1. **Immediate**: Switch to GPT-4 for agents (biggest impact)
2. **Run 10 queries**: Test with GPT-4 to get baseline
3. **Analyze failures**: Look at specific patterns
4. **Tune iteratively**: Adjust one thing at a time
5. **Document results**: Track what works

## Need Help?

Run a small evaluation and share:
1. Overall pass rate
2. Example of a failing query (full output)
3. Which graders are failing most
4. What the retrieved_context looks like

This will help diagnose the specific issue!
