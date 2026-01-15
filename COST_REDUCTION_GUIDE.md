# Cost Reduction Guide

## Problem
Your evaluation system was consuming high API credits due to:
1. Using expensive models (GPT-4, Claude Sonnet 4)
2. Making 11 API calls per query (1 for answer + 10 for model-based graders)
3. For 40 queries: 440 API calls costing ~$4-14

## Solutions Implemented

### ✅ 1. Switched to Cheaper Models (DONE)

Changed default models in the codebase:

| Component | Before | After | Cost Savings |
|-----------|--------|-------|--------------|
| Manager Agent | GPT-4 | GPT-4o-mini | 200x cheaper |
| Summarization Agent | GPT-4 | GPT-4o-mini | 200x cheaper |
| Index Builder | GPT-4 | GPT-4o-mini | 200x cheaper |
| Judge (evaluation) | Claude Sonnet 4 | Claude Haiku | 12x cheaper |

**Files Modified:**
- `src/agents/manager_agent.py` (line 30)
- `src/agents/summarization_agent.py` (line 24)
- `src/indexing/build_indexes.py` (line 33)
- `src/evaluation/judge.py` (line 32)

### ✅ 2. Model Graders Already Disabled by Default

The Streamlit UI has checkboxes **unchecked by default**:
- "Include Model-Based Graders (slower, costs $)"

This reduces from 11 API calls to 1 per query (90% reduction).

## Cost Comparison

### Before Changes
**Per Query (with model graders):**
- 1 × GPT-4 call: ~$0.01-0.05
- 10 × Claude Sonnet 4 calls: ~$0.10-0.30
- **Total: ~$0.11-0.35 per query**

**40 queries: $4.40 - $14.00**

### After Changes (Model Graders Disabled)
**Per Query:**
- 1 × GPT-4o-mini call: ~$0.0005-0.002
- **Total: ~$0.0005-0.002 per query**

**40 queries: $0.02 - $0.08** ✅ **99% cost reduction!**

### After Changes (Model Graders Enabled)
**Per Query:**
- 1 × GPT-4o-mini call: ~$0.0005-0.002
- 10 × Claude Haiku calls: ~$0.01-0.03
- **Total: ~$0.011-0.032 per query**

**40 queries: $0.44 - $1.28** ✅ **90% cost reduction!**

## Recommendations

### For Development/Testing (What You Should Use Now)
- ✅ Use **GPT-4o-mini** (already set)
- ✅ **Disable model-based graders** (already default)
- ✅ Cost: ~$0.02-0.08 for 40 queries

### For Quick Iterations (Even Cheaper)
If quality is acceptable, you can manually change to `gpt-3.5-turbo` in the agent files:
- Edit the 3 files again and change `gpt-4o-mini` → `gpt-3.5-turbo`
- Even cheaper but lower quality

### For Production/Final Evaluation
When you need highest quality:
- Change back to `gpt-4` in the 3 agent files
- Change judge back to `claude-sonnet-4-20250514`
- **Enable model-based graders** for comprehensive evaluation
- Cost: ~$4-14 for 40 queries (acceptable for final evaluation)

## How to Verify Changes

1. **Check model in use:**
   ```bash
   grep "llm_model.*=" src/agents/*.py src/indexing/build_indexes.py
   ```
   Should show `gpt-4o-mini`

2. **Check judge model:**
   ```bash
   grep "judge_model.*=" src/evaluation/judge.py
   ```
   Should show `claude-3-5-haiku-20241022`

3. **Run a test evaluation:**
   - Open Streamlit UI
   - Go to Evaluation tab
   - Keep "Include Model-Based Graders" UNCHECKED
   - Run a few queries
   - Check the cost in your API dashboard

## Model Quality Comparison

| Model | Speed | Cost | Quality | Best For |
|-------|-------|------|---------|----------|
| GPT-4 | Slow | $$$$$ | Excellent | Production |
| GPT-4o-mini | Fast | $ | Very Good | Development/Testing |
| GPT-3.5-turbo | Very Fast | $ | Good | Quick iterations |
| Claude Sonnet 4 | Medium | $$$$ | Excellent | Final evaluation |
| Claude Haiku | Fast | $ | Good | Dev evaluation |

## Next Steps

1. ✅ Changes are already applied
2. ✅ Restart your Streamlit app to use new models
3. ✅ Run evaluations with model graders disabled
4. ✅ Monitor your API costs to verify savings

**Expected result: 99% cost reduction (from ~$4-14 to ~$0.02-0.08 for 40 queries)**
