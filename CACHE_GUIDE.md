# Evaluation Cache System Guide

## Overview

The evaluation cache system saves query results to disk, allowing you to reuse previous evaluations and **significantly reduce API costs**. When cache is enabled, queries that have already been evaluated are loaded from cache instead of re-running through the system.

## Benefits

✅ **Save API Costs**: Avoid redundant evaluations (can save 90%+ on repeated runs)
✅ **Faster Results**: Cached queries load instantly
✅ **Persistent Storage**: Cache survives app restarts
✅ **Selective Invalidation**: Clear specific query types or entire cache
✅ **Automatic Updates**: New runs automatically update cache

## How It Works

```
1. Run Evaluation
   ├─ Check cache first
   ├─ If cached: Load result (instant, free)
   └─ If not cached: Run evaluation → Cache result

2. Subsequent Runs
   └─ Use cached results (saves API calls)

3. Clear Cache
   └─ Force re-evaluation of all queries
```

## Cache Structure

```
./evaluation_cache/
├── needle_results_cache.json    # Needle query results
└── summary_results_cache.json   # Summary query results
```

Each cached entry includes:
- Full evaluation result
- Timestamp when cached
- Model used for evaluation
- Cache version

## Using the Cache in Streamlit

### Enable/Disable Cache

In both Needle and Summary tabs:

```
☑️ Use Cache (saves API costs)  [Checked by default]
```

**When enabled**: Queries already evaluated are loaded from cache
**When disabled**: All queries are re-evaluated (ignores cache)

### Check Cache Status

When cache is enabled, you'll see:
```
💾 Cache enabled: 15/20 needle queries cached (will skip these)
```

This shows:
- How many queries are already cached
- How many will actually run (5 in this example)

### Clear Cache

Click the **🗑️ Clear Cache** button to:
- Remove all cached results for that query type
- Force fresh evaluation on next run

**Note**: Clearing is per-type (needle or summary), not global.

## Cost Savings Example

### Scenario: Testing and tuning evaluations

**Without Cache** (10 iterations × 40 queries):
- 400 total evaluations
- Cost: ~$4-20 (GPT-4) or ~$0.20-0.80 (GPT-4o-mini)

**With Cache** (1st run + 9 cached runs):
- 40 evaluations + 360 cache hits
- Cost: ~$0.40-2 (GPT-4) or ~$0.02-0.08 (GPT-4o-mini)
- **Savings: 90%**

## When to Use Cache

### ✅ Use Cache When:
- Tuning grader thresholds (queries don't change)
- Adjusting expected values (responses stay same)
- Comparing different grader configurations
- Testing UI changes
- Running multiple times for analysis

### ❌ Clear Cache When:
- System configuration changed (different model, retrieval settings)
- Document was updated (new information)
- Indexes were rebuilt
- Need fresh baseline for final evaluation
- Testing new query understanding

## Cache Behavior

### What Gets Cached?
- ✅ Full evaluation result (all grader scores)
- ✅ Query response
- ✅ Retrieved context
- ✅ Tool trace
- ✅ All metadata

### What Triggers Cache Update?
- Running evaluation with cache enabled
- Result is automatically saved after evaluation completes

### What Doesn't Affect Cache?
- Model grader enable/disable (cache stores both)
- UI changes
- Session restarts

## Advanced Usage

### Via Command Line

```python
from src.evaluation.cache_manager import EvaluationCache

# Initialize cache
cache = EvaluationCache()

# Check stats
stats = cache.get_cache_stats()
print(f"Total cached: {stats['total_cached']}")

# List cached queries
needle_queries = cache.list_cached_queries("needle")
print(f"Cached needle queries: {needle_queries}")

# Get specific cache info
info = cache.get_cache_info("Q_NEEDLE_01", "needle")
print(f"Cached at: {info['timestamp']}")

# Invalidate specific query
cache.invalidate_query("Q_NEEDLE_05", "needle")

# Clear all cache
cache.clear_cache()  # Both types
cache.clear_cache("needle")  # Only needle
cache.clear_cache("summary")  # Only summary
```

### Programmatic Usage

```python
from src.evaluation.orchestrator import EvaluationOrchestrator

# Create orchestrator with cache
orchestrator = EvaluationOrchestrator(
    system=system,
    use_cache=True  # Enable cache
)

# Disable cache temporarily
orchestrator.use_cache = False

# Re-enable
orchestrator.use_cache = True
```

## Cache Files

Cache files are stored in `./evaluation_cache/`:

```json
{
  "Q_NEEDLE_01": {
    "result": { /* Full evaluation result */ },
    "timestamp": "2024-01-15T10:30:00",
    "model": "gpt-4o-mini",
    "cache_version": "1.0"
  },
  ...
}
```

**Format**: JSON (human-readable)
**Size**: ~50-100KB per query (~2-4MB for all 40 queries)
**Location**: Git-ignored by default

## Troubleshooting

### Cache not being used?
✅ Check: "Use Cache" checkbox is enabled
✅ Check: Queries are named correctly (Q_NEEDLE_XX or Q_SUMMARY_XX)
✅ Check: Cache files exist in `./evaluation_cache/`

### Results seem outdated?
✅ Clear cache and re-run
✅ Check cache timestamp in files

### Cache files too large?
✅ Clear old cache: `rm -rf evaluation_cache/`
✅ Selectively invalidate unused queries

### Want to share cached results?
✅ Copy `evaluation_cache/` folder
✅ Or: Export to JSON and import

## Best Practices

1. **Enable cache by default** - Saves costs during development
2. **Clear cache after system changes** - Ensure fresh results
3. **Keep cache between runs** - Maximize reuse
4. **Monitor cache status** - Know what's being reused
5. **Clear before final eval** - Get accurate baseline

## Cache Strategy Recommendations

### Development Phase
```
✅ Cache: Enabled
📊 Model: GPT-4o-mini (cheap)
🔄 Iterations: Many
💰 Cost: Minimal
```

### Tuning Phase
```
✅ Cache: Enabled
📊 Model: GPT-4o-mini or GPT-4
🔄 Clear cache: Only when changing system
💰 Cost: Low
```

### Final Evaluation
```
❌ Cache: Clear first, then enable
📊 Model: GPT-4 (accurate)
🔄 Iterations: 1-2
💰 Cost: Full (but only once)
```

## Summary

The cache system is a powerful tool to reduce costs during development and testing. Enable it by default, clear it when making system changes, and enjoy massive cost savings!

**Expected Savings**: 90% reduction in API costs for iterative testing
