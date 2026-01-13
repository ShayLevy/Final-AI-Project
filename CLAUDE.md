# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Insurance Claim Timeline Retrieval System - a multi-agent RAG system for querying insurance claim documents. Uses LlamaIndex for indexing, LangChain for agent orchestration, ChromaDB for vector storage, and RAGAS for evaluation.

## Commands

### Run the Application
```bash
# Activate virtual environment first
source venv/bin/activate

# Run Streamlit UI (main interface)
streamlit run streamlit_app.py

# Run CLI mode
python main.py

# Run evaluation suite (LLM-as-a-Judge with Claude)
python main.py --evaluate
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env (copy from .env.example)
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

## Architecture

### Core Flow
```
User Query → ManagerAgent (router) → Tool Selection → Response
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
SummaryRetriever  NeedleRetriever  MCP Tools
(MapReduce)       (Hierarchical)   (Computations)
    ↓               ↓
Summary Index   Hierarchical Index
    ↓               ↓
    └───────ChromaDB────────┘
```

### Key Components

**Agents** (`src/agents/`):
- `manager_agent.py`: LangGraph ReAct agent that routes queries to appropriate tools
- `summarization_agent.py`: Handles high-level summary queries using MapReduce strategy
- `needle_agent.py`: Handles precise fact-finding using small chunks
- `langchain_integration.py`: Bridges LlamaIndex retrievers to LangChain tools

**Indexing** (`src/indexing/`):
- `document_loader.py`: Loads and parses insurance claim PDFs
- `chunking.py`: Hierarchical chunker (2048/512/128 tokens with 20% overlap)
- `build_indexes.py`: Creates Summary Index (MapReduce) and Hierarchical Index

**Retrieval** (`src/retrieval/`):
- `hierarchical_retriever.py`: Auto-merging retriever with 3-tier fallback (exact match → partial match → semantic search)

**MCP Tools** (`src/mcp/tools.py`):
- `GetDocumentMetadata`: Retrieve claim metadata
- `CalculateDaysBetween`: Date arithmetic
- `EstimateCoveragePayout`: Insurance payout calculations
- `ValidateClaimStatus`: Status validation
- `GetTimelineSummary`: Timeline milestones

**Evaluation** (`src/evaluation/`):
- `judge.py`: LLM-as-a-Judge using Anthropic Claude (separate from GPT-4 generation)
- `test_queries.py`: 10 test queries (5 summary + 5 needle) with ground truth

### Data Flow
1. PDF uploaded via Streamlit or placed in `./data/`
2. Document parsed into sections with metadata
3. Hierarchical chunks created (small→medium→large with parent-child relationships)
4. Chunks indexed in ChromaDB (two collections: `insurance_summaries`, `insurance_hierarchical`)
5. ManagerAgent routes queries to appropriate retriever or MCP tool
6. Response synthesized and returned

### Key Design Decisions
- **Dual indexes**: Summary Index for "what happened" queries; Hierarchical Index for precise facts
- **Three chunk levels**: Small (128), Medium (512), Large (2048) tokens enable precision vs context trade-off
- **20% overlap**: Prevents information loss at chunk boundaries
- **Separate evaluation model**: Claude judges GPT-4 outputs to avoid self-preference bias

## Configuration

**ChromaDB Collections**:
- `insurance_summaries`: Pre-computed section summaries
- `insurance_hierarchical`: Multi-granularity chunks with parent-child metadata

**Retrieval Settings** (in `LangChainIntegration`):
- Default `k=5` chunks per query
- Can be updated via `system.update_retrieval_k(k)`

## API Keys Required

Both keys must be set in `.env`:
- `OPENAI_API_KEY`: RAG system (GPT-4 generation, embeddings, RAGAS)
- `ANTHROPIC_API_KEY`: LLM-as-a-Judge evaluation (Claude)
