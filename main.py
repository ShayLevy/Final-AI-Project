"""
Insurance Claim Timeline Retrieval System
Main Orchestrator

This system demonstrates:
- Multi-agent orchestration
- Data segmentation & hierarchical indexing
- Summary Index (MapReduce) + Hierarchical Chunk Index
- MCP tool integration
- LLM-as-a-judge evaluation

Usage:
    Interactive mode:  python main.py
    Evaluation mode:   python main.py --evaluate
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

# Import project modules
from src.vector_store.setup import VectorStoreManager
from src.indexing.document_loader import InsuranceClaimLoader
from src.indexing.chunking import HierarchicalChunker
from src.indexing.build_indexes import IndexBuilder
from src.retrieval.hierarchical_retriever import HierarchicalRetriever
from src.agents.langchain_integration import LangChainIntegration
from src.agents.manager_agent import ManagerAgent
from src.agents.summarization_agent import SummarizationAgent
from src.agents.needle_agent import NeedleAgent
from src.mcp.tools import get_all_mcp_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class InsuranceClaimSystem:
    """
    Main system orchestrator for Insurance Claim Timeline Retrieval
    """

    def __init__(
        self,
        data_dir: str = "./data",
        chroma_dir: str = "./chroma_db",
        rebuild_indexes: bool = False
    ):
        """
        Initialize the Insurance Claim System

        Args:
            data_dir: Directory containing claim documents
            chroma_dir: Directory for ChromaDB persistence
            rebuild_indexes: Whether to rebuild indexes from scratch
        """
        logger.info("=" * 70)
        logger.info("Initializing Insurance Claim Timeline Retrieval System")
        logger.info("=" * 70)

        self.data_dir = Path(data_dir)
        self.chroma_dir = Path(chroma_dir)

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")

        # Initialize components
        self._initialize_system(rebuild_indexes)

        logger.info("System initialization complete!")
        logger.info("=" * 70)

    def _initialize_system(self, rebuild_indexes: bool):
        """Initialize all system components"""

        # 1. Initialize Vector Store Manager
        logger.info("\n[1/8] Initializing Vector Store Manager...")
        self.vsm = VectorStoreManager(persist_dir=str(self.chroma_dir))

        if rebuild_indexes:
            logger.info("Rebuilding indexes - resetting collections...")
            self.vsm.reset_collections()

        # 2. Load Documents
        logger.info("\n[2/8] Loading insurance claim documents...")
        self.loader = InsuranceClaimLoader(data_dir=str(self.data_dir))
        self.documents = self.loader.load_all_documents()
        logger.info(f"Loaded {len(self.documents)} document sections")

        # Print summary
        summary = self.loader.get_document_summary()
        logger.info(f"Document Summary: {summary}")

        # 3. Create Hierarchical Chunks
        logger.info("\n[3/8] Creating hierarchical chunks...")
        self.chunker = HierarchicalChunker(
            chunk_sizes=[2048, 512, 128],  # large, medium, small
            chunk_overlap_ratio=0.2
        )
        self.nodes = self.chunker.chunk_documents(self.documents)

        # 4. Build Indexes
        logger.info("\n[4/8] Building indexes...")
        self.index_builder = IndexBuilder(self.vsm)

        # Check if indexes already exist
        stats = self.vsm.get_collection_stats()
        summary_exists = stats.get('summary', {}).get('count', 0) > 0
        hier_exists = stats.get('hierarchical', {}).get('count', 0) > 0

        if rebuild_indexes or not (summary_exists and hier_exists):
            logger.info("Building Summary Index with MapReduce...")
            self.summary_index = self.index_builder.build_summary_index(self.documents)

            logger.info("Building Hierarchical Index...")
            self.hierarchical_index, self.nodes = self.index_builder.build_hierarchical_index(self.nodes)
        else:
            logger.info("Loading existing indexes from ChromaDB...")
            # Load existing indexes (simplified for demo)
            from llama_index.core import VectorStoreIndex
            summary_collection = self.vsm.get_summary_collection()
            hier_collection = self.vsm.get_hierarchical_collection()

            summary_storage = self.vsm.create_storage_context(summary_collection)
            hier_storage = self.vsm.create_storage_context(hier_collection)

            self.summary_index = VectorStoreIndex.from_vector_store(
                summary_storage.vector_store
            )
            self.hierarchical_index = VectorStoreIndex.from_vector_store(
                hier_storage.vector_store
            )

        # 5. Create Retrievers
        logger.info("\n[5/8] Creating retrievers...")
        self.hier_retriever = HierarchicalRetriever(self.hierarchical_index, self.nodes)

        # 6. Create MCP Tools
        logger.info("\n[6/8] Initializing MCP tools...")
        self.mcp_tools = get_all_mcp_tools()
        logger.info(f"Loaded {len(self.mcp_tools)} MCP tools")

        # 7. Create LangChain Integration
        logger.info("\n[7/8] Setting up LangChain integration...")
        self.integration = LangChainIntegration(
            summary_index=self.summary_index,
            hierarchical_retriever=self.hier_retriever,
            mcp_tools=self.mcp_tools
        )
        self.all_tools = self.integration.get_all_tools()

        # 8. Initialize Agents
        logger.info("\n[8/8] Initializing agents...")

        # Manager Agent (router)
        self.manager_agent = ManagerAgent(tools=self.all_tools)

        # Specialist Agents
        self.summarization_agent = SummarizationAgent(
            summary_index=self.summary_index
        )
        from langchain_openai import ChatOpenAI
        self.needle_agent = NeedleAgent(
            hierarchical_retriever=self.hier_retriever,
            llm=ChatOpenAI(model="gpt-4", temperature=0)
        )

    def query(self, query: str, use_manager: bool = True) -> dict:
        """
        Query the system

        Args:
            query: User query
            use_manager: Whether to use manager agent (True) or direct routing (False)

        Returns:
            Response dictionary
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'=' * 70}")

        if use_manager:
            # Use manager agent for intelligent routing
            result = self.manager_agent.query(query)
        else:
            # Direct routing based on simple keywords
            query_lower = query.lower()
            if any(word in query_lower for word in ['summarize', 'overview', 'timeline', 'what happened']):
                result = self.summarization_agent.query(query)
            elif any(word in query_lower for word in ['exact', 'specific', 'how much', 'when', 'who']):
                result = self.needle_agent.query(query)
            else:
                result = self.manager_agent.query(query)

        logger.info(f"\nRESPONSE: {result.get('output', 'No output')[:500]}...")

        return result

    def get_statistics(self) -> dict:
        """Get system statistics"""
        stats = self.vsm.get_collection_stats()
        return {
            "documents_loaded": len(self.documents),
            "hierarchical_chunks": len(self.nodes),
            "chroma_stats": stats,
            "available_tools": len(self.all_tools),
            "tool_names": [t.name for t in self.all_tools]
        }

    def update_retrieval_k(self, k: int):
        """
        Update the retrieval k value for all tools

        Args:
            k: Number of chunks to retrieve
        """
        logger.info(f"Updating retrieval k to {k}")

        # Update integration layer with new k
        self.integration = LangChainIntegration(
            summary_index=self.summary_index,
            hierarchical_retriever=self.hier_retriever,
            mcp_tools=self.mcp_tools,
            retrieval_k=k
        )
        self.all_tools = self.integration.get_all_tools()

        # Recreate manager agent with new tools
        self.manager_agent = ManagerAgent(tools=self.all_tools)

        logger.info(f"Retrieval k updated to {k}, tools recreated")


class EvaluationRunner:
    """
    Runs evaluation suite and generates reports using LLM-as-a-Judge.

    Uses Anthropic Claude as judge (separate from OpenAI GPT-4 used for generation)
    to ensure unbiased evaluation.

    Requires ANTHROPIC_API_KEY environment variable to be set.
    """

    def __init__(self, system: InsuranceClaimSystem, output_dir: str = "./evaluation_results"):
        """
        Initialize evaluation runner

        Args:
            system: Initialized InsuranceClaimSystem
            output_dir: Directory to save results

        Note: Uses Anthropic Claude as judge (separate from OpenAI GPT-4 used for generation)
        """
        from src.evaluation.judge import LLMJudge

        self.system = system
        # Use Claude as judge (default: claude-sonnet-4-20250514) - separate from GPT-4 used for generation
        self.judge = LLMJudge(temperature=0)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("EvaluationRunner initialized with Claude judge (separate from GPT-4 generation)")

    def run_full_evaluation(self) -> dict:
        """
        Run complete evaluation on all test queries

        Returns:
            Dictionary with all results
        """
        from src.evaluation.test_queries import TestSuite

        logger.info("=" * 70)
        logger.info("STARTING FULL EVALUATION")
        logger.info("=" * 70)

        test_queries = TestSuite.get_test_queries()
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(test_queries),
            "query_results": [],
            "aggregate_scores": {}
        }

        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Evaluating Query {i}/{len(test_queries)}: {test_case['id']}")
            logger.info(f"{'=' * 70}")

            result = self.evaluate_query(test_case)
            results["query_results"].append(result)

            # Print summary for this query
            self._print_query_summary(result)

        # Calculate aggregate scores
        results["aggregate_scores"] = self._calculate_aggregate_scores(results["query_results"])

        # Save results
        self._save_results(results)

        # Print final summary
        self._print_final_summary(results)

        return results

    def evaluate_query(self, test_case: dict) -> dict:
        """
        Evaluate a single test query

        Args:
            test_case: Test case dictionary

        Returns:
            Evaluation result
        """
        query = test_case["query"]
        query_id = test_case["id"]

        logger.info(f"Query: {query}")

        # Run query through system
        try:
            system_response = self.system.query(query, use_manager=True)
            answer = system_response.get("output", "")
            success = system_response.get("success", False)

            # Extract retrieved context (if available)
            retrieved_context = ""
            if system_response.get("intermediate_steps"):
                for step in system_response["intermediate_steps"]:
                    if len(step) >= 2:
                        retrieved_context += str(step[1]) + "\n\n"

            # Perform evaluation
            eval_result = self.judge.evaluate_full(
                query=query,
                answer=answer,
                ground_truth=test_case["ground_truth"],
                retrieved_context=retrieved_context if retrieved_context else answer,
                expected_chunks=test_case.get("expected_chunks", []),
                retrieved_chunks=[retrieved_context] if retrieved_context else []
            )

            return {
                "query_id": query_id,
                "query": query,
                "query_type": test_case["type"],
                "system_answer": answer,
                "ground_truth": test_case["ground_truth"],
                "system_success": success,
                "evaluation": eval_result,
                "correctness_score": eval_result["correctness"]["score"],
                "relevancy_score": eval_result["relevancy"]["score"],
                "recall_score": eval_result.get("recall", {}).get("score", "N/A"),
                "average_score": eval_result["average_score"]
            }

        except Exception as e:
            logger.error(f"Error evaluating query {query_id}: {e}")
            return {
                "query_id": query_id,
                "query": query,
                "error": str(e),
                "system_success": False
            }

    def _calculate_aggregate_scores(self, query_results: list) -> dict:
        """Calculate aggregate scores across all queries"""
        correctness_scores = []
        relevancy_scores = []
        recall_scores = []
        average_scores = []

        for result in query_results:
            if "correctness_score" in result:
                correctness_scores.append(result["correctness_score"])
                relevancy_scores.append(result["relevancy_score"])

                recall = result.get("recall_score")
                if recall != "N/A":
                    recall_scores.append(recall)

                average_scores.append(result["average_score"])

        return {
            "avg_correctness": sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0,
            "avg_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            "avg_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "overall_average": sum(average_scores) / len(average_scores) if average_scores else 0,
            "total_evaluated": len(query_results),
            "successful_queries": sum(1 for r in query_results if r.get("system_success"))
        }

    def _save_results(self, results: dict):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"evaluation_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {filename}")

    def _print_query_summary(self, result: dict):
        """Print summary for a single query"""
        print(f"\n{'-' * 70}")
        print(f"Query ID: {result['query_id']}")
        print(f"Type: {result.get('query_type', 'N/A')}")
        print(f"{'-' * 70}")
        print(f"Correctness: {result.get('correctness_score', 'N/A')}/5")
        print(f"Relevancy: {result.get('relevancy_score', 'N/A')}/5")
        print(f"Recall: {result.get('recall_score', 'N/A')}/5")
        print(f"Average: {result.get('average_score', 'N/A'):.2f}/5")

    def _print_final_summary(self, results: dict):
        """Print final evaluation summary"""
        agg = results["aggregate_scores"]

        print("\n" + "=" * 70)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal Queries Evaluated: {agg['total_evaluated']}")
        print(f"Successful Queries: {agg['successful_queries']}")
        print(f"\n{'-' * 70}")
        print("AVERAGE SCORES (out of 5)")
        print(f"{'-' * 70}")
        print(f"Correctness:      {agg['avg_correctness']:.2f}")
        print(f"Relevancy:        {agg['avg_relevancy']:.2f}")
        print(f"Recall:           {agg['avg_recall']:.2f}")
        print(f"{'-' * 70}")
        print(f"OVERALL AVERAGE:  {agg['overall_average']:.2f}/5.00")
        print("=" * 70)

        # Performance interpretation
        overall = agg['overall_average']
        if overall >= 4.5:
            grade = "A (Excellent)"
        elif overall >= 4.0:
            grade = "B (Very Good)"
        elif overall >= 3.0:
            grade = "C (Good)"
        elif overall >= 2.0:
            grade = "D (Fair)"
        else:
            grade = "F (Needs Improvement)"

        print(f"\nPerformance Grade: {grade}")
        print("=" * 70 + "\n")


def run_evaluation():
    """Run evaluation mode"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         INSURANCE CLAIM SYSTEM - EVALUATION SUITE        ║
    ║              LLM-as-a-Judge Evaluation                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Initialize system
    logger.info("Initializing system...")
    system = InsuranceClaimSystem(
        data_dir="./data",
        chroma_dir="./chroma_db",
        rebuild_indexes=False
    )

    # Run evaluation
    runner = EvaluationRunner(system)
    results = runner.run_full_evaluation()

    print("\nEvaluation complete! Check ./evaluation_results/ for detailed results.\n")


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Insurance Claim Timeline Retrieval System               ║
    ║  Multi-Agent GenAI System with MCP Integration           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Initialize system
    # Set rebuild_indexes=True for first run or to refresh data
    system = InsuranceClaimSystem(
        data_dir="./data",
        chroma_dir="./chroma_db",
        rebuild_indexes=False  # Set to True to rebuild
    )

    # Print system statistics
    stats = system.get_statistics()
    print("\n=== System Statistics ===")
    for key, value in stats.items():
        if key != "chroma_stats":
            print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("System ready! Example queries:")
    print("=" * 70)

    # Example queries
    example_queries = [
        "What is this insurance claim about?",
        "What was the exact deductible amount?",
        "When did the accident occur?",
        "Summarize the timeline of events",
        "Who was the claims adjuster?",
        "What was the total repair cost?",
        "What did the witnesses say?",
        "How many days between the incident and claim filing?"
    ]

    print("\nExample queries you can try:")
    for i, q in enumerate(example_queries, 1):
        print(f"{i}. {q}")

    print("\n" + "=" * 70)
    print("Interactive mode - enter your queries (or 'quit' to exit)")
    print("=" * 70 + "\n")

    # Interactive query loop
    while True:
        try:
            user_query = input("\n🔍 Your query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not user_query:
                continue

            # Process query
            result = system.query(user_query, use_manager=True)

            # Display result
            print("\n" + "=" * 70)
            print("📊 RESPONSE:")
            print("=" * 70)
            print(result.get('output', 'No response generated'))

            if result.get('intermediate_steps'):
                print("\n" + "-" * 70)
                print("🔧 Tools Used:")
                print("-" * 70)
                for step in result['intermediate_steps']:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        print(f"• {action.tool}: {action.tool_input}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.error(f"Error in query loop: {e}", exc_info=True)


if __name__ == "__main__":
    # Check for --evaluate flag
    if len(sys.argv) > 1 and sys.argv[1] in ['--evaluate', '-e', 'evaluate']:
        run_evaluation()
    else:
        main()
