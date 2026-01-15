"""
Summarization Expert Agent
Specialized in high-level summaries and timeline queries using the Summary Index
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationAgent:
    """
    Specialized agent for high-level summaries and timeline questions
    Uses the Summary Index built via MapReduce
    """

    def __init__(
        self,
        summary_index,
        llm_model: str = "gpt-4o-mini",  # Changed from gpt-4 to save costs (200x cheaper!)
        temperature: float = 0
    ):
        """
        Initialize Summarization Agent

        Args:
            summary_index: LlamaIndex DocumentSummaryIndex
            llm_model: LLM model to use
            temperature: Temperature setting
        """
        self.summary_index = summary_index
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",  # MapReduce strategy
            verbose=True
        )

        logger.info("SummarizationAgent initialized")

    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a summary query

        Args:
            query: User query

        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"SummarizationAgent processing: '{query[:100]}...'")

        # Enhance query with summarization instructions
        enhanced_prompt = f"""Based on the insurance claim documents, {query}

Provide a clear, well-structured summary that includes SPECIFIC DETAILS:
- Claim ID and key dates (incident date, filing date)
- Names of all parties involved (policyholder, at-fault party, witnesses, adjuster)
- Specific amounts (repair costs, deductibles, total claim amount)
- Location of incident
- Key events in chronological order
- Important outcomes or decisions

Be specific and factual. Include actual numbers, dates, and names from the documents.
Do NOT give a generic overview - include the specific details that make this claim unique."""

        try:
            response = self.query_engine.query(enhanced_prompt)

            return {
                "query": query,
                "output": str(response),
                "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                "agent_type": "summarization",
                "index_used": "summary",
                "success": True
            }

        except Exception as e:
            logger.error(f"SummarizationAgent error: {e}")
            return {
                "query": query,
                "output": f"Error generating summary: {str(e)}",
                "error": str(e),
                "agent_type": "summarization",
                "success": False
            }

    def get_timeline(self, query: str = None) -> Dict[str, Any]:
        """
        Get timeline of events

        Args:
            query: Optional specific timeline query

        Returns:
            Timeline information
        """
        if query is None:
            query = "Provide a chronological timeline of all key events in this insurance claim, from the incident through resolution."

        logger.info("Generating timeline...")

        timeline_prompt = f"""{query}

Format the timeline as:
- Date/Time: Event description

Include all significant milestones."""

        return self.query(timeline_prompt)

    def get_overview(self) -> Dict[str, Any]:
        """
        Get high-level overview of the entire claim

        Returns:
            Overview information
        """
        logger.info("Generating claim overview...")

        overview_query = """Provide a comprehensive overview of this insurance claim including:

1. What happened (brief incident description)
2. Who was involved (parties, adjusters, medical providers)
3. Key dates (incident, filing, resolution)
4. Financial summary (total costs, settlements)
5. Current status

Keep it concise but complete."""

        return self.query(overview_query)

    def get_summary_by_category(self, category: str) -> Dict[str, Any]:
        """
        Get summary of a specific category (medical, financial, timeline, etc.)

        Args:
            category: Category to summarize

        Returns:
            Category-specific summary
        """
        logger.info(f"Generating summary for category: {category}")

        category_queries = {
            "medical": "Summarize all medical aspects: injuries, treatments, providers, and medical costs.",
            "financial": "Summarize all financial aspects: repair costs, medical costs, deductibles, total claim amount.",
            "timeline": "Provide a detailed chronological timeline of events.",
            "parties": "List and describe all parties involved: policyholder, at-fault parties, witnesses, adjusters, providers.",
            "damages": "Summarize vehicle damage, repair process, and associated costs."
        }

        query = category_queries.get(
            category.lower(),
            f"Provide a summary of {category} aspects of this insurance claim."
        )

        return self.query(query)


if __name__ == "__main__":
    print("SummarizationAgent module - use via main system")
