"""
LangChain Integration Layer
Bridges LlamaIndex/ChromaDB with LangChain agents
"""

from langchain_core.tools import Tool
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainIntegration:
    """
    Integration layer between LlamaIndex retrievers and LangChain agents
    Wraps retrieval functions as LangChain Tools
    """

    def __init__(
        self,
        summary_index,
        hierarchical_retriever,
        mcp_tools: List[Tool] = None,
        retrieval_k: int = 5
    ):
        """
        Initialize integration layer

        Args:
            summary_index: LlamaIndex DocumentSummaryIndex
            hierarchical_retriever: HierarchicalRetriever instance
            mcp_tools: List of MCP tools (optional)
            retrieval_k: Number of chunks to retrieve (default: 5)
        """
        self.summary_index = summary_index
        self.hier_retriever = hierarchical_retriever
        self.mcp_tools = mcp_tools or []
        self.retrieval_k = retrieval_k

        logger.info(f"LangChain integration layer initialized (k={retrieval_k})")

    def create_summary_tool(self) -> Tool:
        """
        Create LangChain tool for summary queries

        Returns:
            Tool for querying summary index
        """
        def query_summary(query: str) -> str:
            """Query summary index for high-level information"""
            logger.info(f"Summary tool called with query: '{query[:50]}...'")

            try:
                query_engine = self.summary_index.as_query_engine(
                    response_mode="tree_summarize",
                    verbose=True
                )
                response = query_engine.query(query)
                return str(response)
            except Exception as e:
                logger.error(f"Summary tool error: {e}")
                return f"Error querying summary index: {str(e)}"

        return Tool(
            name="SummaryRetriever",
            func=query_summary,
            description="""Use this tool ONLY for broad narrative overviews of the entire claim:
            - "What is this claim about?" (overall summary)
            - "What happened?" (general narrative)
            - "Summarize the entire claim"

            DO NOT use for specific topics like medical treatment, witnesses, or damages.
            For those, use SectionRetriever with the appropriate section name.

            Input: Your question as a string.
            The answer should include specific facts: claim ID, dates, names, amounts."""
        )

    def create_needle_tool(self) -> Tool:
        """
        Create LangChain tool for needle (precise fact) queries

        Returns:
            Tool for precise fact retrieval
        """
        def query_needle(query: str) -> str:
            """Query hierarchical index for precise details"""
            logger.info(f"Needle tool called with query: '{query[:50]}...' (k={self.retrieval_k})")

            try:
                # Use needle_search for precision
                results = self.hier_retriever.needle_search(query, k=self.retrieval_k)

                if not results:
                    return "No specific information found. Try rephrasing your question."

                # Format results
                context = self.hier_retriever.get_retrieval_context(results, include_metadata=True)
                return context

            except Exception as e:
                logger.error(f"Needle tool error: {e}")
                return f"Error in precise search: {str(e)}"

        return Tool(
            name="NeedleRetriever",
            func=query_needle,
            description="""Use this tool for specific, precise factual questions including:
            - Exact dates and times
            - Specific dollar amounts (deductibles, repair costs, etc.)
            - Names of people
            - Specific locations
            - Precise details (policy numbers, VINs, etc.)
            - Direct quotes or statements

            Examples: 'What was the exact deductible?', 'When did the accident occur?',
            'What was the repair cost?', 'Who was the claims adjuster?'

            Input: Your precise question as a string"""
        )

    def create_section_tool(self) -> Tool:
        """
        Create tool for querying specific document sections

        Returns:
            Tool for section-specific queries
        """
        def query_section(section_and_query: str) -> str:
            """Query specific document section"""
            logger.info(f"Section tool called with: '{section_and_query[:50]}...' (k={self.retrieval_k})")

            try:
                # Parse input: "section_name|query"
                if "|" not in section_and_query:
                    return "Format: 'section_name|your_question'. Available sections: POLICY INFORMATION, INCIDENT TIMELINE, WITNESS STATEMENTS, POLICE REPORT, MEDICAL DOCUMENTATION, VEHICLE DAMAGE ASSESSMENT, RENTAL CAR DOCUMENTATION, FINANCIAL SUMMARY"

                section_name, query = section_and_query.split("|", 1)
                section_name = section_name.strip()
                query = query.strip()

                # Query specific section
                results = self.hier_retriever.retrieve_by_section(
                    query=query,
                    section_title=section_name,
                    k=self.retrieval_k
                )

                if not results:
                    return f"No information found in section '{section_name}'. Check section name spelling."

                context = self.hier_retriever.get_retrieval_context(results, include_metadata=True)
                return context

            except Exception as e:
                logger.error(f"Section tool error: {e}")
                return f"Error querying section: {str(e)}"

        return Tool(
            name="SectionRetriever",
            func=query_section,
            description="""Use this tool to query a specific section of the claim document.

            Input format: 'section_name|your_question'

            Available sections:
            - POLICY INFORMATION
            - INCIDENT TIMELINE
            - WITNESS STATEMENTS
            - POLICE REPORT
            - MEDICAL DOCUMENTATION
            - VEHICLE DAMAGE ASSESSMENT
            - RENTAL CAR DOCUMENTATION
            - FINANCIAL SUMMARY

            Example: 'WITNESS STATEMENTS|What did the witnesses see?'
            """
        )

    def get_all_tools(self) -> List[Tool]:
        """
        Get all tools including retrieval and MCP tools

        Returns:
            List of all available tools
        """
        tools = [
            self.create_summary_tool(),
            self.create_needle_tool(),
            self.create_section_tool()
        ]

        # Add MCP tools
        tools.extend(self.mcp_tools)

        logger.info(f"Created {len(tools)} tools for agents")
        return tools

    def get_retrieval_tools(self) -> List[Tool]:
        """Get only retrieval tools (no MCP)"""
        return [
            self.create_summary_tool(),
            self.create_needle_tool(),
            self.create_section_tool()
        ]


if __name__ == "__main__":
    print("LangChainIntegration module - use via agent system")
