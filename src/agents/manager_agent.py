"""
Manager (Router) Agent
Routes queries to appropriate specialist agents based on query type
"""

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManagerAgent:
    """
    Manager/Router Agent that analyzes queries and routes to appropriate tools/agents

    Responsibilities:
    - Classify query type (summary vs precise fact vs hybrid)
    - Select appropriate retrieval tool(s)
    - Coordinate MCP tool usage
    - Synthesize results if multiple tools used
    """

    def __init__(
        self,
        tools: List[Tool],
        llm_model: str = "gpt-4",
        temperature: float = 0
    ):
        """
        Initialize Manager Agent

        Args:
            tools: List of available tools
            llm_model: LLM model to use
            temperature: Temperature setting
        """
        self.tools = tools
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        logger.info(f"ManagerAgent initialized with {len(tools)} tools")
        logger.info(f"Available tools: {[t.name for t in tools]}")

        # Create agent
        self.agent_executor = self._create_agent()

    def _create_agent(self):
        """Create the LangGraph agent with routing logic"""

        system_prompt = """You are a helpful assistant that answers questions about insurance claims.

You have access to tools to retrieve information from the claim documents. Choose the BEST tool based on the question type:

RETRIEVAL TOOLS (choose carefully):
- SummaryRetriever: ONLY for broad narrative overviews and "what happened" questions. NOT for specific topics.
- NeedleRetriever: For specific facts like dates, amounts, names, exact numbers, precise details.
- SectionRetriever: For questions about specific TOPICS. Format: "SECTION|question"
  Use this for questions about: medical treatment, witnesses, police report, damages, financial details, policy info.
  Sections: MEDICAL DOCUMENTATION, WITNESS STATEMENTS, POLICE REPORT, VEHICLE DAMAGE ASSESSMENT, FINANCIAL SUMMARY, POLICY INFORMATION

TOOL SELECTION GUIDE:
- "Summarize the medical treatment" → SectionRetriever with "MEDICAL DOCUMENTATION|medical treatment summary"
- "Who were the witnesses" → SectionRetriever with "WITNESS STATEMENTS|who were the witnesses"
- "What is this claim about?" → SummaryRetriever
- "What was the deductible?" → NeedleRetriever
- Questions about a specific topic (medical, witnesses, damages) → SectionRetriever FIRST

MCP TOOLS (for computations and metadata):
- GetDocumentMetadata: Get claim metadata (filing date, status, adjuster, policyholder info). Input: claim_id e.g. "CLM-2024-001"
- CalculateDaysBetween: Calculate days between two dates. Input: "YYYY-MM-DD,YYYY-MM-DD" e.g. "2024-01-12,2024-01-15"
- EstimateCoveragePayout: Calculate insurance payout. Input: "damage_amount,deductible" e.g. "17111.83,750.00"
- ValidateClaimStatus: Check if claim status is normal. Input: "filed_date,status" e.g. "2024-01-15,Under Review"
- GetTimelineSummary: Get timeline milestones for a claim. Input: claim_id e.g. "CLM-2024-001"

IMPORTANT INSTRUCTIONS:
1. Always use a tool to get information before answering.
2. Include SPECIFIC DETAILS in your answer: dates, names, amounts, locations when available.
3. For summary questions, include key facts: claim ID, date, parties involved, amounts.
4. If the first tool doesn't return useful information, try a different tool.
5. After getting the tool result, provide a comprehensive answer with specific facts from the retrieved information.

Do NOT output code or function call syntax. Simply use the tools and then respond with the answer."""

        return create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=system_prompt
        )

    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the manager agent

        Args:
            query: User query

        Returns:
            Dictionary with output and metadata
        """
        logger.info(f"ManagerAgent processing query: '{query[:100]}...'")

        try:
            # LangGraph uses messages format
            result = self.agent_executor.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )

            # Extract the final response from messages
            messages = result.get("messages", [])
            output = ""

            if messages:
                # Get the last AI message that has actual text content (not tool calls)
                from langchain_core.messages import AIMessage, ToolMessage

                for msg in reversed(messages):
                    # Skip tool messages
                    if isinstance(msg, ToolMessage):
                        continue

                    # For AI messages, check if it has content and no tool calls
                    if isinstance(msg, AIMessage):
                        # If this message has tool_calls, it's not the final answer
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            continue
                        # If it has text content, this is likely the final answer
                        if msg.content and isinstance(msg.content, str) and msg.content.strip():
                            output = msg.content
                            break

                    # Fallback for other message types
                    elif hasattr(msg, 'content') and msg.content:
                        if isinstance(msg.content, str) and msg.content.strip():
                            output = msg.content
                            break

            # If no good output found, try to get any content
            if not output and messages:
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        output = str(msg.content)
                        break

            # Log message types for debugging
            logger.info(f"Messages received: {len(messages)}")
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                content_preview = str(msg.content)[:100] if hasattr(msg, 'content') and msg.content else 'None'
                logger.info(f"  [{i}] {msg_type}: tool_calls={has_tool_calls}, content={content_preview}...")

            return {
                "query": query,
                "output": output,
                "messages": messages,
                "success": True
            }

        except Exception as e:
            logger.error(f"ManagerAgent error: {e}")
            return {
                "query": query,
                "output": f"Error processing query: {str(e)}",
                "error": str(e),
                "success": False
            }

    def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries

        Args:
            queries: List of queries

        Returns:
            List of results
        """
        results = []
        for query in queries:
            result = self.query(query)
            results.append(result)

        return results


if __name__ == "__main__":
    print("ManagerAgent module - use via main system")
