"""
Hierarchical Retriever with Auto-Merging and Metadata Filtering
Supports fine-grained retrieval starting from small chunks and merging upward
"""

from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    """
    Advanced retrieval with hierarchical auto-merging and metadata filtering

    Features:
    - Start with small chunks for precision
    - Auto-merge to parent chunks when context needed
    - Filter by section, chunk level, date range
    - Support for needle-in-haystack queries
    """

    def __init__(self, hierarchical_index: VectorStoreIndex, nodes: List[TextNode]):
        """
        Initialize hierarchical retriever

        Args:
            hierarchical_index: The hierarchical vector store index
            nodes: List of all hierarchical nodes
        """
        self.index = hierarchical_index
        self.nodes = nodes
        self.storage_context = hierarchical_index.storage_context

        logger.info(f"HierarchicalRetriever initialized with {len(nodes)} nodes")

    def retrieve(
        self,
        query: str,
        chunk_level: Optional[str] = "small",
        k: int = 5,
        auto_merge: bool = True
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant chunks with optional auto-merging

        Args:
            query: Search query
            chunk_level: "small", "medium", "large", or None for all levels
            k: Number of results to retrieve
            auto_merge: Whether to use auto-merging retriever

        Returns:
            List of NodeWithScore objects
        """
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        logger.info(f"  Chunk level: {chunk_level}, k={k}, auto_merge={auto_merge}")

        # Build metadata filters if chunk level specified
        filters = None
        if chunk_level:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="chunk_level",
                        value=chunk_level,
                        operator=FilterOperator.EQ
                    )
                ]
            )

        # Create base retriever
        base_retriever = self.index.as_retriever(
            similarity_top_k=k,
            filters=filters
        )

        # Use auto-merging if requested
        if auto_merge:
            retriever = AutoMergingRetriever(
                base_retriever,
                self.storage_context,
                simple_ratio_thresh=0.5,  # Merge to parent if >50% of children retrieved
                verbose=True
            )
            results = retriever.retrieve(query)
        else:
            results = base_retriever.retrieve(query)

        logger.info(f"Retrieved {len(results)} results")
        return results

    def retrieve_by_section(
        self,
        query: str,
        section_title: str,
        k: int = 5
    ) -> List[NodeWithScore]:
        """
        Retrieve from a specific document section with fallback

        Args:
            query: Search query
            section_title: Title of section to search within
            k: Number of results

        Returns:
            List of NodeWithScore objects
        """
        logger.info(f"Retrieving from section: {section_title}")

        # First try exact match with EQ operator
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="section_title",
                    value=section_title,
                    operator=FilterOperator.EQ
                )
            ]
        )

        retriever = self.index.as_retriever(
            similarity_top_k=k,
            filters=filters
        )

        results = retriever.retrieve(query)

        # If no results, try case-insensitive partial match by searching without filter
        # and post-filtering results
        if not results:
            logger.info(f"No exact match for section '{section_title}', trying partial match...")
            # Retrieve more results without filter
            unfiltered_retriever = self.index.as_retriever(similarity_top_k=k * 3)
            all_results = unfiltered_retriever.retrieve(query)

            # Post-filter by section title (case-insensitive partial match)
            section_lower = section_title.lower()
            results = [
                r for r in all_results
                if section_lower in r.node.metadata.get('section_title', '').lower()
            ][:k]

            if results:
                logger.info(f"Found {len(results)} results with partial section match")

        # Final fallback: just return regular search results
        if not results:
            logger.info(f"No section matches found, falling back to regular search")
            fallback_retriever = self.index.as_retriever(similarity_top_k=k)
            results = fallback_retriever.retrieve(query)

        logger.info(f"Retrieved {len(results)} results for section '{section_title}'")
        return results

    def retrieve_by_doc_type(
        self,
        query: str,
        doc_type: str,
        k: int = 3,
        chunk_level: Optional[str] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve from specific document type

        Args:
            query: Search query
            doc_type: Document type (e.g., 'timeline', 'medical_documentation')
            k: Number of results
            chunk_level: Optional chunk level filter

        Returns:
            List of NodeWithScore objects
        """
        logger.info(f"Retrieving from doc_type: {doc_type}")

        filter_list = [
            MetadataFilter(
                key="doc_type",
                value=doc_type,
                operator=FilterOperator.EQ
            )
        ]

        if chunk_level:
            filter_list.append(
                MetadataFilter(
                    key="chunk_level",
                    value=chunk_level,
                    operator=FilterOperator.EQ
                )
            )

        filters = MetadataFilters(filters=filter_list)

        retriever = self.index.as_retriever(
            similarity_top_k=k,
            filters=filters
        )

        results = retriever.retrieve(query)
        logger.info(f"Retrieved {len(results)} results from doc_type '{doc_type}'")
        return results

    def retrieve_by_date_range(
        self,
        query: str,
        start_date: str,
        end_date: str,
        k: int = 3
    ) -> List[NodeWithScore]:
        """
        Retrieve chunks within a specific date range

        Args:
            query: Search query
            start_date: Start date (format: "January 12, 2024" or "2024-01-12")
            end_date: End date
            k: Number of results

        Returns:
            List of NodeWithScore objects
        """
        logger.info(f"Retrieving for date range: {start_date} to {end_date}")

        # For date range filtering, we'll retrieve without date filter
        # and post-process (since ChromaDB date comparisons can be tricky)
        retriever = self.index.as_retriever(similarity_top_k=k * 2)
        all_results = retriever.retrieve(query)

        # Filter by date in post-processing
        filtered_results = []
        for result in all_results:
            timestamp = result.node.metadata.get('timestamp', '')
            if timestamp and (start_date in timestamp or end_date in timestamp):
                filtered_results.append(result)

        filtered_results = filtered_results[:k]
        logger.info(f"Retrieved {len(filtered_results)} results in date range")
        return filtered_results

    def needle_search(
        self,
        query: str,
        k: int = 5
    ) -> List[NodeWithScore]:
        """
        Specialized retrieval for needle-in-haystack queries
        Focuses on small chunks for maximum precision

        Args:
            query: Specific factual query
            k: Number of results

        Returns:
            List of NodeWithScore objects
        """
        logger.info(f"Needle search for: '{query[:50]}...'")

        # Start with small chunks for precision
        results = self.retrieve(
            query=query,
            chunk_level="small",
            k=k,
            auto_merge=False  # Don't merge for precision
        )

        # If not enough results, try medium chunks
        if len(results) < 2:
            logger.info("Not enough results in small chunks, trying medium...")
            results = self.retrieve(
                query=query,
                chunk_level="medium",
                k=k,
                auto_merge=False
            )

        return results

    def get_retrieval_context(
        self,
        nodes: List[NodeWithScore],
        include_metadata: bool = True
    ) -> str:
        """
        Convert retrieved nodes to context string

        Args:
            nodes: Retrieved nodes with scores
            include_metadata: Whether to include metadata in output

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, node_with_score in enumerate(nodes, 1):
            node = node_with_score.node
            score = node_with_score.score

            part = f"\n--- Result {i} (Score: {score:.4f}) ---\n"

            if include_metadata:
                metadata = node.metadata
                part += f"Section: {metadata.get('section_title', 'N/A')}\n"
                part += f"Chunk Level: {metadata.get('chunk_level', 'N/A')}\n"
                part += f"Document Type: {metadata.get('doc_type', 'N/A')}\n"
                part += "\n"

            part += node.text
            context_parts.append(part)

        return "\n".join(context_parts)


if __name__ == "__main__":
    print("HierarchicalRetriever module - use via main system")
