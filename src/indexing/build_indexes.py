"""
Index Builders for Insurance Claim System
Creates Summary Index and Hierarchical Index with ChromaDB backend
"""

from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    ServiceContext,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from llama_index.core.schema import TextNode
from typing import List, Tuple
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class IndexBuilder:
    """
    Builds both Summary and Hierarchical indexes using LlamaIndex with ChromaDB
    """

    def __init__(self, vector_store_manager, llm_model: str = "gpt-4o-mini", temperature: float = 0):  # Changed from gpt-4
        """
        Initialize index builder

        Args:
            vector_store_manager: VectorStoreManager instance
            llm_model: LLM model to use for summarization
            temperature: LLM temperature setting
        """
        self.vsm = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature

        # Configure LlamaIndex settings globally
        Settings.llm = OpenAI(model=llm_model, temperature=temperature)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.chunk_size = 512
        Settings.chunk_overlap = 102

        logger.info(f"IndexBuilder initialized with model: {llm_model}")

    def build_summary_index(
        self,
        documents: List[Document],
        response_mode: str = "tree_summarize"
    ) -> DocumentSummaryIndex:
        """
        Build summary index using MapReduce approach (tree_summarize)

        MapReduce Strategy:
        1. MAP: Each document chunk is summarized individually
        2. REDUCE: Section summaries are combined into document-level summaries
        3. Final summary includes timeline, key entities, and overall context

        Args:
            documents: List of Document objects
            response_mode: Summarization strategy ('tree_summarize' = MapReduce)

        Returns:
            DocumentSummaryIndex stored in ChromaDB
        """
        logger.info("Building Summary Index with MapReduce strategy...")

        # Get ChromaDB collection for summaries
        summary_collection = self.vsm.get_summary_collection()
        storage_context = self.vsm.create_storage_context(summary_collection)

        # Build DocumentSummaryIndex
        # This automatically creates summaries for each document using the specified strategy
        summary_index = DocumentSummaryIndex.from_documents(
            documents,
            storage_context=storage_context,
            response_mode=response_mode,  # tree_summarize = MapReduce
            show_progress=True,
            use_async=False
        )

        # Enhance document metadata
        for doc_id, node in summary_index.docstore.docs.items():
            if isinstance(node, Document):
                node.metadata.update({
                    'index_type': 'summary',
                    'has_summary': True,
                    'summary_strategy': response_mode
                })

        logger.info(f"Summary Index built with {len(summary_index.docstore.docs)} documents")

        return summary_index

    def build_hierarchical_index(
        self,
        nodes: List[TextNode]
    ) -> Tuple[VectorStoreIndex, List[TextNode]]:
        """
        Build hierarchical index for Auto-Merging Retriever

        The hierarchical structure supports:
        - Starting with small chunks for precision
        - Auto-merging up to parent chunks when more context is needed

        Args:
            nodes: Hierarchical nodes from HierarchicalChunker

        Returns:
            Tuple of (VectorStoreIndex, nodes)
        """
        logger.info("Building Hierarchical Index with Auto-Merging support...")

        # Get ChromaDB collection for hierarchical chunks
        hier_collection = self.vsm.get_hierarchical_collection()
        storage_context = self.vsm.create_storage_context(hier_collection)

        # Build VectorStoreIndex from hierarchical nodes
        hierarchical_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
            use_async=False
        )

        logger.info(f"Hierarchical Index built with {len(nodes)} chunks")

        # Log chunk distribution
        self._log_hierarchical_stats(nodes)

        return hierarchical_index, nodes

    def _log_hierarchical_stats(self, nodes: List[TextNode]):
        """Log statistics about hierarchical index"""
        stats = {
            'large': 0,
            'medium': 0,
            'small': 0,
            'total_tokens': 0
        }

        for node in nodes:
            level = node.metadata.get('chunk_level', 'unknown')
            if level in stats:
                stats[level] += 1
            stats['total_tokens'] += len(node.text) // 4  # Rough token estimate

        logger.info("=== Hierarchical Index Stats ===")
        logger.info(f"  Large chunks: {stats['large']}")
        logger.info(f"  Medium chunks: {stats['medium']}")
        logger.info(f"  Small chunks: {stats['small']}")
        logger.info(f"  Estimated tokens: {stats['total_tokens']:,}")

    def rebuild_indexes(self, documents: List[Document], nodes: List[TextNode]) -> Tuple:
        """
        Rebuild both indexes (useful for updates)

        Args:
            documents: Source documents for summary index
            nodes: Hierarchical nodes for hierarchical index

        Returns:
            Tuple of (summary_index, hierarchical_index, nodes)
        """
        logger.info("Rebuilding all indexes...")

        # Reset collections
        self.vsm.reset_collections()

        # Rebuild both indexes
        summary_index = self.build_summary_index(documents)
        hierarchical_index, nodes = self.build_hierarchical_index(nodes)

        logger.info("All indexes rebuilt successfully")

        return summary_index, hierarchical_index, nodes


def create_indexes_from_documents(
    documents: List[Document],
    vector_store_manager,
    hierarchical_nodes: List[TextNode] = None
) -> Tuple:
    """
    Convenience function to create both indexes from documents

    Args:
        documents: Source documents
        vector_store_manager: VectorStoreManager instance
        hierarchical_nodes: Pre-chunked hierarchical nodes (optional)

    Returns:
        Tuple of (summary_index, hierarchical_index, nodes)
    """
    builder = IndexBuilder(vector_store_manager)

    # Build summary index
    summary_index = builder.build_summary_index(documents)

    # Build hierarchical index
    if hierarchical_nodes is None:
        from .chunking import HierarchicalChunker
        chunker = HierarchicalChunker()
        hierarchical_nodes = chunker.chunk_documents(documents)

    hierarchical_index, nodes = builder.build_hierarchical_index(hierarchical_nodes)

    return summary_index, hierarchical_index, nodes


if __name__ == "__main__":
    from ..vector_store.setup import VectorStoreManager
    from .document_loader import InsuranceClaimLoader
    from .chunking import HierarchicalChunker

    print("\n=== Index Builder Test ===\n")

    # Initialize components
    vsm = VectorStoreManager(persist_dir="../../chroma_db")
    loader = InsuranceClaimLoader(data_dir="../../data")
    chunker = HierarchicalChunker()

    # Load documents
    documents = loader.load_all_documents()
    print(f"Loaded {len(documents)} document sections")

    # Create hierarchical chunks
    nodes = chunker.chunk_documents(documents)
    print(f"Created {len(nodes)} hierarchical chunks")

    # Build indexes
    builder = IndexBuilder(vsm)

    print("\nBuilding Summary Index...")
    summary_index = builder.build_summary_index(documents[:3])  # Test with first 3

    print("\nBuilding Hierarchical Index...")
    hierarchical_index, _ = builder.build_hierarchical_index(nodes)

    print("\n=== Indexes built successfully ===")

    # Show stats
    stats = vsm.get_collection_stats()
    print("\nCollection Stats:")
    for name, info in stats.items():
        print(f"  {name}: {info}")
