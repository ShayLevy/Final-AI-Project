"""
Streamlit Interface for Insurance Claim Timeline Retrieval System
Workflow: Upload PDF → Preview Chunks → Index to ChromaDB → Query
"""

import streamlit as st
import os
import sys
import logging
import shutil
import time
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
# This ensures OPENAI_API_KEY and ANTHROPIC_API_KEY are available
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.indexing.document_loader import InsuranceClaimLoader
from src.indexing.chunking import HierarchicalChunker
from src.evaluation.regression import RegressionTracker

# Page config
st.set_page_config(
    page_title="Insurance Claim RAG System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scroll to top if flagged (after indexing completes)
if st.session_state.get('scroll_to_top', False):
    st.session_state.scroll_to_top = False
    # Add anchor at the very top for scrolling
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    js_scroll = """
    <script>
        const tryScroll = () => {
            try {
                // Find the title element and scroll to it
                const title = window.parent.document.querySelector('h1');
                if (title) {
                    title.scrollIntoView({behavior: 'instant', block: 'start'});
                    return;
                }
                // Fallback: scroll the main container
                const main = window.parent.document.querySelector('[data-testid="stMain"]');
                if (main) {
                    main.scrollTo(0, 0);
                }
            } catch(e) {
                console.log('Scroll error:', e);
            }
        };
        // Try immediately and after delays
        tryScroll();
        setTimeout(tryScroll, 50);
        setTimeout(tryScroll, 200);
    </script>
    """
    st.components.v1.html(js_scroll, height=0)

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding from main content */
    .block-container {
        padding-top: 1rem !important;
    }
    /* Remove top padding from sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0 !important;
    }
    /* Hide sidebar collapse button completely */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="baseButton-header"],
    [data-testid="stSidebarNavCollapseIcon"],
    button[kind="header"],
    button[kind="headerNoPadding"],
    .stSidebarCollapse,
    div[data-testid="stSidebarCollapsedControl"],
    section[data-testid="stSidebar"] > button,
    [data-testid="stSidebar"] > div:first-child > button,
    .st-emotion-cache-1gwvy71 {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    .status-box {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
    }
    .status-no-db {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .status-has-db {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .chunk-card {
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .step-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 20px 0 10px 0;
    }
    /* Disable clear button and text editing in selectboxes */
    [data-baseweb="select"] [data-testid="stSelectboxClearIcon"],
    [data-baseweb="select"] svg[title="Clear"],
    [data-baseweb="select"] [aria-label="Clear all"],
    [data-baseweb="select"] .css-1dimb5e-indicatorContainer {
        display: none !important;
        pointer-events: none !important;
    }
    [data-baseweb="select"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
        user-select: none !important;
        -webkit-user-select: none !important;
    }
    /* Make selectbox input non-editable */
    div[data-baseweb="select"] input[aria-autocomplete="list"] {
        pointer-events: none !important;
        user-select: none !important;
    }
    /* Hide the clear button container */
    [data-baseweb="select"] > div > div:last-child > div:first-child {
        display: none !important;
    }

    /* ========== TAB-STYLE RADIO BUTTONS ========== */
    /* Tab container styling */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        gap: 8px;
        background-color: transparent;
        padding: 0;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Individual tab buttons */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background-color: transparent;
        border-radius: 0;
        padding: 8px 20px !important;
        font-weight: 400;
        font-size: 14px;
        color: #666;
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        cursor: pointer;
        transition: all 0.2s ease;
        border-right: 1px solid #ddd;
    }

    /* Remove separator from last tab */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:last-child {
        border-right: none;
    }

    /* Tab hover state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        color: #333;
    }

    /* Active/selected tab */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
        background-color: transparent !important;
        color: #667eea !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #667eea !important;
    }

    /* Flashing warning animation */
    @keyframes flash-warning {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .flash-warning {
        animation: flash-warning 1.5s ease-in-out infinite;
    }

</style>
""", unsafe_allow_html=True)


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that captures logs for Streamlit display"""
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'level': record.levelname,
            'message': log_entry
        })

    def get_logs(self):
        return self.logs

    def clear(self):
        self.logs = []


# Initialize session state
if 'log_handler' not in st.session_state:
    st.session_state.log_handler = StreamlitLogHandler()
    st.session_state.log_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
if 'chunks_preview' not in st.session_state:
    st.session_state.chunks_preview = None
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'system' not in st.session_state:
    st.session_state.system = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Chunk size defaults
DEFAULT_CHUNK_SIZES = {
    'large': 2048,
    'medium': 512,
    'small': 128
}
DEFAULT_OVERLAP_RATIO = 0.2

# Evaluation history file path
EVAL_HISTORY_FILE = Path("./evaluation_results/evaluation_history.json")


def load_evaluation_history():
    """Load evaluation history from file"""
    if EVAL_HISTORY_FILE.exists():
        try:
            with open(EVAL_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_evaluation_history(history):
    """Save evaluation history to file"""
    EVAL_HISTORY_FILE.parent.mkdir(exist_ok=True)
    try:
        with open(EVAL_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    except IOError as e:
        st.warning(f"Could not save evaluation history: {e}")


def setup_logging():
    """Setup logging to capture to our handler"""
    # Only add handler to root logger to avoid duplicate logs from propagation
    root_logger = logging.getLogger()
    if st.session_state.log_handler not in root_logger.handlers:
        root_logger.addHandler(st.session_state.log_handler)
    root_logger.setLevel(logging.INFO)

    # Set log levels for specific loggers without adding duplicate handlers
    for logger_name in ['src.indexing', 'src.vector_store', 'src.agents', 'src.retrieval', 'src.mcp', 'httpx', 'main']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)


def check_chroma_exists():
    """Check if ChromaDB exists and has data"""
    import chromadb
    from chromadb.config import Settings

    chroma_dir = Path("./chroma_db")
    abs_path = str(chroma_dir.absolute())

    if not chroma_dir.exists():
        return {
            "exists": False,
            "status": "No Vector Database Found",
            "path": abs_path,
            "message": "Upload a PDF to create the index",
            "collections": []
        }

    try:
        # Use chromadb directly with same settings as VectorStoreManager
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Check collections
        try:
            summary_col = client.get_collection("insurance_summaries")
            summary_count = summary_col.count()
        except:
            summary_count = 0

        try:
            hier_col = client.get_collection("insurance_hierarchical")
            hierarchical_count = hier_col.count()
        except:
            hierarchical_count = 0

        # Get last modified time
        import os
        mtime = os.path.getmtime(chroma_dir)
        last_modified = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

        if summary_count == 0 and hierarchical_count == 0:
            return {
                "exists": False,
                "status": "Database Empty",
                "path": abs_path,
                "message": "No indexes found. Upload a PDF to create indexes.",
                "collections": []
            }

        return {
            "exists": True,
            "status": "Connected",
            "path": abs_path,
            "last_modified": last_modified,
            "collections": [
                {"name": "Summary Index", "count": summary_count},
                {"name": "Hierarchical Index", "count": hierarchical_count}
            ]
        }
    except Exception as e:
        return {
            "exists": False,
            "status": f"Connection Error",
            "path": abs_path,
            "message": str(e),
            "collections": []
        }


def delete_chroma():
    """Delete ChromaDB directory completely and clear evaluation history"""
    import chromadb
    import gc

    chroma_dir = Path("./chroma_db")

    # Reset system state first
    st.session_state.system = None

    # Clear ChromaDB client cache
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except:
        pass

    gc.collect()

    # Delete evaluation history and results
    if EVAL_HISTORY_FILE.exists():
        try:
            EVAL_HISTORY_FILE.unlink()
        except:
            pass
    # Clear session state evaluation history and results
    if 'evaluation_history' in st.session_state:
        st.session_state.evaluation_history = []
    if 'ragas_results' in st.session_state:
        st.session_state.ragas_results = None
    if 'judge_results' in st.session_state:
        st.session_state.judge_results = None

    if chroma_dir.exists():
        try:
            # Remove directory
            shutil.rmtree(chroma_dir, ignore_errors=True)

            # Double check it's gone, if not try again
            if chroma_dir.exists():
                time.sleep(0.5)
                shutil.rmtree(chroma_dir, ignore_errors=True)

            return True
        except Exception as e:
            st.error(f"Error deleting database: {e}")
            return False
    return False


def load_pdf(uploaded_file):
    """Load uploaded PDF and return documents"""
    # Ensure data directory exists
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Save uploaded file
    file_path = data_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load with our loader
    loader = InsuranceClaimLoader(data_dir=str(data_dir))
    documents = loader.load_document(str(file_path))

    return documents, file_path


def preview_chunks(documents, chunk_sizes, overlap_ratio):
    """Generate chunk preview without indexing"""
    chunker = HierarchicalChunker(
        chunk_sizes=chunk_sizes,
        chunk_overlap_ratio=overlap_ratio
    )
    nodes = chunker.chunk_documents(documents)

    chunks_info = []
    for i, node in enumerate(nodes):
        # Collect all metadata
        metadata = dict(node.metadata)

        # Parse page_numbers from string back to list for display
        page_nums_str = node.metadata.get('page_numbers', '1')
        pages = [int(p) for p in page_nums_str.split(',') if p] if page_nums_str else [1]

        chunks_info.append({
            'id': i + 1,
            'node_id': node.id_,
            'level': node.metadata.get('chunk_level', 'unknown'),
            'pages': pages,
            'start_page': node.metadata.get('start_page', 1),
            'end_page': node.metadata.get('end_page', 1),
            'size': node.metadata.get('chunk_size', 0),
            'char_count': len(node.text),
            'text': node.text,
            'preview': node.text[:300] + '...' if len(node.text) > 300 else node.text,
            'metadata': metadata
        })

    return chunks_info, nodes


def index_documents(documents, chunk_sizes, overlap_ratio):
    """Index documents into ChromaDB"""
    from main import InsuranceClaimSystem
    import gc
    import chromadb

    st.session_state.log_handler.clear()
    setup_logging()

    # Ensure clean state - clear all references
    st.session_state.system = None

    # Reset ChromaDB client cache
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    gc.collect()
    time.sleep(0.5)

    # Force delete existing DB
    chroma_dir = Path("./chroma_db")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
        time.sleep(0.5)  # Wait for filesystem
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)  # Try again without ignore_errors

    # Create system (this will index)
    system = InsuranceClaimSystem(
        data_dir="./data",
        chroma_dir="./chroma_db",
        rebuild_indexes=True
    )

    st.session_state.system = system
    return system


def run_query(query: str, top_k: int = 5):
    """Run a query through the system"""
    if not st.session_state.system:
        return {"error": "System not initialized", "success": False}

    st.session_state.log_handler.clear()
    setup_logging()

    # Update retrieval k value if system supports it
    if hasattr(st.session_state.system, 'update_retrieval_k'):
        st.session_state.system.update_retrieval_k(top_k)

    start_time = time.time()
    result = st.session_state.system.query(query)
    elapsed = time.time() - start_time

    result['elapsed_time'] = f"{elapsed:.2f}s"
    result['top_k'] = top_k
    result['logs'] = st.session_state.log_handler.get_logs()

    st.session_state.query_history.append({
        'query': query,
        'result': result,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

    return result


# =============================================================================
# MAIN UI
# =============================================================================

st.title("Insurance Claim RAG System")

# Check database status
db_status = check_chroma_exists()

# =============================================================================
# SIDEBAR - Status & Controls
# =============================================================================
with st.sidebar:
    # Logo at the top (full width, no margins)
    logo_path = Path("./logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
        st.divider()

    if db_status['exists']:
        # ChromaDB Status header above Indexes
        status_icon = "🟢"
        tooltip_text = db_status.get('path', 'No database')
        st.markdown(f'<h3 style="margin-bottom: 0.5rem; cursor: help;" title="{tooltip_text}">ChromaDB Status {status_icon}</h3>', unsafe_allow_html=True)

        st.subheader("Indexes")
        for col in db_status['collections']:
            st.markdown(f"**{col['name']}**: {col['count']} items")

        st.divider()
        if st.button("📄 Upload New Document", type="secondary", use_container_width=True):
            delete_chroma()
            st.session_state.chunks_preview = None
            st.session_state.documents = None
            st.session_state.uploaded_file_name = None
            st.rerun()

        if st.button("🗑️ Delete Database", type="secondary", use_container_width=True):
            delete_chroma()
            st.session_state.chunks_preview = None
            st.session_state.documents = None
            st.session_state.system = None
            st.rerun()
    else:
        # ChromaDB Status (not connected)
        st.markdown('<h3 style="margin-bottom: 0.5rem;">ChromaDB Status 🔴</h3>', unsafe_allow_html=True)
        st.caption("No database found")

        st.markdown("""
        <p style="font-weight: 600; font-size: 16px; margin-bottom: 5px;">Workflow</p>
        <div style="line-height: 1.4; font-size: 14px;">
        1. 📄 Upload PDF<br>
        2. ⚙️ Configure & Preview Chunks<br>
        3. 💾 Index to Database<br>
        4. 🔍 Query the System
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT - Workflow Steps
# =============================================================================

if not db_status['exists']:
    # ==========================================================================
    # STEP 1: Upload PDF
    # ==========================================================================
    st.markdown('<div class="step-header"><h3 style="margin:0;">Step 1: Upload PDF Document</h3></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload an insurance claim document to process"
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        # Load the PDF (reload if file changed OR if documents are None due to session reset)
        if st.session_state.uploaded_file_name != uploaded_file.name or st.session_state.documents is None:
            with st.spinner("Loading PDF..."):
                documents, file_path = load_pdf(uploaded_file)
                st.session_state.documents = documents
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chunks_preview = None  # Reset preview
            st.success(f"Loaded {len(documents)} document section(s)")

        # ==========================================================================
        # STEP 2: Configure Chunk Sizes & Preview
        # ==========================================================================
        st.markdown('<div class="step-header"><h3 style="margin:0;">Step 2: Configure Chunking & Preview</h3></div>', unsafe_allow_html=True)

        st.caption("Adjust chunk sizes to control how the document is split for retrieval.")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            large_size = st.number_input(
                "Large Chunk (tokens)",
                min_value=512,
                max_value=4096,
                value=DEFAULT_CHUNK_SIZES['large'],
                step=256,
                help="For broad context and summaries"
            )

        with col2:
            medium_size = st.number_input(
                "Medium Chunk (tokens)",
                min_value=128,
                max_value=1024,
                value=DEFAULT_CHUNK_SIZES['medium'],
                step=64,
                help="Balanced context and precision"
            )

        with col3:
            small_size = st.number_input(
                "Small Chunk (tokens)",
                min_value=64,
                max_value=512,
                value=DEFAULT_CHUNK_SIZES['small'],
                step=32,
                help="High precision for specific facts"
            )

        with col4:
            overlap_ratio = st.slider(
                "Overlap Ratio",
                min_value=0.1,
                max_value=0.4,
                value=DEFAULT_OVERLAP_RATIO,
                step=0.05,
                help="Overlap between chunks (% of smallest)"
            )

        # Validate chunk sizes
        chunk_sizes = sorted([large_size, medium_size, small_size], reverse=True)

        if chunk_sizes != [large_size, medium_size, small_size]:
            st.warning("Chunk sizes have been reordered: Large > Medium > Small")

        # Preview button
        if st.button("Preview Chunks", type="primary"):
            if st.session_state.documents is None:
                st.error("Documents not loaded. Please re-upload the PDF.")
            else:
                with st.spinner("Generating chunk preview..."):
                    chunks_info, nodes = preview_chunks(
                        st.session_state.documents,
                        chunk_sizes,
                        overlap_ratio
                    )
                    st.session_state.chunks_preview = chunks_info
                    st.session_state.chunk_sizes = chunk_sizes
                    st.session_state.overlap_ratio = overlap_ratio

        # Display preview
        if st.session_state.chunks_preview:
            chunks = st.session_state.chunks_preview

            st.divider()
            st.subheader("Chunk Preview")

            # Stats
            col1, col2, col3, col4, col5 = st.columns(5)
            large_count = len([c for c in chunks if c['level'] == 'large'])
            medium_count = len([c for c in chunks if c['level'] == 'medium'])
            small_count = len([c for c in chunks if c['level'] == 'small'])

            col1.metric("Total Chunks", len(chunks))
            col2.metric("Large", large_count, help=f"{chunk_sizes[0]} tokens")
            col3.metric("Medium", medium_count, help=f"{chunk_sizes[1]} tokens")
            col4.metric("Small", small_count, help=f"{chunk_sizes[2]} tokens")

            all_pages = sorted(set(p for c in chunks for p in c['pages']))
            col5.metric("Pages Covered", f"{min(all_pages)}-{max(all_pages)}")

            # Filters
            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                level_filter = st.selectbox("Filter by Level", ["All", "large", "medium", "small"])
            with col2:
                page_filter = st.selectbox("Filter by Page", ["All"] + [str(p) for p in all_pages])
            with col3:
                search_text = st.text_input("Search text", "")

            # Apply filters
            filtered = chunks
            if level_filter != "All":
                filtered = [c for c in filtered if c['level'] == level_filter]
            if page_filter != "All":
                filtered = [c for c in filtered if int(page_filter) in c['pages']]
            if search_text:
                filtered = [c for c in filtered if search_text.lower() in c['text'].lower()]

            st.caption(f"Showing {len(filtered)} of {len(chunks)} chunks")

            # Display chunks
            level_colors = {'large': '#e74c3c', 'medium': '#f39c12', 'small': '#27ae60'}

            for chunk in filtered[:30]:
                color = level_colors.get(chunk['level'], '#95a5a6')
                pages_str = ', '.join(map(str, chunk['pages']))

                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid {color}; border-radius: 8px; padding: 12px; margin: 8px 0; background: #fafafa;">
                        <div style="display: flex; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; align-items: center;">
                            <span style="background: {color}; color: white; padding: 2px 12px; border-radius: 12px; font-weight: bold;">
                                {chunk['level'].upper()}
                            </span>
                            <span style="background: #3498db; color: white; padding: 2px 12px; border-radius: 12px;">
                                Pages: {pages_str}
                            </span>
                            <span style="background: #9b59b6; color: white; padding: 2px 12px; border-radius: 12px;">
                                ~{chunk['size']} tokens
                            </span>
                            <span style="color: #666;">({chunk['char_count']} chars)</span>
                            <span style="color: #999; font-size: 11px;">ID: {chunk['node_id'][:12]}...</span>
                        </div>
                        <div style="font-family: monospace; font-size: 12px; color: #333; white-space: pre-wrap; max-height: 120px; overflow-y: auto; background: white; padding: 8px; border-radius: 4px;">
{chunk['preview']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Metadata expander
                    with st.expander(f"📋 View Metadata", expanded=False):
                        metadata = chunk.get('metadata', {})
                        # Display metadata in a nice format
                        col1, col2 = st.columns(2)
                        meta_items = list(metadata.items())
                        half = len(meta_items) // 2 + 1

                        with col1:
                            for key, value in meta_items[:half]:
                                if isinstance(value, list):
                                    value = ', '.join(map(str, value))
                                st.markdown(f"**{key}:** `{value}`")

                        with col2:
                            for key, value in meta_items[half:]:
                                if isinstance(value, list):
                                    value = ', '.join(map(str, value))
                                st.markdown(f"**{key}:** `{value}`")

            if len(filtered) > 30:
                st.info(f"Showing first 30 of {len(filtered)} chunks")

            # ==========================================================================
            # STEP 3: Index to Database
            # ==========================================================================
            st.markdown('<div class="step-header"><h3 style="margin:0;">Step 3: Index to Vector Database</h3></div>', unsafe_allow_html=True)

            st.caption("This will create embeddings and store chunks in ChromaDB. This process calls the OpenAI API.")

            # Initialize indexing state
            if 'indexing_in_progress' not in st.session_state:
                st.session_state.indexing_in_progress = False

            # Create placeholder for progress
            progress_placeholder = st.empty()

            col1, col2 = st.columns([1, 3])
            with col1:
                # Only show button if not currently indexing
                if not st.session_state.indexing_in_progress:
                    if st.button("Index Document", type="primary", use_container_width=True):
                        st.session_state.indexing_in_progress = True
                        st.rerun()

            # Handle indexing in a separate block
            if st.session_state.indexing_in_progress:
                with progress_placeholder.container():
                    progress = st.progress(0, text="Starting indexing...")

                    try:
                        progress.progress(10, text="Initializing system...")
                        system = index_documents(
                            st.session_state.documents,
                            st.session_state.chunk_sizes,
                            st.session_state.overlap_ratio
                        )
                        progress.progress(100, text="Complete!")
                        st.success("Document indexed successfully!")
                        st.session_state.indexing_in_progress = False
                        # Reset to Query tab after indexing
                        st.query_params["tab"] = "query"
                        # Flag to scroll to top after rerun
                        st.session_state.scroll_to_top = True
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.session_state.indexing_in_progress = False
                        st.error(f"Indexing failed: {str(e)}")

else:
    # ==========================================================================
    # DATABASE EXISTS - Show Query Interface with Tabs
    # ==========================================================================

    # Initialize system if needed
    if not st.session_state.system:
        with st.spinner("Loading system..."):
            from main import InsuranceClaimSystem
            setup_logging()
            st.session_state.system = InsuranceClaimSystem(
                data_dir="./data",
                chroma_dir="./chroma_db",
                rebuild_indexes=False
            )

    # Tab options and URL persistence
    tab_options = ["🔍 Query", "📚 Browse Vector DB", "📊 RAGAS Evaluation", "🧪 Code-Based Graders"]
    tab_url_map = {"query": 0, "browse": 1, "eval": 2, "graders": 3}
    url_tab_map = {0: "query", 1: "browse", 2: "eval", 3: "graders"}

    # Get tab from URL query params (persists across refresh)
    query_params = st.query_params
    url_tab = query_params.get("tab", "query")
    default_tab_index = tab_url_map.get(url_tab, 0)

    # Create tab-styled radio buttons
    selected_tab = st.radio(
        "Navigation",
        tab_options,
        index=default_tab_index,
        horizontal=True,
        label_visibility="collapsed",
        key="main_tab_selector"
    )

    # Update URL when tab changes
    new_tab_index = tab_options.index(selected_tab)
    new_url_tab = url_tab_map[new_tab_index]
    if query_params.get("tab") != new_url_tab:
        st.query_params["tab"] = new_url_tab

    # ==========================================================================
    # TAB 1: Query Interface
    # ==========================================================================
    if selected_tab == "🔍 Query":
        st.markdown('<div class="step-header"><h3 style="margin:0;">Query the Insurance Claim</h3></div>', unsafe_allow_html=True)

        # Initialize session state for enter key submission
        if 'query_submitted' not in st.session_state:
            st.session_state.query_submitted = False

        def on_query_enter():
            """Callback when Enter is pressed in query input"""
            st.session_state.query_submitted = True

        # Query input with Enter key detection
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What was the total repair cost? When did the accident occur?",
            key="query_input",
            on_change=on_query_enter
        )

        # Ask button and Top K selector on same row
        col1, col_div, col2, col3, col_div2, col4, col5, col6 = st.columns([1, 0.1, 0.5, 1.2, 0.1, 0.8, 1.2, 2.9])
        with col1:
            query_btn = st.button("Ask", type="primary", use_container_width=True)
        with col_div:
            st.markdown('<div style="border-left: 1px solid #ccc; height: 38px; margin: 0 auto;"></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div style="padding-top: 8px; white-space: nowrap;">Top K:</div>', unsafe_allow_html=True)
        with col3:
            top_k = st.number_input(
                "Top K",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Number of chunks to retrieve (higher = more context, slower)",
                label_visibility="collapsed"
            )
        with col_div2:
            st.markdown('<div style="border-left: 1px solid #ccc; height: 38px; margin: 0 auto;"></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div style="padding-top: 8px; white-space: nowrap;">Temperature:</div>', unsafe_allow_html=True)
        with col5:
            temperature = st.number_input(
                "Temperature",
                min_value=-0.01,
                max_value=2.01,
                value=0.0,
                step=0.1,
                format="%.2g",
                help="LLM temperature (0 = deterministic, higher = more creative)",
                label_visibility="collapsed"
            )

        # Example queries
        with st.expander("Example Questions", expanded=True):
            examples = [
                "What is this insurance claim about?",
                "What was the exact deductible amount?",
                "When did the accident occur?",
                "Summarize the timeline of events",
                "Who was the claims adjuster?",
                "What was the total repair cost?",
                "What did the witnesses say?",
                "How many days between the incident and claim filing?"
            ]
            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        # Store in session state and rerun
                        st.session_state.pending_query = ex
                        st.rerun()

        # Check for pending query from example buttons
        if 'pending_query' in st.session_state and st.session_state.pending_query:
            query = st.session_state.pending_query
            query_btn = True
            st.session_state.pending_query = None

        # Run query (on button click or Enter key press)
        should_run_query = (query_btn or st.session_state.query_submitted) and query
        if st.session_state.query_submitted:
            st.session_state.query_submitted = False  # Reset flag

        if should_run_query:
            with st.spinner("Processing query..."):
                result = run_query(query, top_k=top_k)

            # Answer
            st.subheader("Answer")
            if result.get('success', False):
                st.markdown(result.get('output', 'No output'))
                st.caption(f"Response time: {result.get('elapsed_time', 'N/A')}")
            else:
                st.error(result.get('output', 'Error processing query'))

            # Behind the scenes
            with st.expander("Behind the Scenes (Execution Log)", expanded=True):
                logs = result.get('logs', [])
                if logs:
                    log_text = ""
                    for log in logs[-30:]:
                        color = '🟢' if log['level'] == 'INFO' else '🟡' if log['level'] == 'WARNING' else '🔴'
                        log_text += f"{color} [{log['time']}] {log['message']}\n"
                    st.code(log_text, language=None)
                else:
                    st.info("No logs captured")

        # Query History
        if st.session_state.query_history:
            st.divider()
            st.subheader("Recent Queries")

            for item in reversed(st.session_state.query_history[-5:]):
                with st.expander(f"[{item['timestamp']}] {item['query'][:60]}..."):
                    st.markdown(f"**Q:** {item['query']}")
                    st.markdown(f"**A:** {item['result'].get('output', 'N/A')[:500]}...")

    # ==========================================================================
    # TAB 2: Browse Vector DB
    # ==========================================================================
    elif selected_tab == "📚 Browse Vector DB":
        st.markdown('<div class="step-header"><h3 style="margin:0;">Browse Vector Database</h3></div>', unsafe_allow_html=True)

        # Get ChromaDB data
        import chromadb
        from chromadb.config import Settings

        chroma_dir = Path("./chroma_db")

        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )

            # Collection selector
            col1, col2 = st.columns([2, 1])
            with col1:
                collection_name = st.selectbox(
                    "Select Collection:",
                    ["insurance_hierarchical", "insurance_summaries"]
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Spacer to align with selectbox label
                refresh_btn = st.button("🔄 Refresh", use_container_width=True)

            try:
                collection = client.get_collection(collection_name)
                total_count = collection.count()

                st.info(f"**{collection_name}**: {total_count} items")

                # Filters
                st.subheader("Filters")
                col1, col2, col3 = st.columns(3)

                with col1:
                    level_filter = st.selectbox(
                        "Chunk Level:",
                        ["All", "large", "medium", "small"],
                        key="browse_level"
                    )
                with col2:
                    page_filter = st.text_input("Page Number:", "", key="browse_page")
                with col3:
                    search_text = st.text_input("Search Text:", "", key="browse_search")

                # Build where filter
                where_filter = None
                if level_filter != "All":
                    where_filter = {"chunk_level": level_filter}

                # Get items
                limit = st.slider("Items to display:", 10, 100, 30)

                if where_filter:
                    results = collection.get(
                        where=where_filter,
                        limit=limit,
                        include=["documents", "metadatas", "embeddings"]
                    )
                else:
                    results = collection.get(
                        limit=limit,
                        include=["documents", "metadatas", "embeddings"]
                    )

                # Filter by page if specified
                if page_filter:
                    try:
                        page_num = int(page_filter)
                        filtered_indices = []
                        for i, meta in enumerate(results['metadatas']):
                            pages_str = meta.get('page_numbers', '1')
                            pages = [int(p) for p in pages_str.split(',') if p]
                            if page_num in pages:
                                filtered_indices.append(i)
                        results = {
                            'ids': [results['ids'][i] for i in filtered_indices],
                            'documents': [results['documents'][i] for i in filtered_indices],
                            'metadatas': [results['metadatas'][i] for i in filtered_indices],
                            'embeddings': [results['embeddings'][i] for i in filtered_indices] if results.get('embeddings') is not None else None
                        }
                    except ValueError:
                        pass

                # Filter by search text
                if search_text:
                    filtered_indices = []
                    for i, doc in enumerate(results['documents']):
                        if doc and search_text.lower() in doc.lower():
                            filtered_indices.append(i)
                    results = {
                        'ids': [results['ids'][i] for i in filtered_indices],
                        'documents': [results['documents'][i] for i in filtered_indices],
                        'metadatas': [results['metadatas'][i] for i in filtered_indices],
                        'embeddings': [results['embeddings'][i] for i in filtered_indices] if results.get('embeddings') else None
                    }

                st.caption(f"Showing {len(results['ids'])} items")

                # Build dataframe for table display
                import pandas as pd

                table_data = []
                embeddings_list = results.get('embeddings') if results.get('embeddings') is not None else [None] * len(results['ids'])
                for doc_id, doc, meta, emb in zip(results['ids'], results['documents'], results['metadatas'], embeddings_list):
                    row = {
                        'ID': doc_id[:15] + '...',
                        'Level': meta.get('chunk_level', 'N/A'),
                        'Pages': meta.get('page_numbers', '1'),
                        'Start Page': meta.get('start_page', 1),
                        'End Page': meta.get('end_page', 1),
                        'Tokens': meta.get('chunk_size', 0),
                        'Vector Dims': len(emb) if emb is not None else 0,
                        'Claim ID': meta.get('claim_id', 'N/A'),
                        'Section': meta.get('section_title', 'N/A')[:20],
                        'Content Preview': (doc[:150] + '...') if doc and len(doc) > 150 else (doc or 'No content'),
                        'Full Content': doc or 'No content',
                        'Full ID': doc_id
                    }
                    table_data.append(row)

                df = pd.DataFrame(table_data)

                # Display table
                st.divider()

                # Pagination controls
                if 'browse_table_page' not in st.session_state:
                    st.session_state.browse_table_page = 0

                page_size = 10
                total_items = len(df)
                total_pages = max(1, (total_items + page_size - 1) // page_size)

                # Reset page if out of bounds
                if st.session_state.browse_table_page >= total_pages:
                    st.session_state.browse_table_page = 0

                # Pagination UI
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                with col1:
                    if st.button("⏮️ First", disabled=st.session_state.browse_table_page == 0, use_container_width=True):
                        st.session_state.browse_table_page = 0
                        st.rerun()
                with col2:
                    if st.button("◀️ Prev", disabled=st.session_state.browse_table_page == 0, use_container_width=True):
                        st.session_state.browse_table_page -= 1
                        st.rerun()
                with col3:
                    st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {st.session_state.browse_table_page + 1} of {total_pages} ({total_items} items)</div>", unsafe_allow_html=True)
                with col4:
                    if st.button("Next ▶️", disabled=st.session_state.browse_table_page >= total_pages - 1, use_container_width=True):
                        st.session_state.browse_table_page += 1
                        st.rerun()
                with col5:
                    if st.button("Last ⏭️", disabled=st.session_state.browse_table_page >= total_pages - 1, use_container_width=True):
                        st.session_state.browse_table_page = total_pages - 1
                        st.rerun()

                # Slice dataframe for current page
                start_idx = st.session_state.browse_table_page * page_size
                end_idx = min(start_idx + page_size, total_items)
                df_page = df.iloc[start_idx:end_idx]

                # Column configuration for better display with row selection
                selection = st.dataframe(
                    df_page[['Level', 'Pages', 'Start Page', 'End Page', 'Tokens', 'Vector Dims', 'Claim ID', 'Content Preview']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Level': st.column_config.TextColumn('Level', width='small'),
                        'Pages': st.column_config.TextColumn('Pages', width='small'),
                        'Start Page': st.column_config.NumberColumn('Start', width='small'),
                        'End Page': st.column_config.NumberColumn('End', width='small'),
                        'Tokens': st.column_config.NumberColumn('Tokens', width='small'),
                        'Vector Dims': st.column_config.NumberColumn('Vector', width='small'),
                        'Claim ID': st.column_config.TextColumn('Claim', width='medium'),
                        'Content Preview': st.column_config.TextColumn('Content Preview', width='large'),
                    },
                    selection_mode="single-row",
                    on_select="rerun",
                    key="chunk_table_selection"
                )

                # Detail view for selected chunk (from current page)
                st.divider()
                st.subheader("Chunk Detail View")

                # Get selected row from table click
                selected_rows = selection.selection.rows if selection.selection else []
                if selected_rows:
                    # Map back to full table index
                    selected_page_idx = selected_rows[0]
                    selected_idx = start_idx + selected_page_idx
                    selected = table_data[selected_idx]
                    meta = results['metadatas'][selected_idx]
                    embedding = results['embeddings'][selected_idx] if results.get('embeddings') is not None else None

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Metadata:**")
                        for key, value in meta.items():
                            # Skip _node_content as it's redundant with Full Content
                            if key == '_node_content':
                                continue
                            st.markdown(f"- **{key}:** `{value}`")

                    with col2:
                        st.markdown("**Full Content:**")
                        st.text_area("Content", value=selected['Full Content'], height=300, disabled=True)

                    # Display embedding vector
                    st.divider()
                    st.markdown("**Embedding Vector:**")
                    if embedding is not None:
                        import numpy as np
                        embedding_array = np.array(embedding)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Dimensions", len(embedding))
                        with col2:
                            st.metric("Min Value", f"{embedding_array.min():.6f}")
                        with col3:
                            st.metric("Max Value", f"{embedding_array.max():.6f}")

                        # Show first/last values preview
                        with st.expander("View Embedding Values", expanded=True):
                            st.caption(f"First 20 values: {embedding[:20]}")
                            st.caption(f"Last 20 values: {embedding[-20:]}")
                            # Format embedding as comma-separated values with line breaks
                            formatted_embedding = ",\n".join([f"{v:.8f}" for v in embedding])
                            st.text_area(
                                "Full Embedding Vector",
                                value=formatted_embedding,
                                height=400,
                                disabled=True
                            )
                    else:
                        st.info("No embedding data available")
                else:
                    st.info("Tick the checkbox of a row in the table above to view its details")

            except Exception as e:
                st.warning(f"Collection '{collection_name}' not found or empty: {e}")

        except Exception as e:
            st.error(f"Error connecting to ChromaDB: {e}")

    # ==========================================================================
    # TAB 3: RAGAS Evaluation
    # ==========================================================================
    elif selected_tab == "📊 RAGAS Evaluation":
        st.markdown('<div class="step-header"><h3 style="margin:0;">System Evaluation</h3></div>', unsafe_allow_html=True)

        # Evaluation method selector
        eval_method = st.radio(
            "Select Evaluation Method:",
            ["RAGAS (OpenAI GPT-4o-mini)", "LLM-as-a-Judge (Anthropic Claude)"],
            horizontal=True,
            help="RAGAS uses OpenAI for compatibility. LLM-as-a-Judge uses Claude for truly independent evaluation."
        )

        if eval_method == "RAGAS (OpenAI GPT-4o-mini)":
            st.info("""
            **RAGAS (Retrieval Augmented Generation Assessment)** evaluates RAG pipeline quality using metrics:
            - **Faithfulness**: Is the answer grounded in the retrieved context?
            - **Answer Relevancy**: Is the answer relevant to the question?
            - **Context Precision**: Are the retrieved chunks relevant?
            - **Context Recall**: Does the context contain the information needed?

            *Note: Uses GPT-4o-mini (different from GPT-4 used for generation) due to RAGAS compatibility requirements.*
            """)
        else:
            st.info("""
            **LLM-as-a-Judge** uses Anthropic Claude (completely different provider) for truly independent evaluation:
            - **Correctness**: Does the answer match the ground truth?
            - **Relevancy**: Is the retrieved context relevant to the question?
            - **Recall**: Were all necessary chunks retrieved?

            *Uses Claude Sonnet - completely independent from OpenAI GPT-4 used for generation.*
            """)

        # Initialize session state
        if 'ragas_test_cases' not in st.session_state:
            st.session_state.ragas_test_cases = []
        if 'ragas_results' not in st.session_state:
            st.session_state.ragas_results = None
        if 'judge_results' not in st.session_state:
            st.session_state.judge_results = None
        if 'evaluation_history' not in st.session_state:
            st.session_state.evaluation_history = load_evaluation_history()

        # Test case management
        st.subheader("Test Cases")

        # Add test case form
        with st.expander("Add Test Case", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                test_question = st.text_input("Question:", key="ragas_question",
                    placeholder="e.g., When did the accident occur?")
            with col2:
                ground_truth = st.text_input("Ground Truth (optional):", key="ragas_ground_truth",
                    placeholder="e.g., January 15, 2024")

            if st.button("Add Test Case", type="primary"):
                if test_question:
                    st.session_state.ragas_test_cases.append({
                        'question': test_question,
                        'ground_truth': ground_truth if ground_truth else None
                    })
                    st.success(f"Added test case: {test_question[:50]}...")
                    st.rerun()
                else:
                    st.warning("Please enter a question")

        # Predefined test cases - 5 Summary (General) + 5 Needle queries
        predefined_cases = [
            # ===== SUMMARY / GENERAL QUERIES (5) =====
            {"question": "What is this insurance claim about? Provide a summary.", "ground_truth": "This is an auto insurance claim (CLM-2024-001) for a multi-vehicle collision that occurred on January 12, 2024, at 7:42 AM at the intersection of Wilshire Blvd and Vermont Ave in Los Angeles. Sarah Mitchell's 2021 Honda Accord was struck by Robert Harrison's vehicle which ran a red light while Harrison was driving under the influence (BAC 0.14). Mitchell sustained whiplash injuries, the vehicle required $17,111.83 in repairs, and the total claim amount was $23,370.80.", "category": "Summary"},
            {"question": "Provide a timeline of key events from the incident through vehicle return.", "ground_truth": "January 12, 2024 (7:42 AM): Incident occurred. January 15, 2024: Claim filed. January 26, 2024: Liability accepted by at-fault party. January 29, 2024: Repairs authorized and commenced. February 15, 2024: Repairs completed. February 16, 2024: Vehicle returned to policyholder.", "category": "Summary"},
            {"question": "Who were the witnesses and what did they observe?", "ground_truth": "There were three witnesses: Marcus Thompson (at bus stop) saw Harrison's Camry run a red light at high speed without braking. Elena Rodriguez (stopped in left turn lane) observed the Camry had a red light for several seconds before entering the intersection and noted Harrison appeared intoxicated after the crash. Patricia O'Brien (RN) confirmed the traffic signal timing and noted sunrise was at 6:58 AM with normal lighting conditions.", "category": "Summary"},
            {"question": "Summarize the medical treatment Sarah Mitchell received.", "ground_truth": "Sarah Mitchell was treated at Cedars-Sinai Emergency Department for cervical strain (whiplash) and post-traumatic headache. She had a follow-up with orthopedist Dr. Rachel Kim who prescribed physical therapy. She completed 8 physical therapy sessions at Pacific Coast Physical Therapy with Marcus Rodriguez, PT, DPT, from February 2-27, 2024.", "category": "Summary"},
            {"question": "What was the outcome of the liability determination?", "ground_truth": "Robert Harrison's insurance company (Nationwide Insurance) accepted 100% liability for the accident on January 26, 2024. Harrison was cited for DUI and running a red light. His BAC was 0.14%, above the legal limit of 0.08%.", "category": "Summary"},
            # ===== NEEDLE QUERIES (5) =====
            {"question": "What was the exact collision deductible amount?", "ground_truth": "The collision deductible was exactly $750.", "category": "Needle"},
            {"question": "At what exact time did the accident occur?", "ground_truth": "The accident occurred at exactly 7:42 AM (more precisely 7:42:15 AM based on the incident timeline).", "category": "Needle"},
            {"question": "Who was the claims adjuster assigned to this case?", "ground_truth": "Kevin Park was the claims adjuster assigned to this case.", "category": "Needle"},
            {"question": "What was Robert Harrison's Blood Alcohol Concentration (BAC)?", "ground_truth": "Robert Harrison's Blood Alcohol Concentration (BAC) was 0.14%, which is significantly above the legal limit of 0.08%.", "category": "Needle"},
            {"question": "How many physical therapy sessions did Sarah Mitchell complete?", "ground_truth": "Sarah Mitchell completed exactly 8 physical therapy sessions.", "category": "Needle"},
        ]

        # Auto-load predefined test cases on first visit
        if not st.session_state.ragas_test_cases:
            st.session_state.ragas_test_cases = predefined_cases.copy()

        # Display current test cases
        if st.session_state.ragas_test_cases:
            st.divider()
            st.subheader(f"Test Cases ({len(st.session_state.ragas_test_cases)})")

            # Create dataframe with checkbox column
            import pandas as pd

            # Build initial data for display (Select defaults to True)
            table_data = []
            for i, tc in enumerate(st.session_state.ragas_test_cases):
                table_data.append({
                    'Select': True,
                    'Category': tc.get('category', 'N/A'),
                    'Question': tc['question'],
                    'Ground Truth': tc.get('ground_truth', '')
                })

            df = pd.DataFrame(table_data)

            # Use data_editor for editable checkboxes
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Select': st.column_config.CheckboxColumn(
                        'Select',
                        default=True,
                    ),
                    'Category': st.column_config.TextColumn(
                        'Category',
                    ),
                    'Question': st.column_config.TextColumn(
                        'Question',
                    ),
                    'Ground Truth': st.column_config.TextColumn(
                        'Ground Truth',
                    ),
                },
                disabled=['Category', 'Question', 'Ground Truth'],
                key="ragas_table_editor"
            )

            # Get selection from edited dataframe (data_editor maintains state)
            selected_count = edited_df['Select'].sum()
            st.caption(f"Selected: {int(selected_count)}/{len(st.session_state.ragas_test_cases)}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear All Test Cases", type="secondary"):
                    st.session_state.ragas_test_cases = []
                    st.session_state.ragas_results = None
                    st.session_state.judge_results = None
                    # Clear the data_editor state
                    if 'ragas_table_editor' in st.session_state:
                        del st.session_state['ragas_table_editor']

            with col2:
                # Dynamic button label based on selected evaluation method
                button_label = "Run RAGAS Evaluation" if eval_method == "RAGAS (OpenAI GPT-4o-mini)" else "Run LLM-as-a-Judge"
                run_eval = st.button(button_label, type="primary", use_container_width=True)

            # Initialize evaluation running state
            if 'evaluation_running' not in st.session_state:
                st.session_state.evaluation_running = False

            # Show flashing warning only during evaluation
            warning_placeholder = st.empty()
            if st.session_state.evaluation_running:
                warning_placeholder.markdown(
                    '<div class="flash-warning" style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin: 10px 0;">'
                    '<strong>⚠️ Important:</strong> Do not switch tabs or interact with other UI elements while evaluation is running. This will interrupt the process.'
                    '</div>',
                    unsafe_allow_html=True
                )

            if run_eval and selected_count > 0:
                # Set evaluation running state
                st.session_state.evaluation_running = True
                st.rerun()

            # Handle evaluation in separate block (after rerun with state set)
            if st.session_state.evaluation_running and selected_count > 0:
                # Determine which evaluation method to use
                is_ragas = eval_method == "RAGAS (OpenAI GPT-4o-mini)"
                spinner_text = "Running RAGAS evaluation..." if is_ragas else "Running LLM-as-a-Judge evaluation (Claude)..."

                with st.spinner(spinner_text):
                    try:
                        # Filter to only selected test cases using edited_df
                        selected_cases = [
                            (i, tc) for i, tc in enumerate(st.session_state.ragas_test_cases)
                            if edited_df['Select'].iloc[i]
                        ]
                        total_cases = len(selected_cases)

                        if total_cases == 0:
                            st.warning("No test cases selected. Please select at least one question.")
                            st.stop()

                        if is_ragas:
                            # ============== RAGAS EVALUATION ==============
                            from ragas import evaluate
                            from ragas.metrics import (
                                faithfulness,
                                answer_relevancy,
                                context_precision,
                                context_recall
                            )
                            from ragas.llms import LangchainLLMWrapper
                            from ragas.embeddings import LangchainEmbeddingsWrapper
                            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                            from datasets import Dataset

                            ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
                            ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

                            questions, answers, contexts, ground_truths = [], [], [], []
                            progress = st.progress(0, text="Processing test cases...")

                            for idx, (orig_idx, test_case) in enumerate(selected_cases):
                                progress.progress((idx + 1) / total_cases,
                                    text=f"Processing Q{orig_idx+1} ({idx+1}/{total_cases}): {test_case['question'][:40]}...")

                                result = st.session_state.system.query(test_case['question'])
                                questions.append(test_case['question'])
                                answers.append(result.get('output', ''))

                                retrieved_contexts = []
                                if hasattr(st.session_state.system, 'hierarchical_retriever'):
                                    nodes = st.session_state.system.hierarchical_retriever.retrieve(
                                        test_case['question'], k=3, auto_merge=True)
                                    retrieved_contexts = [node.node.text for node in nodes]

                                contexts.append(retrieved_contexts if retrieved_contexts else [result.get('output', '')])
                                ground_truths.append(test_case['ground_truth'] if test_case['ground_truth'] else '')

                            progress.progress(1.0, text="Running RAGAS metrics...")

                            eval_data = {'question': questions, 'answer': answers, 'contexts': contexts, 'ground_truth': ground_truths}
                            dataset = Dataset.from_dict(eval_data)

                            has_ground_truth = any(gt for gt in ground_truths if gt)
                            metrics = [faithfulness, answer_relevancy, context_precision, context_recall] if has_ground_truth else [faithfulness, answer_relevancy, context_precision]

                            eval_result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_embeddings)

                            scores_dict = {}
                            for metric_key in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                                try:
                                    scores_dict[metric_key] = eval_result[metric_key]
                                except (KeyError, TypeError):
                                    scores_dict[metric_key] = None

                            st.session_state.ragas_results = {
                                'scores': scores_dict, 'details': eval_data,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'method': 'RAGAS',
                                'num_questions': total_cases
                            }
                            # Save to history (session + file)
                            st.session_state.evaluation_history.append(st.session_state.ragas_results.copy())
                            save_evaluation_history(st.session_state.evaluation_history)
                            st.session_state.judge_results = None

                        else:
                            # ============== LLM-AS-A-JUDGE EVALUATION (Claude) ==============
                            from src.evaluation.judge import LLMJudge

                            judge = LLMJudge(temperature=0)  # Uses Claude by default
                            progress = st.progress(0, text="Evaluating with Claude...")

                            query_results = []
                            for idx, (orig_idx, test_case) in enumerate(selected_cases):
                                progress.progress((idx + 1) / total_cases,
                                    text=f"Evaluating Q{orig_idx+1} ({idx+1}/{total_cases}): {test_case['question'][:40]}...")

                                # Run query through the system
                                result = st.session_state.system.query(test_case['question'])
                                answer = result.get('output', '')

                                # Get retrieved context
                                retrieved_context = ""
                                retrieved_chunks = []
                                if hasattr(st.session_state.system, 'hierarchical_retriever'):
                                    nodes = st.session_state.system.hierarchical_retriever.retrieve(
                                        test_case['question'], k=3, auto_merge=True)
                                    retrieved_chunks = [node.node.text for node in nodes]
                                    retrieved_context = "\n\n".join(retrieved_chunks)

                                # Run evaluation
                                ground_truth = test_case['ground_truth'] if test_case['ground_truth'] else ''
                                eval_result = judge.evaluate_full(
                                    query=test_case['question'],
                                    answer=answer,
                                    ground_truth=ground_truth,
                                    retrieved_context=retrieved_context if retrieved_context else answer,
                                    expected_chunks=[],
                                    retrieved_chunks=retrieved_chunks
                                )

                                query_results.append({
                                    'question': test_case['question'],
                                    'answer': answer,
                                    'ground_truth': ground_truth,
                                    'correctness': eval_result['correctness']['score'],
                                    'relevancy': eval_result['relevancy']['score'],
                                    'recall': eval_result.get('recall', {}).get('score', 'N/A'),
                                    'average': eval_result['average_score'],
                                    'details': eval_result
                                })

                            progress.progress(1.0, text="Complete!")

                            # Calculate aggregate scores
                            correctness_scores = [r['correctness'] for r in query_results if isinstance(r['correctness'], (int, float))]
                            relevancy_scores = [r['relevancy'] for r in query_results if isinstance(r['relevancy'], (int, float))]
                            avg_scores = [r['average'] for r in query_results if isinstance(r['average'], (int, float))]

                            st.session_state.judge_results = {
                                'scores': {
                                    'correctness': sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0,
                                    'relevancy': sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
                                    'average': sum(avg_scores) / len(avg_scores) if avg_scores else 0
                                },
                                'query_results': query_results,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'method': 'LLM-as-a-Judge (Claude)',
                                'num_questions': total_cases
                            }
                            # Save to history (session + file)
                            st.session_state.evaluation_history.append(st.session_state.judge_results.copy())
                            save_evaluation_history(st.session_state.evaluation_history)
                            st.session_state.ragas_results = None

                        st.success("Evaluation complete!")
                        st.session_state.evaluation_running = False
                        st.rerun()

                    except ImportError as e:
                        st.session_state.evaluation_running = False
                        st.error(f"""
                        Required packages not installed. Please install:
                        ```
                        pip install ragas datasets langchain-anthropic
                        ```
                        Error: {e}
                        """)
                    except Exception as e:
                        st.session_state.evaluation_running = False
                        st.error(f"Evaluation error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # Display results
        if st.session_state.ragas_results:
            st.divider()
            st.subheader("Evaluation Results")
            st.caption(f"Evaluated at: {st.session_state.ragas_results['timestamp']}")

            results = st.session_state.ragas_results['scores']

            # Display overall scores
            st.markdown("### Overall Scores")
            score_cols = st.columns(4)

            metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            metric_descriptions = [
                "Answer grounded in context",
                "Answer relevant to question",
                "Retrieved chunks relevant",
                "Context has needed info"
            ]

            for i, (metric, desc) in enumerate(zip(metric_names, metric_descriptions)):
                with score_cols[i]:
                    score = results.get(metric, None)
                    if score is not None:
                        # Handle list scores (per-sample) by averaging
                        if isinstance(score, list):
                            score = sum(score) / len(score) if score else 0
                        # Color code based on score
                        if score >= 0.8:
                            color = "🟢"
                        elif score >= 0.5:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.metric(
                            label=f"{color} {metric.replace('_', ' ').title()}",
                            value=f"{score:.2%}",
                            help=desc
                        )
                    else:
                        st.metric(label=metric.replace('_', ' ').title(), value="N/A")

            # Show detailed results
            with st.expander("Detailed Results by Question"):
                details = st.session_state.ragas_results['details']

                for i, (q, a, ctx) in enumerate(zip(
                    details['question'],
                    details['answer'],
                    details['contexts']
                )):
                    st.markdown(f"**Q{i+1}: {q}**")
                    st.markdown(f"**Answer:** {a[:300]}..." if len(a) > 300 else f"**Answer:** {a}")
                    st.markdown(f"**Contexts Retrieved:** {len(ctx)}")
                    st.divider()

            # Export option
            if st.button("Export Results as CSV"):
                import pandas as pd
                export_data = {
                    'Question': st.session_state.ragas_results['details']['question'],
                    'Answer': st.session_state.ragas_results['details']['answer'],
                    'Ground Truth': st.session_state.ragas_results['details']['ground_truth'],
                }
                # Add overall scores
                for metric in metric_names:
                    score = results.get(metric, None)
                    export_data[metric] = [score] * len(export_data['Question']) if score else ['N/A'] * len(export_data['Question'])

                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # RAGAS Improvement Recommendations
            st.divider()
            st.subheader("RAGAS Improvement Recommendations")

            # Get scores (handle list scores by averaging)
            def get_avg_score(score):
                if score is None:
                    return 0
                if isinstance(score, list):
                    return sum(score) / len(score) if score else 0
                return score

            faithfulness_score = get_avg_score(results.get('faithfulness'))
            relevancy_score = get_avg_score(results.get('answer_relevancy'))
            precision_score = get_avg_score(results.get('context_precision'))
            recall_score = get_avg_score(results.get('context_recall'))

            # Faithfulness recommendations
            with st.expander("Faithfulness Recommendations", expanded=True):
                if faithfulness_score >= 0.8:
                    st.success("**Score: Good (80%+)** - Answers are well-grounded in retrieved context.")
                    st.markdown("""
                    **To maintain high faithfulness:**
                    - Continue instructing the LLM to only use retrieved context
                    - Keep prompt instructions clear about not hallucinating
                    """)
                elif faithfulness_score >= 0.5:
                    st.warning("**Score: Moderate (50-80%)** - Some answers contain information not in context.")
                    st.markdown("""
                    **To improve faithfulness:**
                    - **Strengthen prompts**: Add explicit instructions like "Only use information from the provided context"
                    - **Lower temperature**: Use temperature=0 for more deterministic responses
                    - **Add citations**: Require the LLM to cite which chunk each fact comes from
                    - **Increase context**: Retrieve more chunks to provide more grounding information
                    """)
                else:
                    st.error("**Score: Low (<50%)** - Answers frequently contain hallucinated information.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review prompts urgently**: LLM is generating content not grounded in context
                    - **Use structured output**: Force specific answer formats
                    - **Implement fact-checking**: Add a verification step before returning answers
                    - **Consider different model**: Some models are more prone to hallucination
                    - **Reduce context noise**: Too much irrelevant context may confuse the model
                    """)

            # Answer Relevancy recommendations
            with st.expander("Answer Relevancy Recommendations", expanded=True):
                if relevancy_score >= 0.8:
                    st.success("**Score: Good (80%+)** - Answers are highly relevant to questions.")
                    st.markdown("""
                    **To maintain high relevancy:**
                    - Current question-answering approach is working well
                    - Continue monitoring for edge cases
                    """)
                elif relevancy_score >= 0.5:
                    st.warning("**Score: Moderate (50-80%)** - Some answers don't directly address the question.")
                    st.markdown("""
                    **To improve answer relevancy:**
                    - **Refine prompts**: Be more explicit about answering the specific question asked
                    - **Add question rephrasing**: Include the question in the response template
                    - **Use chain-of-thought**: Let the model reason through what's being asked
                    - **Filter verbose responses**: Post-process to extract only relevant parts
                    """)
                else:
                    st.error("**Score: Low (<50%)** - Answers frequently miss the point of questions.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review query understanding**: Model may be misinterpreting questions
                    - **Simplify questions**: Test with simpler, more direct questions first
                    - **Add query classification**: Route different question types appropriately
                    - **Check context relevance first**: Poor context leads to poor answers
                    """)

            # Context Precision recommendations
            with st.expander("Context Precision Recommendations", expanded=True):
                if precision_score >= 0.8:
                    st.success("**Score: Good (80%+)** - Retrieved chunks are highly relevant.")
                    st.markdown("""
                    **To maintain high precision:**
                    - Current retrieval strategy is effective
                    - Consider reducing k if retrieving unnecessary chunks
                    """)
                elif precision_score >= 0.5:
                    st.warning("**Score: Moderate (50-80%)** - Some retrieved chunks are not useful.")
                    st.markdown("""
                    **To improve context precision:**
                    - **Reduce retrieval k**: Retrieve fewer, more relevant chunks
                    - **Add reranking**: Use a cross-encoder to rerank retrieved chunks
                    - **Improve embeddings**: Use domain-specific embeddings if available
                    - **Filter by metadata**: Use section/type filters to narrow results
                    - **Use MMR**: Maximum Marginal Relevance reduces redundant chunks
                    """)
                else:
                    st.error("**Score: Low (<50%)** - Most retrieved chunks are irrelevant.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review embedding model**: Current embeddings may not capture query intent
                    - **Rebuild indexes**: Try different chunk sizes and overlap settings
                    - **Implement hybrid search**: Combine semantic and keyword search
                    - **Add query preprocessing**: Clean and expand queries before retrieval
                    - **Check index health**: Verify ChromaDB indexes are built correctly
                    """)

            # Context Recall recommendations
            with st.expander("Context Recall Recommendations", expanded=True):
                if recall_score >= 0.8:
                    st.success("**Score: Good (80%+)** - All necessary information is being retrieved.")
                    st.markdown("""
                    **To maintain high recall:**
                    - Current retrieval coverage is good
                    - Monitor for queries that need information from multiple sections
                    """)
                elif recall_score >= 0.5:
                    st.warning("**Score: Moderate (50-80%)** - Some needed information is not being retrieved.")
                    st.markdown("""
                    **To improve context recall:**
                    - **Increase retrieval k**: Retrieve more chunks to improve coverage
                    - **Use query expansion**: Add synonyms and related terms
                    - **Multi-index search**: Search both Summary and Hierarchical indexes
                    - **Reduce chunk size**: Smaller chunks may have better coverage
                    - **Add fallback search**: If first search fails, try broader search
                    """)
                else:
                    st.error("**Score: Low (<50%)** - Critical information is frequently missed.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review indexing coverage**: Ensure all document sections are indexed
                    - **Check for parsing issues**: Some content may not be extracted properly
                    - **Increase chunk overlap**: Higher overlap prevents boundary information loss
                    - **Use multiple retrieval strategies**: Combine different approaches
                    - **Add recursive retrieval**: Follow references to related chunks
                    """)

        # Display LLM-as-a-Judge results
        if st.session_state.judge_results:
            st.divider()
            st.subheader("LLM-as-a-Judge Results (Claude)")
            st.caption(f"Evaluated at: {st.session_state.judge_results['timestamp']}")
            st.success("Using Anthropic Claude - completely independent from OpenAI GPT-4 used for generation")

            results = st.session_state.judge_results['scores']

            # Display overall scores
            st.markdown("### Overall Scores (1-5 scale)")
            score_cols = st.columns(3)

            judge_metrics = [
                ('correctness', 'Answer matches ground truth'),
                ('relevancy', 'Context relevant to question'),
                ('average', 'Overall average score')
            ]

            for i, (metric, desc) in enumerate(judge_metrics):
                with score_cols[i]:
                    score = results.get(metric, 0)
                    if isinstance(score, (int, float)):
                        # Color code based on score (1-5 scale)
                        if score >= 4:
                            color = "🟢"
                        elif score >= 3:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.metric(
                            label=f"{color} {metric.title()}",
                            value=f"{score:.2f}/5",
                            help=desc
                        )
                    else:
                        st.metric(label=metric.title(), value="N/A")

            # Show detailed results by question
            with st.expander("Detailed Results by Question", expanded=True):
                for i, qr in enumerate(st.session_state.judge_results['query_results']):
                    st.markdown(f"**Q{i+1}: {qr['question']}**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        c_score = qr['correctness']
                        c_color = "🟢" if c_score >= 4 else "🟡" if c_score >= 3 else "🔴"
                        st.markdown(f"{c_color} **Correctness:** {c_score}/5")
                    with col2:
                        r_score = qr['relevancy']
                        r_color = "🟢" if r_score >= 4 else "🟡" if r_score >= 3 else "🔴"
                        st.markdown(f"{r_color} **Relevancy:** {r_score}/5")
                    with col3:
                        avg = qr['average']
                        a_color = "🟢" if avg >= 4 else "🟡" if avg >= 3 else "🔴"
                        st.markdown(f"{a_color} **Average:** {avg:.2f}/5")

                    answer_preview = qr['answer'][:200] + "..." if len(qr['answer']) > 200 else qr['answer']
                    st.markdown(f"**Answer:** {answer_preview}")
                    st.divider()

            # Export option
            if st.button("Export LLM-as-a-Judge Results as CSV"):
                import pandas as pd
                export_data = {
                    'Question': [qr['question'] for qr in st.session_state.judge_results['query_results']],
                    'Answer': [qr['answer'] for qr in st.session_state.judge_results['query_results']],
                    'Ground Truth': [qr['ground_truth'] for qr in st.session_state.judge_results['query_results']],
                    'Correctness': [qr['correctness'] for qr in st.session_state.judge_results['query_results']],
                    'Relevancy': [qr['relevancy'] for qr in st.session_state.judge_results['query_results']],
                    'Average': [qr['average'] for qr in st.session_state.judge_results['query_results']],
                }

                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"llm_judge_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            # Improvement Recommendations based on scores
            st.divider()
            st.subheader("Improvement Recommendations")

            correctness_avg = results.get('correctness', 0)
            relevancy_avg = results.get('relevancy', 0)
            overall_avg = results.get('average', 0)

            # Correctness recommendations
            with st.expander("Correctness Score Recommendations", expanded=True):
                if correctness_avg >= 4:
                    st.success("**Score: Good (4+/5)** - Your system is generating accurate answers.")
                    st.markdown("""
                    **To maintain high correctness:**
                    - Continue using specific, well-structured prompts
                    - Keep ground truth data up to date
                    - Monitor for any drift in answer quality over time
                    """)
                elif correctness_avg >= 3:
                    st.warning("**Score: Moderate (3-4/5)** - Answers are partially correct but missing some details.")
                    st.markdown("""
                    **To improve correctness:**
                    - **Increase chunk overlap**: Try 25-30% overlap to capture more context at boundaries
                    - **Add more small chunks**: Reduce small chunk size (e.g., 64-96 tokens) for better precision
                    - **Improve prompts**: Make the LLM prompt more specific about extracting exact facts
                    - **Check retrieval**: Ensure the right chunks are being retrieved (see Relevancy)
                    - **Review ground truth**: Ensure your ground truth answers are accurate and complete
                    """)
                else:
                    st.error("**Score: Low (<3/5)** - Answers are frequently incorrect or incomplete.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review chunking strategy**: Current chunks may be too large, splitting important information
                    - **Reduce chunk sizes**: Try smaller chunks (128 tokens or less) for needle queries
                    - **Check document parsing**: Ensure PDF text extraction is working correctly
                    - **Increase retrieval k**: Retrieve more chunks (k=5-10) to capture more relevant information
                    - **Add metadata filtering**: Use section-based filtering to narrow retrieval scope
                    - **Verify embeddings**: Ensure embeddings capture semantic meaning properly
                    - **Review LLM temperature**: Use temperature=0 for more deterministic, factual responses
                    """)

            # Relevancy recommendations
            with st.expander("Relevancy Score Recommendations", expanded=True):
                if relevancy_avg >= 4:
                    st.success("**Score: Good (4+/5)** - Retrieved context is highly relevant to queries.")
                    st.markdown("""
                    **To maintain high relevancy:**
                    - Current retrieval strategy is working well
                    - Continue using hierarchical indexing for query routing
                    - Monitor for edge cases where retrieval might fail
                    """)
                elif relevancy_avg >= 3:
                    st.warning("**Score: Moderate (3-4/5)** - Some retrieved context is not directly relevant.")
                    st.markdown("""
                    **To improve relevancy:**
                    - **Tune retrieval k**: Reduce k if retrieving too much irrelevant content
                    - **Use metadata filters**: Filter by section, date, or document type
                    - **Improve query routing**: Ensure summary vs needle queries go to appropriate indexes
                    - **Add reranking**: Implement a reranker to filter out low-relevance chunks
                    - **Review chunk boundaries**: Ensure chunks contain coherent, complete information
                    """)
                else:
                    st.error("**Score: Low (<3/5)** - Retrieved context is often irrelevant to the query.")
                    st.markdown("""
                    **Critical improvements needed:**
                    - **Review embedding model**: Current embeddings may not capture domain-specific semantics
                    - **Fine-tune embeddings**: Consider domain-specific embedding fine-tuning
                    - **Implement hybrid search**: Combine vector search with keyword (BM25) search
                    - **Add query expansion**: Expand queries with synonyms or related terms
                    - **Review index structure**: Ensure Summary and Hierarchical indexes are built correctly
                    - **Check for data quality issues**: Verify document content is clean and well-formatted
                    - **Implement MMR**: Use Maximum Marginal Relevance to diversify results
                    """)

            # Overall recommendations
            with st.expander("Overall System Recommendations", expanded=True):
                if overall_avg >= 4:
                    st.success("**Overall: Excellent (4+/5)** - System is performing well!")
                    st.markdown("""
                    **Next steps:**
                    - Add more diverse test cases to ensure robustness
                    - Consider A/B testing different configurations
                    - Monitor production performance over time
                    - Document successful patterns for future reference
                    """)
                elif overall_avg >= 3:
                    st.warning("**Overall: Acceptable (3-4/5)** - System works but has room for improvement.")
                    st.markdown("""
                    **Priority improvements:**
                    1. Focus on the lowest-scoring metric first
                    2. Review questions that scored below 3
                    3. Consider adjusting chunk sizes based on query types
                    4. Test with more diverse queries to identify weak spots
                    """)
                else:
                    st.error("**Overall: Needs Work (<3/5)** - System requires significant improvements.")
                    st.markdown("""
                    **Recommended action plan:**
                    1. **Rebuild indexes** with different chunk sizes (try 1024/256/64)
                    2. **Review document quality** - ensure clean text extraction
                    3. **Simplify queries** - test with basic queries first
                    4. **Check API responses** - verify LLM is responding appropriately
                    5. **Increase logging** - add detailed logs to identify failure points
                    6. **Consider different models** - try GPT-4-turbo or Claude for generation
                    """)

        # ==========================================================================
        # REGRESSION TRACKING SECTION
        # ==========================================================================
        st.divider()
        st.subheader("Regression Tracking")

        # Initialize tracker
        tracker = RegressionTracker()

        # Determine current evaluation type based on which results exist
        current_eval_type = None
        current_results = None
        if st.session_state.ragas_results:
            current_eval_type = "ragas"
            current_results = st.session_state.ragas_results
        elif st.session_state.judge_results:
            current_eval_type = "llm_judge"
            current_results = st.session_state.judge_results

        # Get baseline for current type
        baseline = tracker.get_baseline(current_eval_type) if current_eval_type else None

        # Baseline Management Panel
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if baseline:
                st.success(f"**Active Baseline:** {baseline.created_at[:16]}")
                st.caption(f"Type: {baseline.evaluation_type.upper().replace('_', '-')} | Description: {baseline.description or 'No description'}")
                with st.expander("Baseline Scores"):
                    for metric, score in baseline.aggregate_scores.items():
                        if baseline.evaluation_type == "llm_judge":
                            st.markdown(f"- **{metric.replace('_', ' ').title()}:** {score * 5:.2f}/5 ({score:.1%})")
                        else:
                            st.markdown(f"- **{metric.replace('_', ' ').title()}:** {score:.1%}")
            else:
                st.info("No baseline set. Run an evaluation and set it as baseline to enable regression tracking.")

        with col2:
            if current_results:
                baseline_desc = st.text_input("Description:", key="baseline_desc_ragas",
                                               placeholder="e.g., v1.0 release")
                if st.button("Set as Baseline", type="primary", use_container_width=True, key="set_baseline_ragas"):
                    run = tracker.record_run(current_eval_type, current_results)
                    tracker.set_baseline(run.run_id, description=baseline_desc)
                    st.success("Baseline set!")
                    st.rerun()

        with col3:
            if baseline:
                if st.button("Clear Baseline", type="secondary", use_container_width=True, key="clear_baseline_ragas"):
                    tracker.clear_baseline(current_eval_type)
                    st.success("Baseline cleared")
                    st.rerun()

        # Delta Comparison Display (when results and baseline exist)
        if current_results and baseline:
            st.divider()
            st.markdown("### Comparison with Baseline")

            # Record current run
            current_run = tracker.record_run(current_eval_type, current_results)
            deltas = tracker.calculate_deltas(current_run.run_id, compare_to="baseline")

            if "error" not in deltas:
                aggregate_deltas = deltas.get("aggregate", {})
                current_scores = current_run.aggregate_scores

                # Display aggregate deltas with visual indicators
                delta_cols = st.columns(len(aggregate_deltas) if aggregate_deltas else 1)
                for i, (metric, delta) in enumerate(aggregate_deltas.items()):
                    with delta_cols[i % len(delta_cols)]:
                        current_val = current_scores.get(metric, 0)

                        # Format based on evaluation type
                        if current_eval_type == "llm_judge":
                            display_val = f"{current_val * 5:.2f}/5"
                            delta_display = f"{delta * 5:+.2f}"
                        else:
                            display_val = f"{current_val:.1%}"
                            delta_display = f"{delta:+.1%}"

                        st.metric(
                            label=metric.replace("_", " ").title(),
                            value=display_val,
                            delta=delta_display,
                            delta_color="normal" if delta >= 0 else "inverse"
                        )

                # Regression Alerts
                alerts = tracker.check_regressions(current_run.run_id)

                if alerts:
                    st.divider()
                    st.markdown("### Regression Alerts")

                    critical_alerts = [a for a in alerts if a["severity"] == "critical"]
                    warning_alerts = [a for a in alerts if a["severity"] == "warning"]

                    if critical_alerts:
                        st.error(f"**{len(critical_alerts)} Critical Regression(s) Detected**")
                        for alert in critical_alerts:
                            st.markdown(f"- **{alert['metric'].replace('_', ' ').title()}**: {alert['message']}")

                    if warning_alerts:
                        st.warning(f"**{len(warning_alerts)} Warning-Level Change(s)**")
                        for alert in warning_alerts:
                            st.markdown(f"- {alert['message']}")
                else:
                    st.success("No regressions detected - all metrics within threshold!")

        # Trend Visualization
        history = tracker.get_history(current_eval_type, limit=10) if current_eval_type else []

        if len(history) >= 2:
            st.divider()
            st.markdown("### Performance Trends")

            import pandas as pd

            # Build dataframe for charting
            trend_data = []
            for run in reversed(history):  # Oldest first for chart
                row = {"Run": run.timestamp[5:16]}  # Short timestamp
                for metric, score in run.aggregate_scores.items():
                    if current_eval_type == "llm_judge":
                        row[metric.replace("_", " ").title()] = score * 5  # Convert to 1-5 scale
                    else:
                        row[metric.replace("_", " ").title()] = score * 100  # Convert to percentage
                trend_data.append(row)

            df_trends = pd.DataFrame(trend_data)

            if len(df_trends.columns) > 1:
                # Line chart for key metrics
                metrics_to_chart = [m for m in df_trends.columns if m != "Run"]

                st.line_chart(
                    df_trends.set_index("Run")[metrics_to_chart],
                    use_container_width=True
                )

                if current_eval_type == "llm_judge":
                    st.caption("Y-axis: Score (1-5 scale)")
                else:
                    st.caption("Y-axis: Score (0-100%)")

        # ==========================================================================
        # EVALUATION HISTORY SECTION
        # ==========================================================================
        if st.session_state.evaluation_history:
            st.divider()
            st.subheader("Evaluation History")
            st.caption(f"Total evaluations: {len(st.session_state.evaluation_history)}")

            # Create comparison table
            import pandas as pd

            history_data = []
            for i, eval_run in enumerate(reversed(st.session_state.evaluation_history)):
                row = {
                    '#': len(st.session_state.evaluation_history) - i,
                    'Timestamp': eval_run.get('timestamp', 'N/A'),
                    'Method': eval_run.get('method', 'N/A'),
                    'Questions': eval_run.get('num_questions', 'N/A'),
                }

                scores = eval_run.get('scores', {})
                if eval_run.get('method') == 'RAGAS':
                    # RAGAS scores (0-1 scale, convert to %)
                    faith = scores.get('faithfulness')
                    if isinstance(faith, list):
                        faith = sum(faith) / len(faith) if faith else 0
                    row['Faithfulness'] = f"{faith:.1%}" if faith is not None else 'N/A'

                    ans_rel = scores.get('answer_relevancy')
                    if isinstance(ans_rel, list):
                        ans_rel = sum(ans_rel) / len(ans_rel) if ans_rel else 0
                    row['Answer Rel.'] = f"{ans_rel:.1%}" if ans_rel is not None else 'N/A'

                    precision = scores.get('context_precision')
                    if isinstance(precision, list):
                        precision = sum(precision) / len(precision) if precision else 0
                    row['Ctx Precision'] = f"{precision:.1%}" if precision is not None else 'N/A'

                    recall = scores.get('context_recall')
                    if isinstance(recall, list):
                        recall = sum(recall) / len(recall) if recall else 0
                    row['Ctx Recall'] = f"{recall:.1%}" if recall is not None else 'N/A'

                    # LLM-as-a-Judge columns (N/A for RAGAS)
                    row['Correctness'] = 'N/A'
                    row['Relevancy'] = 'N/A'
                    row['Average'] = 'N/A'
                else:
                    # RAGAS columns (N/A for LLM-as-a-Judge)
                    row['Faithfulness'] = 'N/A'
                    row['Answer Rel.'] = 'N/A'
                    row['Ctx Precision'] = 'N/A'
                    row['Ctx Recall'] = 'N/A'

                    # LLM-as-a-Judge scores (1-5 scale)
                    row['Correctness'] = f"{scores.get('correctness', 0):.2f}/5"
                    row['Relevancy'] = f"{scores.get('relevancy', 0):.2f}/5"
                    row['Average'] = f"{scores.get('average', 0):.2f}/5"

                history_data.append(row)

            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)

            # Clear history button
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Clear History", type="secondary"):
                    st.session_state.evaluation_history = []
                    st.session_state.ragas_results = None
                    st.session_state.judge_results = None
                    save_evaluation_history([])  # Clear file too
                    st.rerun()

    # ==========================================================================
    # TAB 4: Code-Based Graders
    # ==========================================================================
    elif selected_tab == "🧪 Code-Based Graders":
        st.markdown('<div class="step-header"><h3 style="margin:0;">Code-Based Evaluation Graders</h3></div>', unsafe_allow_html=True)

        st.info("""
        **Code-Based Graders** provide deterministic evaluation - fast, cheap, objective, reproducible, and easy to debug.
        All graders use **binary pass/fail scoring** (0 or 1) with **no LLM calls required**.

        Based on Anthropic's "Demystifying Evals for AI Agents" recommendations.
        """)

        # Show grader types overview
        with st.expander("View All Grader Types (36 Total Tests)", expanded=False):
            st.markdown("""
            | Grader Type | Tests | Description |
            |-------------|-------|-------------|
            | **Exact Match & Regex** | 10 | Query RAG system, verify exact values in response |
            | **Numerical Validation** | 5 | Validate amounts with tolerance (±$0.01 or ±1%) |
            | **Consistency Checking** | 3 | Verify chronological order, sum constraints, name consistency |
            | **Key Fact Coverage** | 5 | Check all required facts present per topic |
            | **Fuzzy String Matching** | 5 | Handle name variations with similarity threshold (80%+) |
            | **Standalone Regex** | 8 | Validate regex patterns against sample text |
            """)

        # Import code-based graders
        from src.evaluation.code_graders import CodeBasedGraders, GROUND_TRUTH, REGEX_PATTERNS, GROUND_TRUTH_NUMERICAL, FACT_GROUPS
        from src.evaluation.code_grader_tests import CodeGraderTestSuite

        # Initialize session state for code graders
        if 'code_grader_results' not in st.session_state:
            st.session_state.code_grader_results = None
        if 'code_grader_mode' not in st.session_state:
            st.session_state.code_grader_mode = "Exact Match & Regex"

        # Grader type selector
        grader_type = st.radio(
            "Select Grader Type:",
            [
                "Exact Match & Regex",
                "Numerical Validation",
                "Consistency Checking",
                "Key Fact Coverage",
                "Fuzzy String Matching",
                "Standalone Regex Validation"
            ],
            horizontal=True,
            help="Select the type of code-based evaluation to run against the RAG system."
        )
        st.session_state.code_grader_mode = grader_type

        st.divider()

        if grader_type == "Exact Match & Regex":
            st.subheader("RAG Response Grading")
            st.markdown("Query the RAG system, then apply code-based graders to evaluate the response.")

            # Get RAG test cases
            rag_test_cases = CodeGraderTestSuite.get_rag_test_cases()

            # Display test cases with selection
            st.markdown(f"**Available Test Cases ({len(rag_test_cases)})**")

            # Build selection table
            import pandas as pd
            table_data = []
            for tc in rag_test_cases:
                table_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Query": tc["query"][:60] + "..." if len(tc["query"]) > 60 else tc["query"],
                    "Expected": tc["expected_value"],
                    "Category": tc.get("category", "other")
                })

            df = pd.DataFrame(table_data)

            # Editable dataframe for selection
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Expected": st.column_config.TextColumn("Expected", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Query", "Expected", "Category"],
                hide_index=True,
                use_container_width=True,
                key="rag_grader_table"
            )

            # Run evaluation button
            selected_count = edited_df["Select"].sum()
            st.markdown(f"**Selected: {selected_count} test cases**")

            if st.button("🚀 Run RAG Grading", type="primary", disabled=selected_count == 0):
                if 'system' not in st.session_state or st.session_state.system is None:
                    st.error("Please upload and index a document first (use the Query tab)")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    selected_indices = edited_df[edited_df["Select"]].index.tolist()

                    for i, idx in enumerate(selected_indices):
                        test_case = rag_test_cases[idx]
                        status_text.text(f"Running test {i+1}/{len(selected_indices)}: {test_case['id']}")

                        # Query RAG system
                        try:
                            rag_result = st.session_state.system.query(test_case["query"])
                            answer = rag_result.get("output", "")

                            # Grade the response
                            grade_result = CodeBasedGraders.run_rag_test(answer, test_case)
                            grade_result["rag_answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                            results.append(grade_result)

                        except Exception as e:
                            results.append({
                                "test_id": test_case["id"],
                                "passed": False,
                                "score": 0,
                                "details": f"Error: {str(e)}",
                                "rag_answer": ""
                            })

                        progress_bar.progress((i + 1) / len(selected_indices))

                    status_text.text("Evaluation complete!")
                    st.session_state.code_grader_results = {
                        "mode": "RAG Response Grading",
                        "results": results,
                        "summary": CodeBasedGraders.calculate_summary(results),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()

            # Display results
            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") in ["RAG Response Grading", "Exact Match & Regex"]:
                st.divider()
                st.subheader("Results")

                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    pass_color = "green" if summary["pass_rate"] >= 80 else "orange" if summary["pass_rate"] >= 60 else "red"
                    st.metric("Passed", summary["passed"], delta=None)
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                # Detailed results
                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Expected": r.get("expected", ""),
                        "Found": r.get("found", "N/A"),
                        "RAG Answer": r.get("rag_answer", "")[:100] + "..."
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

                # Export button
                csv_data = pd.DataFrame(results_table).to_csv(index=False)
                st.download_button(
                    "📥 Export Results (CSV)",
                    csv_data,
                    f"code_grader_rag_results_{results_data['timestamp'].replace(' ', '_').replace(':', '-')}.csv",
                    "text/csv"
                )

        elif grader_type == "Numerical Validation":
            st.subheader("Numerical Validation Grading")
            st.markdown("Validate numerical values (amounts, percentages, counts) with configurable tolerance.")

            # Get numerical validation test cases
            num_test_cases = CodeGraderTestSuite.get_numerical_validation_test_cases()

            st.markdown(f"**Available Test Cases ({len(num_test_cases)})**")

            import pandas as pd
            table_data = []
            for tc in num_test_cases:
                table_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Query": tc["query"][:50] + "..." if len(tc["query"]) > 50 else tc["query"],
                    "Expected": tc["expected_value"],
                    "Tolerance": f"{tc['tolerance_value']} ({tc['tolerance_type']})",
                    "Category": tc.get("category", "other")
                })

            df = pd.DataFrame(table_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Expected": st.column_config.NumberColumn("Expected", width="small"),
                    "Tolerance": st.column_config.TextColumn("Tolerance", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Query", "Expected", "Tolerance", "Category"],
                hide_index=True,
                use_container_width=True,
                key="numerical_grader_table"
            )

            selected_count = edited_df["Select"].sum()
            st.markdown(f"**Selected: {selected_count} test cases**")

            if st.button("🚀 Run Numerical Validation", type="primary", disabled=selected_count == 0):
                if 'system' not in st.session_state or st.session_state.system is None:
                    st.error("Please upload and index a document first (use the Query tab)")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    selected_indices = edited_df[edited_df["Select"]].index.tolist()

                    for i, idx in enumerate(selected_indices):
                        test_case = num_test_cases[idx]
                        status_text.text(f"Running test {i+1}/{len(selected_indices)}: {test_case['id']}")

                        try:
                            rag_result = st.session_state.system.query(test_case["query"])
                            answer = rag_result.get("output", "")
                            grade_result = CodeBasedGraders.run_rag_test(answer, test_case)
                            grade_result["rag_answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                            results.append(grade_result)
                        except Exception as e:
                            results.append({
                                "test_id": test_case["id"],
                                "passed": False,
                                "score": 0,
                                "details": f"Error: {str(e)}",
                                "rag_answer": ""
                            })

                        progress_bar.progress((i + 1) / len(selected_indices))

                    status_text.text("Evaluation complete!")
                    st.session_state.code_grader_results = {
                        "mode": "Numerical Validation",
                        "results": results,
                        "summary": CodeBasedGraders.calculate_summary(results),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()

            # Display results
            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") == "Numerical Validation":
                st.divider()
                st.subheader("Results")
                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    st.metric("Passed", summary["passed"])
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Expected": r.get("expected_value", ""),
                        "Found": r.get("found_value", "N/A"),
                        "Difference": f"{r.get('difference', 'N/A'):.4f}" if r.get('difference') is not None else "N/A",
                        "RAG Answer": r.get("rag_answer", "")[:80] + "..."
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

        elif grader_type == "Consistency Checking":
            st.subheader("Consistency Checking")
            st.markdown("Verify internal consistency of facts (chronological order, sum constraints, name consistency).")

            cons_test_cases = CodeGraderTestSuite.get_consistency_check_test_cases()

            st.markdown(f"**Available Test Cases ({len(cons_test_cases)})**")

            import pandas as pd
            table_data = []
            for tc in cons_test_cases:
                table_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Query": tc["query"][:50] + "..." if len(tc["query"]) > 50 else tc["query"],
                    "Check Type": tc["check_type"],
                    "Category": tc.get("category", "other")
                })

            df = pd.DataFrame(table_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Check Type": st.column_config.TextColumn("Check Type", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Query", "Check Type", "Category"],
                hide_index=True,
                use_container_width=True,
                key="consistency_grader_table"
            )

            selected_count = edited_df["Select"].sum()
            st.markdown(f"**Selected: {selected_count} test cases**")

            if st.button("🚀 Run Consistency Checks", type="primary", disabled=selected_count == 0):
                if 'system' not in st.session_state or st.session_state.system is None:
                    st.error("Please upload and index a document first (use the Query tab)")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    selected_indices = edited_df[edited_df["Select"]].index.tolist()

                    for i, idx in enumerate(selected_indices):
                        test_case = cons_test_cases[idx]
                        status_text.text(f"Running test {i+1}/{len(selected_indices)}: {test_case['id']}")

                        try:
                            rag_result = st.session_state.system.query(test_case["query"])
                            answer = rag_result.get("output", "")
                            grade_result = CodeBasedGraders.run_rag_test(answer, test_case)
                            grade_result["rag_answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                            results.append(grade_result)
                        except Exception as e:
                            results.append({
                                "test_id": test_case["id"],
                                "passed": False,
                                "score": 0,
                                "details": f"Error: {str(e)}",
                                "rag_answer": ""
                            })

                        progress_bar.progress((i + 1) / len(selected_indices))

                    status_text.text("Evaluation complete!")
                    st.session_state.code_grader_results = {
                        "mode": "Consistency Checking",
                        "results": results,
                        "summary": CodeBasedGraders.calculate_summary(results),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()

            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") == "Consistency Checking":
                st.divider()
                st.subheader("Results")
                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    st.metric("Passed", summary["passed"])
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Check Type": r.get("check_type", ""),
                        "Details": r.get("details", "")[:80],
                        "RAG Answer": r.get("rag_answer", "")[:60] + "..."
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

        elif grader_type == "Key Fact Coverage":
            st.subheader("Key Fact Coverage")
            st.markdown("Check if responses contain all required facts for completeness.")

            fact_test_cases = CodeGraderTestSuite.get_key_fact_coverage_test_cases()

            st.markdown(f"**Available Test Cases ({len(fact_test_cases)})**")

            import pandas as pd
            table_data = []
            for tc in fact_test_cases:
                table_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Query": tc["query"][:50] + "..." if len(tc["query"]) > 50 else tc["query"],
                    "Fact Group": tc["fact_group"],
                    "Category": tc.get("category", "other")
                })

            df = pd.DataFrame(table_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Fact Group": st.column_config.TextColumn("Fact Group", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Query", "Fact Group", "Category"],
                hide_index=True,
                use_container_width=True,
                key="coverage_grader_table"
            )

            selected_count = edited_df["Select"].sum()
            st.markdown(f"**Selected: {selected_count} test cases**")

            if st.button("🚀 Run Fact Coverage Check", type="primary", disabled=selected_count == 0):
                if 'system' not in st.session_state or st.session_state.system is None:
                    st.error("Please upload and index a document first (use the Query tab)")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    selected_indices = edited_df[edited_df["Select"]].index.tolist()

                    for i, idx in enumerate(selected_indices):
                        test_case = fact_test_cases[idx]
                        status_text.text(f"Running test {i+1}/{len(selected_indices)}: {test_case['id']}")

                        try:
                            rag_result = st.session_state.system.query(test_case["query"])
                            answer = rag_result.get("output", "")
                            grade_result = CodeBasedGraders.run_rag_test(answer, test_case)
                            grade_result["rag_answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                            results.append(grade_result)
                        except Exception as e:
                            results.append({
                                "test_id": test_case["id"],
                                "passed": False,
                                "score": 0,
                                "details": f"Error: {str(e)}",
                                "rag_answer": ""
                            })

                        progress_bar.progress((i + 1) / len(selected_indices))

                    status_text.text("Evaluation complete!")
                    st.session_state.code_grader_results = {
                        "mode": "Key Fact Coverage",
                        "results": results,
                        "summary": CodeBasedGraders.calculate_summary(results),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()

            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") == "Key Fact Coverage":
                st.divider()
                st.subheader("Results")
                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    st.metric("Passed", summary["passed"])
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    missing = [f["key"] for f in r.get("facts_missing", [])]
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Fact Group": r.get("fact_group", ""),
                        "Coverage": f"{r.get('facts_found_count', 0)}/{r.get('total_facts', 0)}",
                        "Missing Facts": ", ".join(missing) if missing else "None"
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

        elif grader_type == "Fuzzy String Matching":
            st.subheader("Fuzzy String Matching")
            st.markdown("Handle name and string variations using similarity matching.")

            fuzzy_test_cases = CodeGraderTestSuite.get_fuzzy_match_test_cases()

            st.markdown(f"**Available Test Cases ({len(fuzzy_test_cases)})**")

            import pandas as pd
            table_data = []
            for tc in fuzzy_test_cases:
                table_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Query": tc["query"][:50] + "..." if len(tc["query"]) > 50 else tc["query"],
                    "Expected": tc["expected_value"],
                    "Threshold": f"{tc['similarity_threshold']*100:.0f}%",
                    "Category": tc.get("category", "other")
                })

            df = pd.DataFrame(table_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Query": st.column_config.TextColumn("Query", width="large"),
                    "Expected": st.column_config.TextColumn("Expected", width="medium"),
                    "Threshold": st.column_config.TextColumn("Threshold", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Query", "Expected", "Threshold", "Category"],
                hide_index=True,
                use_container_width=True,
                key="fuzzy_grader_table"
            )

            selected_count = edited_df["Select"].sum()
            st.markdown(f"**Selected: {selected_count} test cases**")

            if st.button("🚀 Run Fuzzy Matching", type="primary", disabled=selected_count == 0):
                if 'system' not in st.session_state or st.session_state.system is None:
                    st.error("Please upload and index a document first (use the Query tab)")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    selected_indices = edited_df[edited_df["Select"]].index.tolist()

                    for i, idx in enumerate(selected_indices):
                        test_case = fuzzy_test_cases[idx]
                        status_text.text(f"Running test {i+1}/{len(selected_indices)}: {test_case['id']}")

                        try:
                            rag_result = st.session_state.system.query(test_case["query"])
                            answer = rag_result.get("output", "")
                            grade_result = CodeBasedGraders.run_rag_test(answer, test_case)
                            grade_result["rag_answer"] = answer[:200] + "..." if len(answer) > 200 else answer
                            results.append(grade_result)
                        except Exception as e:
                            results.append({
                                "test_id": test_case["id"],
                                "passed": False,
                                "score": 0,
                                "details": f"Error: {str(e)}",
                                "rag_answer": ""
                            })

                        progress_bar.progress((i + 1) / len(selected_indices))

                    status_text.text("Evaluation complete!")
                    st.session_state.code_grader_results = {
                        "mode": "Fuzzy String Matching",
                        "results": results,
                        "summary": CodeBasedGraders.calculate_summary(results),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.rerun()

            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") == "Fuzzy String Matching":
                st.divider()
                st.subheader("Results")
                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    st.metric("Passed", summary["passed"])
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Expected": r.get("expected_value", ""),
                        "Best Match": r.get("best_match", "N/A"),
                        "Similarity": f"{r.get('similarity_ratio', 0)*100:.1f}%",
                        "Threshold": f"{r.get('similarity_threshold', 0)*100:.0f}%"
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

        elif grader_type == "Standalone Regex Validation":
            st.subheader("Standalone Regex Validation")
            st.markdown("Test regex patterns against sample text to verify pattern correctness.")

            # Get regex test cases
            regex_test_cases = CodeGraderTestSuite.get_regex_test_cases()

            # Display available patterns
            st.markdown(f"**Available Regex Patterns ({len(regex_test_cases)})**")

            import pandas as pd
            pattern_data = []
            for tc in regex_test_cases:
                pattern_data.append({
                    "Select": True,
                    "ID": tc["id"],
                    "Pattern Name": tc["pattern_name"],
                    "Regex": tc["regex_pattern"][:40] + "..." if len(tc["regex_pattern"]) > 40 else tc["regex_pattern"],
                    "Category": tc.get("category", "other")
                })

            df_patterns = pd.DataFrame(pattern_data)

            edited_patterns = st.data_editor(
                df_patterns,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=True),
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Pattern Name": st.column_config.TextColumn("Pattern", width="medium"),
                    "Regex": st.column_config.TextColumn("Regex", width="large"),
                    "Category": st.column_config.TextColumn("Category", width="small")
                },
                disabled=["ID", "Pattern Name", "Regex", "Category"],
                hide_index=True,
                use_container_width=True,
                key="regex_grader_table"
            )

            selected_pattern_count = edited_patterns["Select"].sum()
            st.markdown(f"**Selected: {selected_pattern_count} patterns**")

            if st.button("🚀 Run Regex Validation", type="primary", disabled=selected_pattern_count == 0):
                results = []
                selected_indices = edited_patterns[edited_patterns["Select"]].index.tolist()

                for idx in selected_indices:
                    test_case = regex_test_cases[idx]
                    result = CodeBasedGraders.run_standalone_regex_test(test_case["pattern_name"])
                    result["test_id"] = test_case["id"]
                    result["description"] = test_case["description"]
                    results.append(result)

                st.session_state.code_grader_results = {
                    "mode": "Standalone Regex Validation",
                    "results": results,
                    "summary": CodeBasedGraders.calculate_summary(results),
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.rerun()

            # Display results
            if st.session_state.code_grader_results and st.session_state.code_grader_results.get("mode") == "Standalone Regex Validation":
                st.divider()
                st.subheader("Results")

                results_data = st.session_state.code_grader_results
                summary = results_data["summary"]

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Tests", summary["total_tests"])
                with col2:
                    st.metric("Passed", summary["passed"])
                with col3:
                    st.metric("Failed", summary["failed"])
                with col4:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")

                # Detailed results
                st.markdown("**Detailed Results**")
                results_table = []
                for r in results_data["results"]:
                    results_table.append({
                        "Test ID": r.get("test_id", ""),
                        "Pattern": r.get("pattern_name", ""),
                        "Status": "✅ PASS" if r.get("passed") else "❌ FAIL",
                        "Matches Found": str(r.get("matches", []))[:50],
                        "Sample Text": r.get("sample_text", "")[:60] + "..."
                    })

                st.dataframe(pd.DataFrame(results_table), use_container_width=True, hide_index=True)

                # Show full pattern details in expander
                with st.expander("View Full Pattern Details"):
                    for r in results_data["results"]:
                        status_icon = "✅" if r.get("passed") else "❌"
                        st.markdown(f"**{status_icon} {r.get('pattern_name', '')}**")
                        st.code(r.get("pattern", ""), language="regex")
                        st.markdown(f"Matches: `{r.get('matches', [])}`")
                        st.markdown(f"Sample: {r.get('sample_text', '')}")
                        st.divider()

                # Export button
                csv_data = pd.DataFrame(results_table).to_csv(index=False)
                st.download_button(
                    "📥 Export Results (CSV)",
                    csv_data,
                    f"code_grader_regex_results_{results_data['timestamp'].replace(' ', '_').replace(':', '-')}.csv",
                    "text/csv"
                )

        # ==========================================================================
        # CODE GRADERS REGRESSION TRACKING SECTION
        # ==========================================================================
        # Always show regression tracking for the currently selected grader type
        st.divider()
        st.subheader("Regression Tracking")

        # Initialize tracker
        code_tracker = RegressionTracker()

        # Map grader_type (radio selection) to evaluation type
        grader_type_to_eval = {
            "Exact Match & Regex": "code_graders_rag_response_grading",
            "Numerical Validation": "code_graders_numerical_validation",
            "Consistency Checking": "code_graders_consistency_checking",
            "Key Fact Coverage": "code_graders_key_fact_coverage",
            "Fuzzy String Matching": "code_graders_fuzzy_string_matching",
            "Standalone Regex Validation": "code_graders_standalone_regex_validation"
        }

        # Use the currently selected grader type (not from results)
        eval_type = grader_type_to_eval.get(grader_type, "code_graders_unknown")
        grader_subtype = eval_type.replace("code_graders_", "")

        # Get baseline for the CURRENTLY SELECTED grader type
        code_baseline = code_tracker.get_baseline(eval_type)

        # Check if we have results that match the currently selected grader type
        results_match_selected = False
        if st.session_state.code_grader_results:
            results_mode = st.session_state.code_grader_results.get("mode", "")
            # Map results mode to eval_type for comparison
            results_eval_type = f"code_graders_{results_mode.lower().replace(' ', '_').replace('&', 'and')}"
            results_match_selected = (results_eval_type == eval_type)

        # Baseline Management Panel - always show for the selected grader type
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if code_baseline:
                baseline_pass_rate = code_baseline.aggregate_scores.get("pass_rate", 0) * 100
                st.success(f"**Active Baseline:** {code_baseline.created_at[:16]}")
                st.caption(f"Type: {grader_type} | Pass Rate: {baseline_pass_rate:.1f}% | {code_baseline.description or 'No description'}")
            else:
                st.info(f"No baseline set for {grader_type}. Run tests and set as baseline to track regressions.")

        with col2:
            baseline_desc_code = st.text_input("Description:", key=f"baseline_desc_{grader_subtype}",
                                                placeholder="e.g., v1.0 release")
            # Only enable "Set as Baseline" if we have results for this grader type
            set_baseline_disabled = not results_match_selected
            if st.button("Set as Baseline", type="primary", use_container_width=True,
                        key=f"set_baseline_{grader_subtype}", disabled=set_baseline_disabled):
                code_run = code_tracker.record_run(eval_type, st.session_state.code_grader_results, grader_subtype=grader_subtype)
                code_tracker.set_baseline(code_run.run_id, description=baseline_desc_code)
                st.success("Baseline set!")
                st.rerun()
            if set_baseline_disabled and not code_baseline:
                st.caption("Run tests first to set baseline")

        with col3:
            if code_baseline:
                if st.button("Clear Baseline", type="secondary", use_container_width=True, key=f"clear_baseline_{grader_subtype}"):
                    code_tracker.clear_baseline(eval_type)
                    st.success("Baseline cleared")
                    st.rerun()

        # Delta Comparison Display - only show when we have results for the selected grader type
        if code_baseline and results_match_selected:
            st.divider()
            st.markdown("### Comparison with Baseline")

            # Record current run and calculate deltas
            code_run = code_tracker.record_run(eval_type, st.session_state.code_grader_results, grader_subtype=grader_subtype)

            current_pass_rate = st.session_state.code_grader_results["summary"]["pass_rate"]
            baseline_pass_rate = code_baseline.aggregate_scores.get("pass_rate", 0) * 100
            delta = current_pass_rate - baseline_pass_rate

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Pass Rate vs Baseline",
                    f"{current_pass_rate:.1f}%",
                    f"{delta:+.1f}%",
                    delta_color="normal" if delta >= 0 else "inverse"
                )
            with col2:
                st.metric(
                    "Tests Passed",
                    f"{st.session_state.code_grader_results['summary']['passed']}/{st.session_state.code_grader_results['summary']['total_tests']}",
                )
            with col3:
                st.metric(
                    "Baseline Pass Rate",
                    f"{baseline_pass_rate:.1f}%",
                )

            # Per-Query Regression Table
            st.markdown("**Per-Query Comparison**")

            comparison_data = []
            for result in st.session_state.code_grader_results["results"]:
                query_id = result.get("test_id", "unknown")
                current_passed = result.get("passed", False)

                # Get baseline status for this query
                baseline_query = code_baseline.per_query_scores.get(query_id, {})
                baseline_passed_score = baseline_query.get("passed")
                baseline_passed = baseline_passed_score == 1.0 if baseline_passed_score is not None else None

                # Determine status
                if baseline_passed is None:
                    status = "NEW"
                    status_icon = "🆕"
                elif current_passed and not baseline_passed:
                    status = "IMPROVED"
                    status_icon = "📈"
                elif not current_passed and baseline_passed:
                    status = "REGRESSED"
                    status_icon = "📉"
                else:
                    status = "UNCHANGED"
                    status_icon = "➖"

                comparison_data.append({
                    "Query ID": query_id,
                    "Current": "✅ PASS" if current_passed else "❌ FAIL",
                    "Baseline": "✅ PASS" if baseline_passed else ("❌ FAIL" if baseline_passed is not None else "N/A"),
                    "Status": f"{status_icon} {status}"
                })

            df_comparison = pd.DataFrame(comparison_data)

            # Highlight regressions and improvements
            def highlight_status(row):
                if "REGRESSED" in row["Status"]:
                    return ["background-color: #ffcccc"] * len(row)
                elif "IMPROVED" in row["Status"]:
                    return ["background-color: #ccffcc"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df_comparison.style.apply(highlight_status, axis=1),
                hide_index=True,
                use_container_width=True
            )

            # Regression Alerts
            alerts = code_tracker.check_regressions(code_run.run_id)

            if alerts:
                st.divider()
                critical_alerts = [a for a in alerts if a["severity"] == "critical"]
                warning_alerts = [a for a in alerts if a["severity"] == "warning"]

                if critical_alerts:
                    st.error(f"**{len(critical_alerts)} Critical Regression(s) Detected**")
                    for alert in critical_alerts:
                        st.markdown(f"- {alert['message']}")

                if warning_alerts:
                    st.warning(f"**{len(warning_alerts)} Warning-Level Change(s)**")
                    for alert in warning_alerts:
                        st.markdown(f"- {alert['message']}")
            else:
                st.success("No regressions detected - all metrics within threshold!")

        # Ground Truth Reference
        with st.expander("📋 View Ground Truth Reference"):
            st.markdown("**Expected values from insurance_claim_CLM2024001.pdf:**")
            gt_table = []
            for key, value in GROUND_TRUTH.items():
                gt_table.append({"Field": key, "Expected Value": value})
            st.dataframe(pd.DataFrame(gt_table), use_container_width=True, hide_index=True)

        # Regex Patterns Reference
        with st.expander("🔍 View Regex Patterns Reference"):
            st.markdown("**Available regex patterns for extraction:**")
            pattern_table = []
            for name, pattern in REGEX_PATTERNS.items():
                pattern_table.append({"Pattern Name": name, "Regex": pattern})
            st.dataframe(pd.DataFrame(pattern_table), use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption("Insurance Claim RAG System | LlamaIndex + LangGraph + ChromaDB + RAGAS")
