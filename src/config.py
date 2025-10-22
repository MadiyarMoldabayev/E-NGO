# src/config.py

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# --- Project Root and .env Loading ---
# Assumes this config file is in 'src/', so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from the .env file in the project root
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"INFO: Loaded environment variables from: {dotenv_path}", file=sys.stdout)
else:
    print(f"WARNING: .env file not found at {dotenv_path}. Relying on system environment variables.", file=sys.stderr)

# --- Configuration Dataclasses ---

@dataclass
class PathSettings:
    """Manages all file and directory paths for the application."""
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "data" / "vector_store"
    
    LATEST_FAISS_INDEX_PATH: Path = field(init=False)
    LATEST_METADATA_PATH: Path = field(init=False)
    LATEST_BM25_INDEX_PATH: Path = field(init=False)

    def __post_init__(self):
        """Finds the most recently created index files in the vector store directory."""
        self.LATEST_FAISS_INDEX_PATH = self._find_latest_file("faiss_index", ".bin")
        self.LATEST_METADATA_PATH = self._find_latest_file("chunks_metadata", ".pkl")
        self.LATEST_BM25_INDEX_PATH = self._find_latest_file("bm25_index", ".pkl")
        
        # Log the paths to be used for verification
        print(f"INFO: Targeting FAISS index: {self.LATEST_FAISS_INDEX_PATH}")
        print(f"INFO: Targeting Metadata file: {self.LATEST_METADATA_PATH}")
        print(f"INFO: Targeting BM25 index: {self.LATEST_BM25_INDEX_PATH}")

    def _find_latest_file(self, prefix: str, suffix: str) -> Path:
        """Helper function to find the most recent file with a given prefix and suffix."""
        try:
            if not self.VECTOR_STORE_DIR.exists():
                raise FileNotFoundError(f"Vector store directory not found at: {self.VECTOR_STORE_DIR}")
                
            files = sorted(
                self.VECTOR_STORE_DIR.glob(f"{prefix}_*{suffix}"), 
                key=os.path.getmtime, 
                reverse=True
            )
            
            if not files:
                raise FileNotFoundError(f"No '{prefix}*{suffix}' files found in {self.VECTOR_STORE_DIR}.")
                
            return files[0]
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}. Please ensure you have run 'build_indexes.py' successfully.", file=sys.stderr)
            # Exit the application if essential files are missing.
            sys.exit(1)

@dataclass
class LLMSettings:
    """Settings for the LLM API and prompts."""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    completion_model: str = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
    
    # This is the new, simplified system prompt for our RAG application.
    system_prompt: str = """
# ROLE
You are an expert AI assistant. Your purpose is to provide direct, accurate, and concise answers based ONLY on the provided document context.

# RESPONSE DIRECTIVES (MANDATORY)
1.  **STRICT GROUNDING:** Your entire answer MUST be derived directly from the information found in the `<document_context>`. You are strictly forbidden from using any outside knowledge or making assumptions.
2.  **DIRECT & CONCISE:** Get straight to the point. Do not use conversational filler like "Of course, I can help with that..." or "Based on the document...".
3.  **CITE YOUR SOURCES:** This is not implemented yet, but is a good practice to keep in mind.
4.  **HANDLE "I DON'T KNOW":** If the provided `<document_context>` does not contain the information needed to answer the question, you MUST respond with only this exact phrase: "The provided document does not contain enough information to answer this question."
"""

@dataclass
class AppSettings:
    """General application settings for the Streamlit UI."""
    app_title: str = "Chat with Your Document"
    app_icon: str = "📄"
    app_placeholder: str = "Ask any question about the document..."

@dataclass
class AppConfig:
    """Root configuration class holding all settings."""
    paths: PathSettings = field(default_factory=PathSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    app: AppSettings = field(default_factory=AppSettings)

# --- Singleton Instance ---
# This creates a single configuration object that can be imported and used anywhere in the app.
config = AppConfig()