# src/retriever.py

import pickle
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import faiss
import numpy as np
from nltk.stem.snowball import SnowballStemmer

# --- ADAPTATION: Import our new, simplified config ---
from src.config import config 

import logging
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Handles dense (FAISS) and sparse (BM25) search. This version is adapted for
    our specific project with an English stemmer and simplified logic.
    """
    def __init__(self):
        logger.info("Initializing HybridRetriever...")
        self.paths = config.paths
        self.faiss_index, self.chunks_metadata = self._load_faiss_and_metadata()
        self.bm25_index = self._load_bm25_index(self.paths.LATEST_BM25_INDEX_PATH)
        
        # --- ADAPTATION: CRITICAL FIX - Changed stemmer to English ---
        self.stemmer = SnowballStemmer("english")
        
        # Validation check
        if self.bm25_index and self.bm25_index.corpus_size != len(self.chunks_metadata):
            logger.error(
                f"CRITICAL MISMATCH: BM25 index size ({self.bm25_index.corpus_size}) "
                f"vs metadata size ({len(self.chunks_metadata)})."
            )

    def _load_faiss_and_metadata(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Loads the FAISS index and the corresponding chunk metadata."""
        index_path = self.paths.LATEST_FAISS_INDEX_PATH
        metadata_path = self.paths.LATEST_METADATA_PATH
        
        logger.info(f"Loading FAISS index from: {index_path}")
        index = faiss.read_index(str(index_path))
        
        logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        if index.ntotal != len(metadata):
            raise ValueError("Mismatch between FAISS index size and metadata count.")
            
        logger.info(f"Successfully loaded FAISS index ({index.ntotal} vectors) and metadata.")
        return index, metadata

    def _load_bm25_index(self, path: Path) -> Optional[Any]:
        """Loads the pickled BM25 index file."""
        logger.info(f"Loading BM25 index from: {path}")
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {path}: {e}", exc_info=True)
            return None

    def _stemmed_tokenizer(self, text: str) -> List[str]:
        """Tokenizes and stems text for the English language."""
        if not text or not isinstance(text, str):
            return []
        try:
            # Simple regex to remove punctuation and convert to lowercase
            text = re.sub(r'[^\w\s]', '', str(text)).lower()
            tokens = text.split()
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            return stemmed_tokens
        except Exception as e:
            logger.warning(f"Tokenization failed for text '{text[:50]}...': {e}")
            return str(text).lower().split()

    def search(self, query_text: str, query_embedding: List[float], top_k: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Performs a hybrid search and returns two separate, ranked lists:
        one from FAISS (semantic) and one from BM25 (keyword).
        """
        if not query_text or not query_embedding:
            logger.warning("Empty query or embedding provided. Returning empty search results.")
            return [], []

        # --- 1. FAISS Search (Semantic) ---
        faiss_results = []
        try:
            query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_np, top_k)
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    result_meta = self.chunks_metadata[int(idx)].copy()
                    # Convert L2 distance to a similarity score (0-1)
                    score = 1.0 / (1.0 + np.sqrt(float(distances[0][i])))
                    result_meta['semantic_score'] = score
                    faiss_results.append(result_meta)
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)

        # --- 2. BM25 Search (Keyword) ---
        bm25_results = []
        if self.bm25_index:
            try:
                tokenized_query = self._stemmed_tokenizer(query_text)
                doc_scores = self.bm25_index.get_scores(tokenized_query)
                
                # Get the indices of the top_k scores
                top_n_indices = np.argsort(doc_scores)[-top_k:][::-1]
                
                for idx in top_n_indices:
                    score = doc_scores[idx]
                    if score > 0:
                        # Fetch the full metadata for the result
                        chunk_meta = self.chunks_metadata[int(idx)].copy()
                        chunk_meta['bm25_score'] = score  # Keep the raw BM25 score for now
                        bm25_results.append(chunk_meta)
            except Exception as e:
                logger.error(f"Error during BM25 search: {e}", exc_info=True)

        logger.info(f"Retriever found {len(faiss_results)} semantic candidates and {len(bm25_results)} keyword candidates.")
        return faiss_results, bm25_results