# build_indexes.py (Version 2.0 - Robust and Validated)

import os
import sys
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple

import faiss
import numpy as np
import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_MD_FILE = "251010-INPAS-Standard-Final.md"
DATA_OUTPUT_DIRECTORY = PROJECT_ROOT / "data"
VECTOR_STORE_DIRECTORY = DATA_OUTPUT_DIRECTORY / "vector_store"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536 # For text-embedding-3-small

# --- STAGE 1: LOADING ---
def load_single_document(md_path: Path) -> Dict[str, Any]:
    if not md_path.exists():
        logger.error(f"The input file '{md_path}' was not found. Aborting.")
        sys.exit(1)
    logger.info(f"Loading document from '{md_path}'...")
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return {"doc_id": md_path.stem, "content": content}

# --- STAGE 2: CHUNKING ---
def chunk_document(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_id, content = document['doc_id'], document['content']
    if not content or not content.strip():
        logger.error(f"Document '{doc_id}' is empty. Aborting.")
        sys.exit(1)

    logger.info(f"Performing structure-aware chunking on document: {doc_id}")
    elements = partition_md(text=content)
    chunked_elements = chunk_by_title(
        elements, max_characters=1500, new_after_n_chars=1200, combine_text_under_n_chars=500
    )

    final_chunks = [
        {"doc_id": doc_id, "chunk_id": str(uuid.uuid4()), "chunk_index": i, "text": chunk.text}
        for i, chunk in enumerate(chunked_elements)
    ]
    logger.info(f"Successfully split document into {len(final_chunks)} semantic chunks.")
    return final_chunks

# --- STAGE 3: EMBEDDING & UNIFICATION ---
class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.client = OpenAI()
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.MAX_TOKENS_PER_INPUT = 8190

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]

    def generate_and_unify_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        REQUEST_BATCH_SIZE = 100
        chunks_with_embeddings = []
        
        progress_bar = tqdm(total=len(chunks), desc=f"Generating Embeddings (Batch Size: {REQUEST_BATCH_SIZE})")

        for i in range(0, len(chunks), REQUEST_BATCH_SIZE):
            batch_of_chunks = chunks[i:i + REQUEST_BATCH_SIZE]
            texts_for_api = [chunk['text'] for chunk in batch_of_chunks]
            
            try:
                # Get embeddings for the current batch
                embeddings = self._get_embeddings_with_backoff(texts_for_api)
                
                # --- THE CRITICAL UNIFICATION STEP ---
                # Immediately add the embedding to its corresponding chunk dictionary
                for j, chunk in enumerate(batch_of_chunks):
                    chunk['embedding'] = embeddings[j]
                    chunks_with_embeddings.append(chunk)

            except Exception as e:
                logger.error(f"FATAL API Error on batch starting at index {i}: {e}. Discarding batch and aborting.")
                sys.exit(1)
            
            progress_bar.update(len(batch_of_chunks))
        
        progress_bar.close()
        
        # --- POST-GENERATION VALIDATION ---
        if len(chunks_with_embeddings) != len(chunks):
            logger.error("Critical Error: The number of chunks with embeddings does not match the original number of chunks. Aborting.")
            sys.exit(1)
        
        for chunk in chunks_with_embeddings:
            if 'embedding' not in chunk or len(chunk['embedding']) != EMBEDDING_DIMENSION:
                logger.error(f"Validation failed: Chunk {chunk['chunk_id']} is missing a valid embedding. Aborting.")
                sys.exit(1)
        
        logger.info("Validation successful: All chunks have been unified with valid embeddings.")
        return chunks_with_embeddings

# --- STAGE 4: INDEX CONSTRUCTION ---
class IndexBuilder:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")

    def _tokenizer(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', '', text).lower()
        return [self.stemmer.stem(token) for token in text.split()]

    def build_bm25_index(self, chunks: List[Dict[str, Any]]) -> BM25Okapi:
        logger.info("Building BM25 index...")
        tokenized_corpus = [self._tokenizer(chunk['text']) for chunk in tqdm(chunks, desc="Tokenizing for BM25")]
        return BM25Okapi(tokenized_corpus)

    def build_faiss_index(self, chunks: List[Dict[str, Any]]) -> faiss.Index:
        logger.info(f"Building FAISS index (IndexFlatL2) with dimension {EMBEDDING_DIMENSION}...")
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        index.add(embeddings)
        return index

def main():
    logger.info("--- Starting Robust Knowledge Base Build (V2.0) ---")
    start_time = time.time()
    
    # Stages 1 & 2: Load and Chunk
    document = load_single_document(PROJECT_ROOT / INPUT_MD_FILE)
    chunks = chunk_document(document)
    
    # Stage 3: Generate Embeddings and Unify Data
    embedder = EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME)
    chunks_with_embeddings = embedder.generate_and_unify_embeddings(chunks)
    
    # Stage 4: Build Indexes from the single, unified data source
    index_builder = IndexBuilder()
    bm25_index = index_builder.build_bm25_index(chunks_with_embeddings)
    faiss_index = index_builder.build_faiss_index(chunks_with_embeddings)
    
    # --- STAGE 5: FINAL VALIDATION AND SAVE ---
    logger.info("Performing final validation before saving artifacts...")
    if not (len(chunks_with_embeddings) == faiss_index.ntotal == bm25_index.corpus_size):
        logger.error(
            "FATAL VALIDATION ERROR: Final artifact counts do not match. "
            f"Chunks: {len(chunks_with_embeddings)}, FAISS: {faiss_index.ntotal}, BM25: {bm25_index.corpus_size}. "
            "Aborting to prevent saving corrupt data."
        )
        sys.exit(1)
    
    logger.info("Final validation successful. All artifacts are synchronized.")
    
    logger.info("--- Saving All Final Artifacts ---")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    VECTOR_STORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    # 1. FAISS Index
    faiss.write_index(faiss_index, str(VECTOR_STORE_DIRECTORY / f"faiss_index_{timestamp}.bin"))
    
    # 2. Metadata (now containing the embeddings)
    with open(VECTOR_STORE_DIRECTORY / f"chunks_metadata_{timestamp}.pkl", 'wb') as f:
        pickle.dump(chunks_with_embeddings, f)
        
    # 3. BM25 Index
    with open(VECTOR_STORE_DIRECTORY / f"bm25_index_{timestamp}.pkl", 'wb') as f:
        pickle.dump(bm25_index, f)
    
    logger.info(f"All artifacts saved with timestamp {timestamp} to '{VECTOR_STORE_DIRECTORY}'")
    logger.info(f"--- Build Finished. Total time: {time.time() - start_time:.2f} seconds. ---")

if __name__ == "__main__":
    main()