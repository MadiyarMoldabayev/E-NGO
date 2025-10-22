# src/rag_pipeline.py

import time
from typing import Dict, Any, List

import numpy as np
from openai import OpenAI

# --- ADAPTATION: Import our new, simplified components ---
from src.config import config
from src.retriever import HybridRetriever

import logging
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Orchestrates the entire Retrieval-Augmented Generation process.
    """
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        self.config = config
        self.retriever = HybridRetriever()
        
        # Initialize the OpenAI client
        if not self.config.llm.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
        self.llm_client = OpenAI(api_key=self.config.llm.api_key)
        
        logger.info("RAG Pipeline initialized successfully.")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Main method to orchestrate the RAG pipeline for a given question.
        """
        start_time = time.time()
        logger.info(f"--- [START QUERY] --- Raw Query: '{question}'")

        # 1. Generate an embedding for the user's query
        query_embedding = self._get_embedding(question)
        if not query_embedding:
            return self._build_error_response("Could not generate query embedding.", start_time)

        # 2. Retrieve candidate chunks using our Hybrid Retriever
        faiss_chunks, bm25_chunks = self.retriever.search(
            query_text=question, 
            query_embedding=query_embedding, 
            top_k=20 # Retrieve a healthy number of candidates for reranking
        )

        # 3. Fuse the results using Reciprocal Rank Fusion (RRF)
        fused_chunks = self._reciprocal_rank_fusion(faiss_chunks, bm25_chunks)

        # 4. Rerank the fused candidates using embedding similarity
        reranked_chunks = self._rerank_with_embeddings(fused_chunks, np.array(query_embedding))
        
        # 5. Assemble the final context to be sent to the LLM
        final_context_chunks = reranked_chunks[:5] # Use the top 5 reranked chunks
        context_str = self._assemble_context(final_context_chunks)

        # 6. Generate the final answer using the LLM
        final_answer = self._generate_final_answer(question, context_str)
        
        end_time = time.time()
        logger.info(f"--- [END QUERY] --- Total time: {end_time - start_time:.2f}s")
        
        return self._build_final_response(final_answer, final_context_chunks, start_time)

    def _get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for a given text using the OpenAI API."""
        try:
            response = self.llm_client.embeddings.create(
                input=[text],
                model=self.config.llm.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            return []

    def _normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
        """Normalizes scores in a list of results to a 0-1 range."""
        scores = [r.get(score_key, 0.0) for r in results]
        if not scores: return []
        max_score, min_score = max(scores), min(scores)
        if max_score == min_score:
            return [{**r, 'normalized_score': 1.0} for r in results]
        
        for r in results:
            r['normalized_score'] = (r.get(score_key, 0.0) - min_score) / (max_score - min_score + 1e-9)
        return results

    def _reciprocal_rank_fusion(self, faiss_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Performs Reciprocal Rank Fusion on normalized FAISS and BM25 results.
        Returns a single, fused list of chunk metadata.
        """
        # Normalize scores before fusion
        norm_faiss = self._normalize_scores(faiss_results, 'semantic_score')
        norm_bm25 = self._normalize_scores(bm25_results, 'bm25_score')
        
        ranked_lists = [norm_faiss, norm_bm25]
        scores = {}
        for results in ranked_lists:
            sorted_results = sorted(results, key=lambda x: x.get('normalized_score', 0.0), reverse=True)
            for rank, result in enumerate(sorted_results, start=1):
                chunk_id = result.get('chunk_id')
                if chunk_id:
                    scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank)
        
        # Create a map of all unique chunks from both result sets
        all_chunks_map = {c['chunk_id']: c for c in faiss_results + bm25_results}
        
        # Create the final fused list with full metadata, sorted by RRF score
        fused_list = []
        for chunk_id in sorted(scores, key=scores.get, reverse=True):
            if chunk_id in all_chunks_map:
                chunk = all_chunks_map[chunk_id]
                chunk['rrf_score'] = scores[chunk_id]
                fused_list.append(chunk)

        return fused_list

    def _rerank_with_embeddings(self, chunks: List[Dict[str, Any]], query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        """
        Reranks chunks based on cosine similarity with the query embedding.
        """
        if not chunks or query_embedding.size == 0:
            return []

        # Get embeddings for all candidate chunks. These should already be in the metadata.
        chunk_embeddings = np.array([chunk.get('embedding', []) for chunk in chunks], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity calculation
        query_emb_norm = query_embedding / np.linalg.norm(query_embedding)
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1)[:, np.newaxis]
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarities = np.dot(chunk_embeddings_norm, query_emb_norm.T).flatten()
        
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(similarities[i])
            
        # Sort chunks by the new rerank_score in descending order
        chunks.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        return chunks

    def _assemble_context(self, chunks: List[Dict]) -> str:
        """Assembles the text from the top chunks into a single string for the LLM."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"--- Document Chunk {i+1} (Source: {chunk.get('doc_id')}) ---\n{chunk.get('text')}")
        return "\n\n".join(context_parts)

    def _generate_final_answer(self, query: str, context: str) -> str:
        """Generates the final answer from the LLM using the provided context."""
        if not context:
            # This is a fallback. The system prompt should handle this, but it's good practice.
            return "The provided document does not contain enough information to answer this question."

        user_prompt = f"""
<user_question>
{query}
</user_question>

<document_context>
{context}
</document_context>
"""
        messages = [
            {"role": "system", "content": self.config.llm.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.completion_model,
                messages=messages,
                temperature=self.config.llm.temperature
            )
            return response.choices[0].message.content or "Failed to generate a valid response."
        except Exception as e:
            logger.error(f"Error during final answer generation: {e}", exc_info=True)
            return "An error occurred while generating the answer."

    def _build_final_response(self, answer: str, sources: List[Dict], start_time: float) -> Dict[str, Any]:
        """Builds the final dictionary response for the UI."""
        return {
            "answer": answer,
            "sources": [{"doc_id": s.get('doc_id'), "chunk_index": s.get('chunk_index'), "score": s.get('rerank_score')} for s in sources],
            "request_duration": time.time() - start_time
        }

    def _build_error_response(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Builds a standardized error response."""
        return {
            "answer": f"An error occurred: {error_message}",
            "sources": [],
            "request_duration": time.time() - start_time
        }