import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, Field
import uvicorn

# -------------------- NLTK Setup --------------------
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Data Classes --------------------
@dataclass
class ChunkMetrics:
    semantic_coherence: float
    information_density: float
    contextual_completeness: float
    size_penalty: float
    overall_score: float

@dataclass
class DocumentChunk:
    text: str
    page_number: int
    chunk_id: str
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    metrics: Optional[ChunkMetrics] = None

# -------------------- Chunker --------------------
class DynamicChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.min_chunk_size = 100
        self.max_chunk_size = 800
        self.overlap_ratio = 0.1
        self.coherence_weight = 0.3
        self.density_weight = 0.25
        self.completeness_weight = 0.25
        self.size_weight = 0.2

    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        try:
            reader = PdfReader(pdf_path)
            pages_text = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages_text.append((text, page_num + 1))
            return pages_text
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise

    def calculate_semantic_coherence(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        embeddings = self.embedding_model.encode(sentences)
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        return np.mean(similarities)

    def calculate_information_density(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        unique_words = set(words)
        sentences = sent_tokenize(text)
        lexical_diversity = len(unique_words) / len(words)
        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
        density_score = (lexical_diversity * 0.6) + (min(avg_sentence_length / 20, 1.0) * 0.4)
        return min(density_score, 1.0)

    def calculate_contextual_completeness(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        complete_sentences = sum(1 for sent in sentences if sent.strip().endswith(('.', '!', '?')))
        sentence_completeness = complete_sentences / len(sentences)
        paragraphs = text.split('\n\n')
        paragraph_score = min(len(paragraphs) / 3, 1.0)
        return (sentence_completeness * 0.7) + (paragraph_score * 0.3)

    def calculate_size_penalty(self, chunk_size: int) -> float:
        optimal_size = 400
        if chunk_size < self.min_chunk_size:
            return 0.5
        elif chunk_size > self.max_chunk_size:
            return 0.3
        size_score = np.exp(-((chunk_size - optimal_size) ** 2) / (2 * (optimal_size * 0.5) ** 2))
        return size_score

    def evaluate_chunk_quality(self, text: str) -> ChunkMetrics:
        sentences = sent_tokenize(text)
        chunk_size = len(self.tokenizer.encode(text))
        coherence = self.calculate_semantic_coherence(sentences)
        density = self.calculate_information_density(text)
        completeness = self.calculate_contextual_completeness(text)
        size_penalty = self.calculate_size_penalty(chunk_size)
        overall_score = (
            coherence * self.coherence_weight +
            density * self.density_weight +
            completeness * self.completeness_weight +
            size_penalty * self.size_weight
        )
        return ChunkMetrics(
            semantic_coherence=coherence,
            information_density=density,
            contextual_completeness=completeness,
            size_penalty=size_penalty,
            overall_score=overall_score
        )

    def create_dynamic_chunks(self, text: str, page_number: int) -> List[DocumentChunk]:
        sentences = sent_tokenize(text)
        if not sentences:
            return []
        chunks = []
        current_chunk_sentences = []
        current_chunk_start = 0
        best_chunk_size = self.min_chunk_size
        best_score = 0.0
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            current_text = ' '.join(current_chunk_sentences)
            current_size = len(self.tokenizer.encode(current_text))
            if current_size >= self.min_chunk_size:
                metrics = self.evaluate_chunk_quality(current_text)
                if metrics.overall_score > best_score:
                    best_score = metrics.overall_score
                    best_chunk_size = current_size
                if (current_size >= best_chunk_size * 1.2 or 
                    i == len(sentences) - 1 or
                    current_size >= self.max_chunk_size):
                    chunk_id = hashlib.md5(current_text.encode()).hexdigest()[:8]
                    chunk = DocumentChunk(
                        text=current_text,
                        page_number=page_number,
                        chunk_id=chunk_id,
                        start_char=current_chunk_start,
                        end_char=current_chunk_start + len(current_text),
                        metrics=metrics
                    )
                    chunks.append(chunk)
                    overlap_size = int(len(current_chunk_sentences) * self.overlap_ratio)
                    current_chunk_sentences = current_chunk_sentences[-overlap_size:] if overlap_size > 0 else []
                    current_chunk_start += len(current_text) - (len(' '.join(current_chunk_sentences)) if current_chunk_sentences else 0)
        return chunks

# -------------------- Vector Database --------------------
class VectorDatabase:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension

    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        faiss.write_index(self.index, "CLAUDE_faiss.index")
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector database")

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or not self.chunks:
            return []
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

# -------------------- LLM Client --------------------
class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:instruct"):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt: str, context: str) -> str:
        try:
            
            # -------------------- LLM Prompt Started --------------------
            full_prompt = f"""You are a helpful assistant. Use only the information provided in the context below to answer the user's question. If the context does not contain enough information, clearly state your answer.

Instructions:
- Base your answer strictly on the context below.
- Be concise, accurate, and clear.

Context:
{context}

Question: {prompt}
Answer: """
            # -------------------- LLM Prompt Ended --------------------
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["response"].strip()
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return "Error generating response from LLM"
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "Error generating response"

# -------------------- RAG System --------------------
class RAGSystem:
    def __init__(self, pdf_path: str):
        self.chunker = DynamicChunker()
        self.vector_db = VectorDatabase(self.chunker.embedding_model)
        self.llm_client = LLMClient()
        self.pdf_path = pdf_path
        self.is_initialized = False

    async def initialize(self):
        if self.is_initialized:
            return
        logger.info("Initializing RAG system...")
        pages_text = self.chunker.extract_text_from_pdf(self.pdf_path)
        all_chunks = []
        for text, page_num in pages_text:
            chunks = self.chunker.create_dynamic_chunks(text, page_num)
            all_chunks.extend(chunks)
        self.vector_db.add_chunks(all_chunks)
        self.is_initialized = True
        logger.info(f"RAG system initialized with {len(all_chunks)} chunks")

    def calculate_confidence_score(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], generated_answer: str) -> float:
        if not retrieved_chunks:
            return 0.0
        avg_retrieval_score = np.mean([score for _, score in retrieved_chunks])
        query_embedding = self.chunker.embedding_model.encode([query])
        answer_embedding = self.chunker.embedding_model.encode([generated_answer])
        answer_similarity = cosine_similarity(query_embedding, answer_embedding)[0][0]
        confidence = (avg_retrieval_score * 0.6) + (answer_similarity * 0.4)
        return min(confidence, 1.0)

    async def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        if not self.is_initialized:
            await self.initialize()
        if not question.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        retrieved_chunks = self.vector_db.search(question, k)
        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the document.",
                "source_page": None,
                "confidence_score": 0.0,
                "chunk_size": 0
            }
        context_parts = []
        total_chunk_size = 0
        chunk_data = []
        for chunk, score in retrieved_chunks:
            print(chunk.page_number, "ppage")
            context_parts.append(f"[Page {chunk.page_number}] {chunk.text}")
            total_chunk_size += len(chunk.text)
            chunk_data.append({
                "page_number": chunk.page_number,
                "score": score
            })
        context = "\n\n".join(context_parts)
        answer = self.llm_client.generate_response(question, context)
        confidence = self.calculate_confidence_score(question, retrieved_chunks, answer)
        source_page = retrieved_chunks[0][0].page_number
        avg_chunk_size = total_chunk_size // len(retrieved_chunks)
        return {
            "answer": answer,
            "source_page": source_page,
            "confidence_score": round(confidence, 3),
            "chunk_size": avg_chunk_size,
            # "chunks": chunk_data
        }

# -------------------- FastAPI App --------------------
app = FastAPI(title="RAG System API", description="Retrieval-Augmented Generation System")
rag_system = None

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    source_page: Optional[int]
    confidence_score: float
    chunk_size: int
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="List of retrieved chunks with metadata")      

# -------------------- Initialize on Startup --------------------
@app.on_event("startup")
async def startup_event():
    global rag_system
    pdf_path = "medicare-and-you.pdf"
    if not os.path.exists(pdf_path):
        logger.info("Downloading Medicare PDF...")
        url = "https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf"
        response = requests.get(url)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        logger.info("PDF downloaded successfully")
    rag_system = RAGSystem(pdf_path)
    await rag_system.initialize()

# -------------------- API Endpoints --------------------
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    try:
        result = await rag_system.query(request.question, request.k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "endpoints": {
            "POST /": "Submit a question to the RAG system", 
            "GET /": "This is RAG System API",
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)