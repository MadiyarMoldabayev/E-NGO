from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our RAG pipeline
from src.rag_pipeline import RAGPipeline
from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title=config.app.app_title,
    description="AI-powered Q&A system for NGO standards and regulations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    try:
        logging.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline()
        logging.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize RAG Pipeline: {e}")
        raise

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list = []

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    try:
        logging.info(f"Processing question: {request.question}")
        response = rag_pipeline.answer_question(request.question)
        
        return QuestionResponse(
            answer=response.get("answer", "Sorry, I encountered an error."),
            sources=response.get("sources", [])
        )
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "rag_initialized": rag_pipeline is not None}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
