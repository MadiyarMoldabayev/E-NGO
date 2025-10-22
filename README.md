# E-NGO RAG Application

A Streamlit-based RAG (Retrieval-Augmented Generation) application for querying NGO standards and regulations using AI.

## Features

- **Document Q&A**: Ask questions about NGO standards and get AI-powered answers
- **Hybrid Retrieval**: Uses both semantic (FAISS) and keyword (BM25) search for better results
- **Source Attribution**: Shows sources for each answer with confidence scores
- **Multi-language Support**: Supports questions in multiple languages
- **Real-time Chat Interface**: Interactive chat interface powered by Streamlit

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT models, FAISS vector search, BM25 keyword search
- **Backend**: Python with Pydantic for configuration
- **Document Processing**: PyPDF2 for PDF text extraction

## Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd E-NGO
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the app**:
   Open your browser and go to `http://localhost:8501`

## Project Structure

```
E-NGO/
├── app.py                 # Main Streamlit application
├── build_indexes.py       # Script to build vector indexes
├── extract_text_from_pdf.py # PDF text extraction utility
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py         # Configuration management
│   ├── rag_pipeline.py   # Main RAG pipeline
│   └── retriever.py      # Document retrieval logic
└── data/
    └── vector_store/     # Pre-built vector indexes
        ├── faiss_index_*.bin
        ├── chunks_metadata_*.pkl
        └── bm25_index_*.pkl
```

## Usage

1. **Start the app** and wait for the knowledge base to load
2. **Ask questions** about NGO standards, regulations, or procedures
3. **View sources** by expanding the "View Sources" section for each answer
4. **Get detailed information** about specific topics related to NGO operations

## Example Questions

- "What are the requirements for NGO registration?"
- "How should I handle grant accounting?"
- "What are the compliance requirements for NGOs?"
- "How to manage foreign donor funds?"

## Deployment

This application can be deployed to various platforms:

- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using the included Procfile
- **Docker**: Containerized deployment
- **Netlify**: Static site deployment (with modifications)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
