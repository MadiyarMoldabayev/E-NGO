# E-NGO RAG Application (FastAPI + Netlify)

A FastAPI-based RAG (Retrieval-Augmented Generation) application for querying NGO standards and regulations using AI, deployed on Netlify with serverless functions.

## Features

- **Document Q&A**: Ask questions about NGO standards and get AI-powered answers
- **Hybrid Retrieval**: Uses both semantic (FAISS) and keyword (BM25) search for better results
- **Source Attribution**: Shows sources for each answer with confidence scores
- **Multi-language Support**: Supports questions in multiple languages
- **Real-time Chat Interface**: Interactive chat interface with modern UI
- **Serverless Deployment**: Deployed on Netlify using serverless functions

## Technology Stack

- **Backend**: FastAPI with Python serverless functions
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **AI/ML**: OpenAI GPT models, FAISS vector search, BM25 keyword search
- **Deployment**: Netlify serverless functions
- **Document Processing**: PyPDF2 for PDF text extraction

## Project Structure

```
E-NGO-FastAPI/
├── main.py                    # FastAPI application (for local development)
├── index.html                 # Frontend HTML interface
├── netlify.toml              # Netlify configuration
├── requirements.txt           # Python dependencies
├── netlify/
│   └── functions/
│       ├── ask.py             # Serverless function for Q&A
│       └── requirements.txt  # Function-specific dependencies
├── src/
│   ├── config.py             # Configuration management
│   ├── rag_pipeline.py       # Main RAG pipeline
│   └── retriever.py          # Document retrieval logic
└── data/
    └── vector_store/         # Pre-built vector indexes (not in repo)
```

## Local Development

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd E-NGO-FastAPI
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

4. **Run the application locally**:
   ```bash
   python main.py
   ```

5. **Access the app**:
   Open your browser and go to `http://localhost:8000`

## Netlify Deployment

### Automatic Deployment

1. **Connect to Netlify**:
   - Go to [netlify.com](https://netlify.com)
   - Sign in and click "New site from Git"
   - Connect your GitHub account
   - Select this repository

2. **Configure build settings**:
   - Build command: `echo 'No build needed'`
   - Publish directory: `.`
   - Add environment variable: `OPENAI_API_KEY`

3. **Deploy**:
   - Click "Deploy site"
   - Your app will be available at `https://your-site-name.netlify.app`

### Manual Deployment

1. **Install Netlify CLI**:
   ```bash
   npm install -g netlify-cli
   ```

2. **Login to Netlify**:
   ```bash
   netlify login
   ```

3. **Deploy**:
   ```bash
   netlify deploy --prod
   ```

## Usage

1. **Start the app** and wait for the knowledge base to load
2. **Ask questions** about NGO standards, regulations, or procedures
3. **View sources** by checking the source attribution for each answer
4. **Get detailed information** about specific topics related to NGO operations

## Example Questions

- "What are the requirements for NGO registration?"
- "How should I handle grant accounting?"
- "What are the compliance requirements for NGOs?"
- "How to manage foreign donor funds?"
- "What documents are needed for NGO setup?"

## API Endpoints

- `GET /` - Serves the main HTML interface
- `POST /.netlify/functions/ask` - Processes Q&A requests
- `GET /api/health` - Health check endpoint (local only)

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with `python main.py`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Notes

- The vector store files are not included in the repository due to size limitations
- For production deployment, you may need to rebuild the vector indexes
- The serverless function has a timeout limit on Netlify (10 seconds for free tier)