# app.py

import streamlit as st
import logging.config

# --- ADAPTATION: Import our new, self-contained RAG pipeline ---
from src.rag_pipeline import RAGPipeline
from src.config import config

# --- Basic Logging Setup ---
# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
# Set the page title, icon, and layout for a better user experience.
st.set_page_config(
    page_title=config.app.app_title,
    page_icon=config.app.app_icon,
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_rag_pipeline():
    """
    Loads the RAGPipeline using Streamlit's caching.
    This ensures the model and indexes are loaded only once per session,
    making the app much faster.
    """
    with st.spinner("Loading knowledge base... This may take a moment."):
        pipeline = RAGPipeline()
    return pipeline

# Load the pipeline and store it in a variable
rag_pipeline = load_rag_pipeline()

# --- Session State Initialization ---
# `st.session_state` is Streamlit's way of preserving variables between user interactions.
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message to start the chat
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I have read the document. How can I help you?"}
    )

# --- UI Rendering ---

# Display the main title of the application
st.title(config.app.app_title)

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message is from the assistant and has sources, display them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(f"Source: {source['doc_id']} | Chunk Index: {source['chunk_index']} | Score: {source['score']:.2f}")

# --- User Input and Chat Logic ---
# `st.chat_input` creates a text input box at the bottom of the page.
if prompt := st.chat_input(config.app.app_placeholder):
    # 1. Add the user's message to the session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get the bot's response from our RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the main method of our pipeline
            response = rag_pipeline.answer_question(prompt)
            answer = response.get("answer", "Sorry, I encountered an error.")
            sources = response.get("sources", [])
            
            st.markdown(answer)
            
            # Display the sources in an expander for transparency
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.info(f"Source: {source['doc_id']} | Chunk Index: {source['chunk_index']} | Score: {source['score']:.2f}")

    # 3. Add the complete bot response (with sources) to the session state
    st.session_state.messages.append(
        {
            "role": "assistant", 
            "content": answer, 
            "sources": sources
        }
    )