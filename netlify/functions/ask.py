import json
import os
import logging
from src.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global RAG pipeline instance
rag_pipeline = None

def initialize_rag():
    """Initialize RAG pipeline if not already done"""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            logging.info("Initializing RAG Pipeline...")
            rag_pipeline = RAGPipeline()
            logging.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG Pipeline: {e}")
            raise

def handler(event, context):
    """Netlify serverless function handler"""
    try:
        # Initialize RAG pipeline
        initialize_rag()
        
        # Parse the request
        if event['httpMethod'] == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                },
                'body': ''
            }
        
        if event['httpMethod'] != 'POST':
            return {
                'statusCode': 405,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Method not allowed'})
            }
        
        # Parse request body
        try:
            body = json.loads(event['body'])
            question = body.get('question', '')
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Invalid JSON'})
            }
        
        if not question:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Question is required'})
            }
        
        # Process the question
        logging.info(f"Processing question: {question}")
        response = rag_pipeline.answer_question(question)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'answer': response.get('answer', 'Sorry, I encountered an error.'),
                'sources': response.get('sources', [])
            })
        }
        
    except Exception as e:
        logging.error(f"Error in handler: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }
