exports.handler = async (event, context) => {
  // Handle CORS
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      },
      body: ''
    };
  }

  // Only allow POST requests
  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ error: 'Method not allowed' })
    };
  }

  try {
    // Parse the request body
    const { question } = JSON.parse(event.body);
    
    if (!question) {
      return {
        statusCode: 400,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ error: 'Question is required' })
      };
    }

    // For now, return a demo response since we need Python for the RAG pipeline
    // In a real deployment, you'd call your Python RAG service here
    const demoResponses = [
      "Based on NGO standards, organizations must maintain proper documentation and follow compliance requirements.",
      "For grant accounting, NGOs should maintain detailed records of all transactions and ensure transparency.",
      "Compliance requirements include regular reporting, financial transparency, and adherence to local regulations.",
      "Foreign donor funds must be properly documented and reported according to international standards."
    ];

    const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];

    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        answer: randomResponse,
        sources: [
          { doc_id: 'NGO_Standards_Doc', chunk_index: 1, score: 0.95 },
          { doc_id: 'Compliance_Guide', chunk_index: 3, score: 0.87 }
        ]
      })
    };

  } catch (error) {
    console.error('Error:', error);
    return {
      statusCode: 500,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ error: 'Internal server error' })
    };
  }
};
