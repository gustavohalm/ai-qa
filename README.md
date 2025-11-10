# iPhone 17 AI API

A Flask REST API that answers questions about the iPhone 17 using Langchain and Qdrant vector database.

## Features

- 🤖 AI-powered question answering using OpenAI and Langchain
- 🔍 Vector search with Qdrant for relevant context retrieval
- 🚀 RESTful API with Flask
- 📱 Specialized knowledge base about iPhone 17
- 🕷️ Built-in crawler to ingest Apple Support docs (seeded URLs)
- 🌐 Optional web search fallback (DuckDuckGo) when RAG context is weak
- 📄 Local PDF ingestion for Product Environmental Reports under `data/`

## Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant (can run locally with Docker or use cloud service)

## Installation

1. Clone the repository:
```bash
cd ai-test
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key and other configuration:
```
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
QDRANT_URL=http://localhost:6333
ENABLE_WEB_FALLBACK=true
# Crawl controls
CRAWL_MAX_DEPTH=1
CRAWL_MAX_PAGES=30
CRAWL_REQUEST_TIMEOUT_SEC=15
CRAWL_REQUEST_DELAY_SEC=0.5
# Seed URLs (override if needed)
SEED_URL_IPHONE_17=https://support.apple.com/en-us/125089
SEED_URL_IPHONE_USER_GUIDE=https://support.apple.com/guide/iphone/welcome/26/ios/26
SEED_URL_IPHONE_17_PRO_MAX=https://support.apple.com/en-us/125091
SEED_URL_IPHONE_PRO_DOCS=https://support.apple.com/en-us/docs/iphone/301245
PORT=5000
DEBUG=True
```

## Running Qdrant

### Option 1: Using Docker (Recommended)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Option 2: Using Docker Compose
Create a `docker-compose.yml`:
```yaml
version: '3'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then run:
```bash
docker-compose up -d
```

## Running the API

1. Make sure Qdrant is running (see above)

2. Start the Flask application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### Ask a Question (POST)
```bash
POST /api/iphone17
Content-Type: application/json

{
  "question": "What are the camera specifications of iPhone 17?"
}
```

Optional request field:
```json
{
  "question": "What are the camera specifications of iPhone 17?",
  "web_fallback": true
}
```
- **web_fallback**: Overrides the `ENABLE_WEB_FALLBACK` environment setting for this request.

Response:
```json
{
  "question": "What are the camera specifications of iPhone 17?",
  "answer": "The iPhone 17 comes with a 48MP main camera...",
  "status": "success"
}
```

### Get General Info (GET)
```bash
GET /api/iphone17
```

Response:
```json
{
  "info": "The iPhone 17 is the latest flagship...",
  "status": "success"
}
```

## Example Usage

### Using curl:
```bash
curl -X POST http://localhost:5000/api/iphone17 \
  -H "Content-Type: application/json" \
  -d '{"question": "What colors is the iPhone 17 available in?"}'
```

### Using Python requests:
```python
import requests

response = requests.post(
    'http://localhost:5000/api/iphone17',
    json={'question': 'How long does the battery last?'}
)

print(response.json())
```

## Project Structure

```
ai-test/
├── app.py              # Flask application with API endpoints
├── ai_service.py       # AI service class with Langchain and Qdrant
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .env.example       # Example environment variables
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Architecture

- **Flask**: Web framework for the REST API
- **Langchain**: Framework for building LLM applications
- **OpenAI**: Language model for generating answers
- **Qdrant**: Vector database for storing and searching iPhone 17 knowledge
- **OpenAI Embeddings**: For converting text to vector embeddings
- **Crawler**: Fetches and ingests content from Apple Support seed URLs
- **DuckDuckGo Search**: Fallback internet search when RAG doesn't have enough context

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8
```

## Extending the Knowledge Base

The service automatically crawls and ingests content from Apple Support seed URLs on first run (or when the Qdrant collection is empty). You can also add specific facts programmatically:

```python
from ai_service import AIService

service = AIService()
service.add_knowledge("The iPhone 17 supports MagSafe charging up to 25W.")
```

## Data Sources
- iPhone 17 - Tech Specs: `https://support.apple.com/en-us/125089`
- iPhone User Guide: `https://support.apple.com/guide/iphone/welcome/26/ios/26`
- iPhone 17 Pro Max - Tech Specs: `https://support.apple.com/en-us/125091`
- iPhone Pro Documentation Hub: `https://support.apple.com/en-us/docs/iphone/301245`
- Product Environmental Report(s): any `*.pdf` placed under `data/`

## License

MIT

## Notes

- Make sure to keep your `.env` file secure and never commit it to version control
- On first run with an empty collection, the service will crawl the Apple Support seed URLs (depth and page limits are configurable)
- For production use, consider adding authentication, rate limiting, and error handling

