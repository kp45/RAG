# RAG-Based Retrieval System with Dynamic Chunking

Retrieval-Augmented Generation (RAG) system that intelligently processes PDF documents with dynamic chunk sizing and provides accurate, contextual answers to user queries.

## 🚀 Features

- **Dynamic Chunk Sizing**: Intelligent chunking algorithm that determines optimal chunk sizes based on content analysis
- **Multi-Metric Evaluation**: Comprehensive chunk quality assessment using:
- **Vector Search**: Efficient similarity search using FAISS
- **LLM Integration**: Uses Ollama for open-source language model inference
- **RESTful API**: FastAPI-based API with comprehensive error handling
- **Confidence Scoring**: Provides confidence scores for generated responses

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  Dynamic        │───▶│  Vector         │
│                 │    │  Chunker        │    │  Database       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │◀───│      LLM        │◀───│   Retrieval     │
│   Response      │    │   Generation    │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔗 Live Server Endpoint

**URL:** [https://a847-180-211-111-99.ngrok-free.app](https://a847-180-211-111-99.ngrok-free.app)

You can use this endpoint to test or connect your client-side application.

### 📦 Example (cURL)
```bash
curl -X POST https://a847-180-211-111-99.ngrok-free.app/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Are laboratory tests covered by Medicare, and what is the cost", "k": 3}'
```

## 📋 Prerequisites

- Python 3.10+
- Ollama (for LLM inference)
- 6GB+ RAM (recommended)
- Internet connection (for initial setup)

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kp45/RAG.git
cd RAG
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Setup Ollama

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull the LLM model**:
   ```bash
   ollama pull llama3:instruct
   ```

### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

## 🚀 Usage

### Starting the Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Making Queries

#### Using curl:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What are the important deadlines for Medicare enrollment?"}'
```
     
or
```bash

curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What ISN'\''T covered by Part A and Part B?", "k": 4}'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What home health services does Medicare cover??"}
)

print(response.json())
```

### Example Response

```json
{
    "answer": "Medicare covers home health services as long as you need part-time or intermittent skilled services and as long as you're \"homebound,\" which means:\n\n* You have trouble leaving your home without help (like using a cane, wheelchair, walker, or crutches; special transportation; or help from another person) because of an illness or injury.\n* Leaving your home isn't recommended because of your condition.\n* You're normally unable to leave your home because it's a major effort.\n\nYou pay nothing for covered home health services.",
    "source_page": 44,
    "confidence_score": 0.72,
    "chunk_size": 1320,
}
```

## 📚 API Reference

### POST `/query`

Submit a question to the RAG system.

**Request Body:**
```json
{
  "question": "string",
  "k": 3  // Optional: number of chunks to retrieve (1-10)
}
```

**Response:**
```json
{
  "answer": "string",
  "source_page": "integer|null",
  "confidence_score": "float",
  "chunk_size": "integer"
}
```

## 🧪 Testing
### Query Examples

```bash
# Basic query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Medicare?"}'

# Query with custom retrieval count
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I apply for Medicare?", "k": 5}'
```


## 📁 Project Structure

```
rag-system/
├── app.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # Check if model is available
   ollama list
   ```

2. **PDF Download Issues**:
   ```bash
   # Manually download the PDF
   wget https://www.medicare.gov/Pubs/pdf/10050-medicare-and-you.pdf
   ```

3. **Memory Issues**:
   ```bash
   # Reduce chunk size or use GPU acceleration
   pip install faiss-gpu torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🚀 Performance Optimization

### For Better Performance:

1. **Use GPU acceleration**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install faiss-gpu
   ```

2. **Optimize chunk parameters**:
   - Adjust `min_chunk_size` and `max_chunk_size` based on your documents

3. **Use faster embedding models**:
   ```python
   # In DynamicChunker.__init__()
   self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster
   # or
   self.embedding_model = SentenceTransformer("all-mpnet-base-v2")  # More accurate
   ``` 

## 📄 License

This project is Just for Assignment Purphose.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Ollama](https://ollama.ai/) for LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

