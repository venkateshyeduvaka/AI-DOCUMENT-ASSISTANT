# RAG Q&A System - OpenAI + FastAPI + React

A modern RAG (Retrieval-Augmented Generation) Q&A system that allows users to upload PDF documents and chat with their content using OpenAI's GPT models.

## Features

- **3-Step Process**: API Key → Document Upload → Chat Interface
- **Modern UI**: React with Tailwind CSS
- **FastAPI Backend**: High-performance Python backend
- **OpenAI Integration**: Uses OpenAI's GPT and embedding models
- **Chat History**: Maintains conversation context
- **Multiple PDF Support**: Upload and process multiple PDF files
- **Real-time Chat**: Interactive chat interface with loading states

## Project Structure

```
rag-qa-system/
├── backend/
│   ├── main.py              # FastAPI backend
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main React component
    │   └── index.css        # Tailwind CSS imports
    ├── public/
    │   └── index.html       # HTML template
    ├── package.json         # Node.js dependencies
    └── tailwind.config.js   # Tailwind configuration
```

## Setup Instructions

### Backend Setup

1. **Create a virtual environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Set up Tailwind CSS:**
   ```bash
   npx tailwindcss init -p
   ```

3. **Create the CSS file (src/index.css):**
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```


4. **Start the React development server:**
   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /upload-documents
Upload PDF documents and create vector store.

**Request:**
- `files`: Multiple PDF files
- `session_id`: Unique session identifier
- `openai_api_key`: OpenAI API key

**Response:**
```json
{
  "message": "Successfully processed X documents",
  "session_id": "session_123",
  "documents_count": 5,
  "chunks_count": 150
}
```

### POST /chat
Send a message and get AI response.

**Request:**
```json
{
  "message": "What is the main topic of the document?",
  "session_id": "session_123",
  "openai_api_key": "sk-..."
}
```

**Response:**
```json
{
  "answer": "The main topic of the document is...",
  "session_id": "session_123"
}
```

### GET /sessions/{session_id}
Get session information.

### GET /sessions/{session_id}/history
Get chat history for a session.

### DELETE /sessions/{session_id}
Clear session data.

## Usage

1. **Enter OpenAI API Key**: Start by entering your OpenAI API key
2. **Upload Documents**: Select and upload one or more PDF files
3. **Chat**: Ask questions about your documents and get AI-powered responses

## Key Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Live status updates during document processing
- **Error Handling**: Comprehensive error handling and user feedback
- **Session Management**: Each session maintains its own document store and chat history
- **Modern UI/UX**: Clean, modern interface with smooth animations

## Environment Variables

The system uses OpenAI API keys provided through the frontend interface. No server-side environment variables are required for API keys.

## Dependencies

### Backend
- FastAPI: Web framework
- LangChain: LLM orchestration
- OpenAI: Language models and embeddings
- ChromaDB: Vector database
- PyPDF: PDF processing

### Frontend
- React: UI framework
- Tailwind CSS: Styling
- Lucide React: Icons

## Development Notes

- The backend runs on port 8000
- The frontend runs on port 3000
- CORS is enabled for local development
- Sessions are stored in memory (consider Redis for production)
- Vector stores are temporary (consider persistent storage for production)