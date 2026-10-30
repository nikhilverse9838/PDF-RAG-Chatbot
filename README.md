
# PDF Based RAG Chatbot

An AI-powered chatbot that answers questions from PDF documents using Retrieval Augmented Generation (RAG).

## Features
- PDF document loading
- Text chunking
- Hugging Face embeddings
- Semantic search using ChromaDB
- Gemini API for answer generation
- Conversation history
- Follow-up question support

## Tech Stack
- Python
- LangChain
- Hugging Face
- ChromaDB
- Google Gemini API

## Project Setup

### 1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL

### 2. Open project folder
cd PDF-RAG-Chatbot

### 3. Create virtual environment
uv venv --python 3.12

### 4. Activate environment (Windows)
.venv\Scripts\activate

### 5. Install dependencies
uv pip install -r requirements.txt

### 6. Configure API Key

Create a `.env` file in the project root:

GOOGLE_API_KEY=your_gemini_api_key

### 7. Add PDF
Place your PDF inside the documents folder.

### 8. Create vector database
python create_database.py

### 9. Run chatbot
python main.py

## Project Structure

PDF-RAG-Chatbot/
- documents/
- create_database.py
- main.py
- requirements.txt
- .gitignore
- README.md
