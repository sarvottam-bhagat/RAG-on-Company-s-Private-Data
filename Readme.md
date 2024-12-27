# RAG-Based Project QA System for Private Enterprise Data 🔍

## 📚 Introduction

Retrieval-Augmented Generation (RAG) is revolutionizing how companies leverage their institutional knowledge. This project implements a RAG system that empowers organizations to unlock insights from their private internal data while maintaining data privacy and security.

### What is RAG?

RAG combines two powerful components:
1. **Retrieval**: Efficiently extracts relevant documents from your company's internal knowledge base
2. **Generation**: Uses advanced language models to create precise, contextual responses

### Business Value
- **Knowledge Discovery**: Quickly find relevant past projects and experiences
- **Informed Decision Making**: Learn from historical successes and failures
- **Productivity Enhancement**: Reduce time spent searching through documentation
- **Institutional Memory**: Preserve and easily access organizational knowledge
- **Privacy-First**: Process sensitive data locally within your network

## 🌟 Features

- **Smart Document Processing**: Efficiently handles internal project documentation
- **Vector Search**: Uses FAISS for lightning-fast similarity search
- **Context-Aware Responses**: Maintains conversation history for better understanding
- **Structured Output**: Organized response format including:
  - Project details
  - Algorithms attempted
  - Success metrics
  - Lessons learned
- **Privacy-Focused**: Runs LLM operations within your infrastructure
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI

## 🛠️ Technologies Used

- Streamlit - Web Interface
- LangChain - LLM Framework
- FAISS - Vector Database
- Google Generative AI - Embeddings
- Groq - LLM Provider

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key
- Google API Key

## 🚀 Installation & Running

1. **Clone the repository**
   ```bash
   git clone https://github.com/sarvottam-bhagat/RAG-on-Company-s-Private-Data.git
   cd RAG-on-Company-s-Private-Data
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # Linux/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**  
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY="your_groq_api_key"
   GOOGLE_API_KEY="your_google_api_key"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default web browser at `http://localhost:8501`

## 💻 Using the Application

1. Upload your text file using the file uploader in the UI
2. Wait for the knowledge base to be created
3. Start asking questions in the chat interface
4. View color-coded responses showing:
   - Questions (red)
   - Answers (blue)
   - Source documents (green)
