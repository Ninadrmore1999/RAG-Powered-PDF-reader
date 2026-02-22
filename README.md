RAG PDF Assistant 🤖📄
Python Streamlit License GitHub stars

An intelligent Retrieval-Augmented Generation (RAG) system that allows you to query PDF documents using natural language. Get instant, accurate answers without manually reading through documents.

🚀 Live Demo
Streamlit App

Note: Deploying soon - follow deployment instructions below

✨ Features
🔒 100% Private - All processing happens locally on your machine
💸 Completely Free - No API keys, no usage limits, no costs
📄 Universal PDF Support - Works with resumes, research papers, reports, contracts, manuals
🤖 Smart Hybrid Search - Combines semantic + keyword search for better accuracy
🎯 Context-Aware Answers - Answers based only on your document content
⚡ Fast Processing - Local embeddings with FAISS vector database
🎨 Beautiful UI - Modern Streamlit interface with real-time processing
🛠️ Tech Stack
Technology	Purpose
Python	Backend logic and AI processing
Streamlit	Web application framework
Sentence Transformers	Local text embeddings (all-MiniLM-L6-v2)
FAISS	Vector similarity search
PyPDF	PDF text extraction
NumPy	Numerical computations
🎯 Use Cases
📚 Research - Quickly find information in academic papers and articles
💼 Recruitment - Analyze resumes and extract key information
⚖️ Legal - Query contracts and legal documents
📊 Business - Extract insights from reports and manuals
🎓 Education - Study and analyze educational materials
📦 Installation & Setup
Prerequisites
Python 3.8 or higher
pip package manager
Step-by-Step Installation
# 1. Clone the repository
git clone https://github.com/prashantkadu25/RAG-PDF-Assistant.git
cd RAG-PDF-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
streamlit run app.py

# 4. Open your browser and go to http://localhost:8501
