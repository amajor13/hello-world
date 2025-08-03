# 🤖 RAG Chatbot - End-to-End Implementation

A comprehensive Retrieval-Augmented Generation (RAG) chatbot system that allows you to chat with your documents using advanced AI technology. This project combines document processing, vector search, and large language models to create an intelligent question-answering system.

## 🌟 Features

- **🆓 100% FREE**: Works completely with open-source models - no API keys required!
- **📄 Multi-Format Support**: PDF, DOCX, TXT, and Markdown files
- **🧠 Smart Processing**: Intelligent chunking and embedding generation
- **🔍 Vector Search**: Efficient similarity search using ChromaDB
- **💬 Conversation Memory**: Context-aware responses with conversation history
- **🎯 Dual Interface**: Both beautiful web UI and command-line interface
- **⚙️ Flexible Models**: Free models (default) or OpenAI models (optional)
- **📚 Source Attribution**: Always shows which documents were used for answers
- **📤 Real-time Upload**: Add documents on-the-fly via web interface
- **🎨 Modern Design**: Beautiful, responsive interface with custom styling
- **⚡ Quick Launch**: One-command setup and launch

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │  Vector Store   │    │   LLM Model     │
│  (PDF, DOCX,    │───▶│   (ChromaDB)    │───▶│   (OpenAI or    │
│   TXT, MD)      │    │                 │    │    Local)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document Loader │    │   Embeddings    │    │ Response Generator│
│   & Chunking    │    │   & Retrieval   │    │ & Source Citing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │   User Interface│
                    │ (Web + CLI)     │
                    └─────────────────┘
```

## 🚀 Quick Start (100% FREE!)

### 🎯 Super Quick Launch (Recommended)

```bash
# Just run this - it handles everything!
python launch.py
```

### 🛠️ Manual Setup

```bash
git clone <repository-url>
cd rag-chatbot
pip install -r requirements.txt
python launch.py
```

### 🔧 Configuration

**Default (FREE models)** - No setup needed!
```env
USE_OPENAI=false
FREE_LLM_MODEL=conversational     # Microsoft DialoGPT
FREE_EMBEDDING_MODEL=fast        # Sentence Transformers
```

**Optional (OpenAI models)** - Better quality but requires API key:
```env
USE_OPENAI=true
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. Add Documents

Place your documents in the `documents/` folder:
- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- Markdown files (`.md`)

### 4. Run the Application

**🎯 Easiest Way:**
```bash
python launch.py  # Choose CLI or Web interface
```

**🌐 Web Interface:**
```bash
streamlit run app.py
```

**💻 Command Line Interface:**
```bash
python cli_chat.py --load-docs  # Load documents first
python cli_chat.py             # Start chatting
```

## 📖 Usage Examples

### Web Interface

1. **Initialize**: Click "Initialize Chatbot" in the sidebar
2. **Load Documents**: Use "Load Documents" to process your files
3. **Upload Files**: Drag and drop files directly in the web interface
4. **Chat**: Ask questions about your documents in natural language

### Command Line Interface

```bash
# Load documents and start interactive chat
python cli_chat.py --load-docs

# Single query mode
python cli_chat.py --query "What is machine learning?"

# Use local models instead of OpenAI
python cli_chat.py --local --load-docs

# Reset vector store and reload documents
python cli_chat.py --load-docs --reset

# Show system status
python cli_chat.py --status
```

### Sample Queries

- "What are the main types of machine learning?"
- "Explain how Python classes work"
- "What are the benefits of using virtual environments?"
- "How do I implement error handling in Python?"

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- **No API keys required!** (OpenAI API key optional for enhanced performance)

### Manual Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Launch the chatbot
python launch.py
```

### Alternative Setup

```bash
# Traditional setup with virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python launch.py
```

## 📁 Project Structure

```
rag-chatbot/
├── src/                      # Core application modules
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── document_loader.py   # Document processing
│   ├── vector_store.py      # Vector database operations
│   └── rag_system.py        # Main RAG implementation
├── documents/               # Document storage
│   ├── sample_ai_overview.txt
│   └── python_programming_guide.md
├── vector_store/           # Vector database (auto-generated)
├── app.py                  # Streamlit web application
├── cli_chat.py            # Command-line interface
├── setup.py               # Automated setup script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | None |
| `OPENAI_MODEL` | OpenAI model to use | gpt-3.5-turbo |
| `EMBEDDING_MODEL` | Embedding model | text-embedding-ada-002 |
| `USE_OPENAI` | Use OpenAI or local models | true |
| `VECTOR_STORE_PATH` | Vector store directory | ./vector_store |
| `DOCUMENTS_PATH` | Documents directory | ./documents |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `RETRIEVAL_K` | Number of documents to retrieve | 4 |

### Customization Options

**Text Chunking:**
```python
# Adjust in src/config.py
CHUNK_SIZE = 1000      # Larger for more context
CHUNK_OVERLAP = 200    # Higher for better continuity
```

**Retrieval Settings:**
```python
RETRIEVAL_K = 4        # More documents for comprehensive answers
```

**Model Selection:**
```python
OPENAI_MODEL = "gpt-4"  # Use GPT-4 for better responses
```

## 🧪 Testing

### Built-in Sample Documents

The project includes sample documents to test the system:

1. **AI Overview** (`documents/sample_ai_overview.txt`)
   - Comprehensive guide to AI and machine learning
   - Test queries: "What is deep learning?", "Explain supervised learning"

2. **Python Guide** (`documents/python_programming_guide.md`)
   - Complete Python programming tutorial
   - Test queries: "How do I create a class?", "What are list comprehensions?"

### Running Tests

```bash
# Test system status
python cli_chat.py --status

# Test with sample queries
python cli_chat.py --query "What is artificial intelligence?"
python cli_chat.py --query "How do Python functions work?"
```

## 🔧 Advanced Features

### Local Model Support

For privacy or cost considerations, you can use local models:

```bash
# Set in .env
USE_OPENAI=false

# Run with local models
python cli_chat.py --local --load-docs
```

### Custom Document Processing

Extend document support by modifying `src/document_loader.py`:

```python
def load_custom_format(self, file_path: str) -> str:
    # Add support for new file formats
    pass
```

### Custom Embeddings

Use different embedding models by modifying `src/vector_store.py`:

```python
# Use Hugging Face transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('your-preferred-model')
```

## 🔍 Troubleshooting

### Common Issues

**1. "No documents loaded" error:**
```bash
# Solution: Load documents first
python cli_chat.py --load-docs
```

**2. OpenAI API errors:**
```bash
# Check API key
echo $OPENAI_API_KEY
# Or use local models
python cli_chat.py --local
```

**3. Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**4. Vector store issues:**
```bash
# Reset vector store
python cli_chat.py --reset --load-docs
```

### Performance Tips

1. **Optimize chunk size** based on your document types
2. **Use GPT-4** for better comprehension (higher cost)
3. **Increase retrieval_k** for more comprehensive answers
4. **Local models** for faster responses (lower quality)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Development Setup

```bash
git clone <repository-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Areas for Improvement

- [ ] Support for more document formats (CSV, JSON, etc.)
- [ ] Advanced chunking strategies
- [ ] Multi-language support
- [ ] Integration with more LLM providers
- [ ] Advanced conversation management
- [ ] Document metadata extraction
- [ ] Caching mechanisms
- [ ] Performance monitoring

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - For the comprehensive RAG framework
- **OpenAI** - For powerful language models and embeddings
- **ChromaDB** - For efficient vector storage and retrieval
- **Streamlit** - For the beautiful web interface
- **Sentence Transformers** - For local embedding models

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check this README and code comments
- **Community**: Join discussions in GitHub Discussions

---

**Happy chatting with your documents! 🤖📚**
