# Multimodal RAG Pipeline using NVIDIA NIM, Milvus & Llama3

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Aaryan04/Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3)](https://github.com/Aaryan04/Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3/issues)
[![GitHub Stars](https://img.shields.io/github/stars/Aaryan04/Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3)](https://github.com/Aaryan04/Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3/stargazers)

## 🚀 Overview

A production-ready **Multimodal Retrieval-Augmented Generation (RAG)** system built with cutting-edge AI technologies. This application seamlessly processes diverse document formats including text files, PDFs, PowerPoint presentations, and images, leveraging NVIDIA's NIM microservices, Milvus vector database, and Llama3 for intelligent content understanding and retrieval.

### 🎯 Key Highlights

- **Enterprise-Grade Performance**: Built on NVIDIA NIM microservices for high-throughput inference
- **Multimodal Intelligence**: Processes text, images, charts, and complex documents with advanced AI models
- **Scalable Vector Search**: Powered by Milvus with GPU acceleration support
- **Production Ready**: Comprehensive error handling, logging, and monitoring capabilities
- **Interactive Interface**: User-friendly Streamlit web application

## 🏗️ Architecture

![Architecture Diagram](images/architecture_diagram.png)

*The system architecture shows the flow from document ingestion through multimodal processing to vector storage and retrieval, enabling intelligent question-answering capabilities.*

## ✨ Features

### 📄 Document Processing
- **Multi-format Support**: Text files (`.txt`, `.md`), PDFs, PowerPoint presentations (`.pptx`), and images (`.jpg`, `.png`, `.gif`)
- **Advanced Text Extraction**: Intelligent extraction from PDFs and PowerPoint slides including tables and embedded content
- **OCR Capabilities**: Extract text from images and scanned documents
- **Batch Processing**: Process entire directories with multiple file types simultaneously

### 🤖 AI-Powered Analysis
- **Vision Language Models**: NVIDIA NeVA for comprehensive image understanding
- **Chart Analysis**: Google's DePlot integration for graph and chart interpretation
- **Semantic Understanding**: Advanced text comprehension using Llama3
- **Context-Aware Responses**: Maintains conversation context for better user experience

### 🔍 Vector Search & Retrieval
- **High-Performance Indexing**: Milvus vector database with GPU acceleration
- **Similarity Search**: Advanced semantic similarity matching
- **Real-time Retrieval**: Fast query processing with optimized vector operations
- **Scalable Storage**: Handles large document collections efficiently

### 💬 Interactive Interface
- **Chat-Based Queries**: Natural language interaction with your documents
- **Real-time Processing**: Live document processing with progress indicators
- **Multi-session Support**: Handle multiple document collections
- **Export Capabilities**: Save conversations and results

## 🛠️ Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.10 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 5GB free space for dependencies and vector storage
- **GPU** (Optional): NVIDIA GPU for accelerated vector operations

### Required Services
- **Docker & Docker Compose**: For Milvus vector database
- **NVIDIA API Key**: Access to NVIDIA NIM microservices
- **Internet Connection**: For model downloads and API access

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aaryan04/Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3.git
cd Multimodal-RAG-pipeline-using-NVIDIA-NIM-Milvus-Llama3
```

### 2. Set Up Python Environment

#### Using Conda (Recommended)
```bash
conda create --name multimodal-rag python=3.10
conda activate multimodal-rag
```

#### Using Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Create .env file
echo "NVIDIA_API_KEY=your-nvidia-api-key-here" > .env

# Or export directly
export NVIDIA_API_KEY="your-nvidia-api-key-here"
```

> **Note**: Get your NVIDIA API key from [NVIDIA NGC](https://catalog.ngc.nvidia.com/)

### 5. Start Milvus Vector Database

#### Download Milvus Docker Compose
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

#### Start Milvus (GPU-accelerated)
```bash
# For GPU acceleration
sudo docker compose up -d

# Verify Milvus is running
docker ps | grep milvus
```

#### Alternative: CPU-only Milvus
```bash
# Download CPU version
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose-cpu.yml -O docker-compose.yml
sudo docker compose up -d
```

## 🎮 Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Access the Web Interface
Open your browser and navigate to: `http://localhost:8501`

### 3. Process Documents

#### Option A: Upload Files
1. Click **"Browse files"** in the sidebar
2. Select multiple files (PDF, PPTX, images, text files)
3. Click **"Process Files"** to start indexing

#### Option B: Process Directory
1. Enter the **directory path** containing your documents
2. Click **"Process Directory"** to process all supported files

### 4. Query Your Documents
1. Wait for processing to complete
2. Use the chat interface to ask questions about your documents
3. Get contextual answers with source references

### 5. Advanced Features
- **Multi-turn Conversations**: Ask follow-up questions
- **Source Attribution**: See which documents informed each answer
- **Export Results**: Save conversations for later reference

## 📱 Application Interface

![Streamlit App Screenshot](images/streamlit_app_screenshot.png)

*The intuitive Streamlit interface allows users to easily upload documents, process them, and interact with their content through natural language queries.*

## 📁 Project Structure

```
multimodal-rag/
├── 📄 app.py                    # Main Streamlit application
├── 🔧 utils.py                  # Utility functions for image processing
├── 📋 document_processors.py    # Document processing functions
├── 📦 requirements.txt          # Python dependencies
├── 🗃️ vectorstore/             # Auto-generated vector storage
├── 🐳 docker-compose.yml        # Milvus database configuration
├── 📖 README.md                 # This file
└── 📄 .env                      # Environment variables (create this)
```

## 🔧 Configuration

### Milvus Configuration
```python
# In app.py - customize vector store settings
vector_store = MilvusVectorStore(
    host="127.0.0.1",
    port=19530,
    dim=1024,
    collection_name="multimodal_docs",
    gpu_id=0  # For GPU acceleration
)
```

### Model Configuration
```python
# Customize AI models in utils.py
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL = "meta/llama-3.1-8b-instruct"
VLM_MODEL = "nvidia/neva-22b"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Milvus Connection Failed
```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus
sudo docker compose down
sudo docker compose up -d

# Check logs
docker logs milvus-standalone
```

#### 2. NVIDIA API Key Issues
```bash
# Verify API key is set
echo $NVIDIA_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models
```

#### 3. Memory Issues
- Reduce batch size in `document_processors.py`
- Process fewer files simultaneously
- Use CPU-only Milvus for lower memory usage

#### 4. GPU Acceleration Not Working
```bash
# Check GPU availability
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Performance Optimization

#### For Large Document Collections
1. **Increase batch size**: Modify `BATCH_SIZE` in `document_processors.py`
2. **Use GPU acceleration**: Ensure Milvus GPU support is enabled
3. **Optimize chunk size**: Adjust text chunking parameters

#### For Better Response Quality
1. **Adjust similarity threshold**: Modify retrieval parameters
2. **Fine-tune chunk overlap**: Improve context preservation
3. **Customize prompt templates**: Enhance response formatting





## 👨‍💻 Author

**Aaryan Shah**
- GitHub: [@Aaryan04](https://github.com/Aaryan04)
- Email: shah.aaryan27117@gmail.com
- LinkedIn: [Connect with me](https://linkedin.com/in/aaryanshah04)

## 🙏 Acknowledgments

- **NVIDIA**: For providing cutting-edge NIM microservices and AI models
- **Milvus**: For the high-performance vector database
- **LlamaIndex**: For the excellent RAG framework
- **Streamlit**: For the intuitive web application framework
- **Open Source Community**: For the amazing tools and libraries

## 🌟 Support

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting issues
- 💡 Suggesting improvements
- 🤝 Contributing to the codebase

---

<div align="center">
  <b>Built with ❤️ by Aaryan Shah</b>
  <br>
  <sub>Powered by NVIDIA NIM, Milvus, and Llama3</sub>
</div>