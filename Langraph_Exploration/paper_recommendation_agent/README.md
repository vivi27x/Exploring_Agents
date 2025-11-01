# 🤖 Academic Paper Recommendation Agent

An AI agent that automates finding relevant academic papers based on your research interests. The agent reasons, plans, and executes paper discovery using fine-tuned models.

## Features

- **Multi-agent architecture** with specialized planners, searchers, and analyzers
- **Fine-tuned relevance model** using LoRA for personalized recommendations
- **Vector search** with ChromaDB and semantic embeddings
- **Free open-source models** (Mistral, Llama 3 via Ollama)
- **Web interface** with Streamlit
- **Comprehensive evaluation** framework

## Quick Start

## Free Cloud Setup (No Local Models)

### Option 1: Hugging Face Inference API (Recommended)

1. **Get free API token:**
   - Create account at [huggingface.co](https://huggingface.co)
   - Go to Settings → Access Tokens
   - Create new token with "read" access

2. **Setup environment:**
   ```bash
   python setup_environment.py