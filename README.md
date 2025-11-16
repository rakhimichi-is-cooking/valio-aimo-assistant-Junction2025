# Valio AI - Supply Chain Forecasting System

Graph Neural Network-based demand forecasting and shortage detection for supply chain management.

## Features

- **GNN Forecasting**: product relationships using PyTorch Geometric
- **Shortage Detection**: Multi-factor risk scoring and substitution suggestions
- **Real-time Predictions**: LSTM+GNN hybrid architecture for demand forecasting

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_demo.txt  # For demo features
```

### 2. Build Product Graph

```bash
python extract_product_features.py
python build_product_graph_v2.py
```

This creates a verified graph of 696 products with 111,969 edges from sales data.

### 3. Run Applications

**Shortage Dashboard** (Simple API frontend):
```bash
# Start backend
python -m uvicorn backend.main:app --reload

# Start frontend
streamlit run ui/app.py
```

**Chat Interface**:
```bash
# CLI
python chat.py

# Web UI
streamlit run chat_web.py
```

## Architecture

### Backend (`backend/`)
- `main.py` - FastAPI REST API
- `gnn_forecaster.py` - Graph Neural Network forecaster
- `neural_forecaster.py` - Traditional LSTM/GRU forecaster
- `shortage_engine.py` - Shortage detection
- `replacement_engine.py` - Product substitution suggestions
- `aimo_agent.py` - AI agent for customer communications
- `conversational_agent.py` - Natural language query interface

### Applications
- `demo_app.py` - Full GNN demo with vision AI (572 lines)
- `frontend/app.py` - Simple shortage dashboard (111 lines)
- `chat.py` - CLI chat interface
- `chat_web.py` - Web-based chat interface

### Data Utilities
- `build_product_graph_v2.py` - Builds product relationship graph
- `extract_product_features.py` - Extracts product features

## Data Flow

1. **Graph Construction**: Sales data → Co-purchase patterns → 111,969 edges
2. **GNN Training**: Historical sequences + Graph structure → Trained model
3. **Prediction**: Current demand + Neighbor signals → 7-day forecast
4. **Shortage Detection**: Forecast + Inventory → Risk scores + Substitutes

## Tech Stack

- **PyTorch + PyTorch Geometric** - GNN implementation
- **FastAPI** - REST API backend
- **Streamlit** - Web interfaces
- **LM Studio** - Local LLM for explanations and vision AI
- **Pandas + NumPy** - Data processing

## Documentation

- `CLAUDE.md` - Project instructions for AI assistants

## License

Built for Valio Aimo Hackathon 2025
