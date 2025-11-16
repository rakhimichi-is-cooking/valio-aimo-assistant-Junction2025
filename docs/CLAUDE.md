# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Valio AI, a state-of-the-art supply chain forecasting system that uses Graph Neural Networks (GNN) to predict demand shortages and suggest substitutions. The system combines temporal LSTM models with graph-based relationships between 696 products using 111,969 verified edges.

## Architecture

### Backend Components (`backend/`)
- **main.py**: FastAPI server with REST endpoints for shortage detection, forecasting, and messaging
- **gnn_forecaster.py**: Core GNN implementation using PyTorch Geometric with LSTM+GraphSAGE hybrid architecture
- **neural_forecaster.py**: Traditional neural forecaster for comparison
- **aimo_agent.py**: AI agent for generating customer communications using LLM integration
- **shortage_engine.py**: Core logic for detecting shortage events from sales/delivery data
- **replacement_engine.py**: Substitution suggestion engine using product similarity
- **data_loader.py**: Data loading utilities for sales, purchases, and product catalogs
- **pattern_analysis.py**: Historical pattern analysis and insights generation
- **forecasting_engine.py**: Multi-method forecasting (Prophet, ARIMA, XGBoost, ensemble)

### Frontend/Demo
- **demo_app.py**: Streamlit web application for interactive demos
- **frontend/app.py**: Alternative frontend interface

### Data Infrastructure
- **data/**: Product catalogs, sales data, and the built product graph
- **build_product_graph_v2.py**: Builds the 111,969-edge product relationship graph
- **extract_product_features.py**: Feature extraction from product data

## Key Commands

### Environment Setup
```bash
pip install -r requirements.txt
# For demo functionality:
pip install -r requirements_demo.txt
```

### Data Preparation
```bash
# Build the product relationship graph (required first step)
python build_product_graph_v2.py

# Extract product features
python extract_product_features.py
```

### Running Applications
```bash
# Start FastAPI backend server
python backend/main.py
# Alternative: uvicorn backend.main:app --reload

# Start Streamlit demo application
streamlit run demo_app.py

# Start alternative frontend
python frontend/app.py
```

### Testing and Validation
```bash
# Test GNN forecaster
python test_gnn_forecaster.py

# Run neural network validation
python test_neural_validation.py

# Run comprehensive verification
python run_full_verification.py

# Verify graph authenticity
python verify_graph_not_fake.py
python verify_graph_confidence.py
```

## Development Architecture

### Graph Neural Network Pipeline
1. **Data Loading**: Sales/delivery data loaded via `data_loader.py`
2. **Graph Construction**: Product relationships built using co-purchase patterns, demand correlations, and substitutions
3. **Model Training**: Hybrid LSTM+GNN model trains on temporal sequences with graph structure
4. **Prediction**: Real-time forecasting leveraging both historical patterns and network effects

### Key Design Patterns
- **Hybrid Architecture**: Combines temporal (LSTM) and spatial (GNN) information
- **Verification System**: Three-layer validation ensuring data authenticity (basic, statistical, independent)
- **Modular Forecasting**: Multiple forecasting methods available (Prophet, ARIMA, XGBoost, GNN)
- **LLM Integration**: Uses LM Studio for vision AI (stock photo analysis) and forecast explanations

### Data Flow
1. Raw sales data → Product graph construction → Edge verification
2. Historical demand sequences → GNN training → Trained model
3. Real-time prediction: Current demand + Graph neighbors → Forecast
4. Post-processing: Risk scoring, substitute suggestions, customer messaging

## Critical Dependencies

### Required Packages
- **PyTorch + PyTorch Geometric**: For GNN implementation
- **FastAPI + Uvicorn**: Backend API server
- **Streamlit**: Demo web interface
- **Pandas + NumPy**: Data manipulation
- **Scikit-learn**: Traditional ML methods

### External Services
- **LM Studio**: Local LLM server for AI explanations and vision analysis (runs on localhost:1234)
- **Redis**: Optional caching backend

## File Structure Context

### Core Modules
- Backend modules are designed to be imported individually (`from backend.gnn_forecaster import GNNForecaster`)
- All forecasters implement a common interface with `predict()` method returning standardized format
- Data loaders handle CSV parsing and provide consistent DataFrame formats

### Graph Data Format
- Product graph stored as pickle file: `data/product_graph/product_graph.pkl`
- Graph metadata available in: `data/product_graph/graph_summary.json`
- Verification results: `data/product_graph/verification_results.json`

## Important Notes

- The product graph MUST be built before running GNN forecaster (`python build_product_graph_v2.py`)
- LM Studio must be running on port 1234 for demo vision AI features
- System supports both CPU and GPU training (auto-detects CUDA availability)
- All product codes are treated as strings throughout the system for consistency
- The system has been extensively verified - 100% of randomly sampled edges verified against raw data

## Demo Configuration

The Streamlit demo expects:
1. LM Studio running with vision model loaded
2. Product graph built and available
3. Sample sales data in `data/` directory
4. All dependencies installed from `requirements_demo.txt`