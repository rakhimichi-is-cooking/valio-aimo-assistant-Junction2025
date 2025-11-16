# 🚚 Valio Aimo – Zero-Fail Logistics Assistant  
### Junction Hackathon 2025 (Helsinki, 14–16 November)  
**Team: Western Buddies**

Hi! I’m Kirill — thanks for checking out this project.

This repository contains the final working version of our solution for the **Zero-Fail Logistics** challenge by **Valio Aimo**, created during **Junction Hackathon 2025** in Helsinki.

Our team **Western Buddies** delivered a complete operator-focused tool that helps Valio Aimo handle delivery issues, product shortages, and customer communication efficiently and transparently.  
The solution emphasizes **practical logic, clear UX, and workflow automation** — without unnecessary complexity or heavy ML stacks.

---

## 🧭 Overview

This assistant supports logistics operators by enabling them to:

- Inspect delivery issues and shortage scenarios  
- Quickly draft customer messages (email, phone, SMS)  
- View replacement product suggestions  
- Navigate through a clear, operator-friendly dashboard  
- Run terminal-based demo tools for scenario exploration  

The project is fully local, lightweight, and easy to run.

---

## 🧩 Key Features

- **Shortage & delivery issue inspector**  
- **Explanation blocks** to guide operator decisions  
- **Replacement suggestion system** (rule-based, metadata-driven)  
- **Customer message generator**  
- **Streamlit dashboard** for real-time case navigation  
- **Terminal utilities** for demo and debugging  
- 100% **offline** — no external services required  

---

## 👤 My Role

During the hackathon I focused on **system design, operator workflows, and integrating all components into a polished, working product**.

My contributions included:

- Designing the full workflow:  
  **shortage → explanation → replacement → customer message**
- Building and refining the **Streamlit dashboard**
- Creating **terminal demo tools** used during testing and the final pitch  
- Integrating backend logic, data flow, and UI into a cohesive system  
- Preparing scenario materials and assisting with presentation structure  

In short: **I acted as a systems integrator and product engineer — ensuring that every part of the product worked seamlessly together and was ready to showcase.**

---

## ▶️ How to Run the Project (Step-by-Step)

### 1. Clone the repository

`git clone`

`cd valio-aimo-case`

### 2. Create and activate a virtual environment

`python3 -m venv .venv`

`source .venv/bin/activate`        # macOS / Linux

### or

`.\.venv\Scripts\activate`         # Windows

### 3. Install dependencies

`pip install streamlit plotly pandas numpy requests`

### 4. Data files (IMPORTANT)

The UI requires local data files.

Expected structure:

`ui/data/
├── valio_aimo_sales_and_deliveries_junction_2025.csv
├── valio_aimo_purchases_junction_2025.csv`

If these files are missing, the dashboard will run but metrics and graphs will be empty.

---

## ▶️ Using the System

### 1. Launch the Operator Dashboard (UI)

`cd ui`

`streamlit run app.py`

You can:

Inspect shortages and delivery issues
View suggested replacements
Generate customer-facing communication
Navigate scenario data

### 2. Run Backend Logic (optional)

`cd backend`

`python main.py`

Backend includes:

Shortage scenario construction
Replacement product logic
Explanation generation
Basic validation utilities

### 3. Terminal Tools

Full scenario demo:

`python tools/demo_app.py`

CLI-based operator assistant:

`python tools/chat.py`

Web-based chat prototype:

`python tools/chat_web.py`

---

## 🗂️ Key Directories

| Directory        | Description |
|------------------|-------------|
| **backend/**     | Core shortage logic, replacement selection, explanation blocks |
| **ui/**          | Streamlit operator dashboard used in demos |
| **frontend/**    | Optional UI components and visual prototypes |
| **models/**      | Product metadata utilities, feature extractors (no ML models) |
| **tools/**       | CLI tools, demo scripts, terminal chat interfaces |
| **docs/**        | Pitch script, scenario plans, dashboard notes |
| **terminal-ui/** | Node-based terminal prototype (not used in final demo) |

---

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

---

## Data Flow

1. **Graph Construction**: Sales data → Co-purchase patterns → 111,969 edges
2. **GNN Training**: Historical sequences + Graph structure → Trained model
3. **Prediction**: Current demand + Neighbor signals → 7-day forecast
4. **Shortage Detection**: Forecast + Inventory → Risk scores + Substitutes

---

## Tech Stack

- **PyTorch + PyTorch Geometric** - GNN implementation
- **FastAPI** - REST API backend
- **Streamlit** - Web interfaces
- **LM Studio** - Local LLM for explanations and vision AI
- **Pandas + NumPy** - Data processing

---

## Documentation

- `CLAUDE.md` - Project instructions for AI assistants

---

## License

Built by **Western Buddies** team for Valio Aimo case at Junction Hackathon 2025, Helsinki


