# 🧮 Math Mentor

### AI-Powered JEE Mathematics Tutor with RAG, Multi-Agent System & Human-in-the-Loop

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Math Mentor is a **full-stack intelligent tutoring system** designed to solve
JEE-level mathematics problems with step-by-step reasoning, verification,
and continuous learning from user feedback.

---

## 📌 Key Highlights

| Feature | Description |
|---------|-------------|
| 🎯 **Multimodal Input** | Text, Image (Tesseract OCR), Audio (Whisper ASR) |
| 📚 **RAG Pipeline** | FAISS + Sentence Transformers for formula & method grounding |
| 🤖 **Multi-Agent System** | Parser → Router → Solver → Verifier → Explainer |
| 🛡️ **Guardrails** | Input validation, prompt injection detection, content safety |
| 🧑‍🤝‍🧑 **Human-in-the-Loop** | Confidence-based HITL for low-confidence OCR/ASR extractions |
| 🧠 **Memory System** | Learns from user corrections & feedback |
| 🐳 **Dockerised** | Single-command deployment with Docker Compose |


---

## ✨ Features

### 🧾 Input Methods

- **Text Input** — Type math problems directly
- **Image Input (OCR)** — Upload handwritten or printed problems via Tesseract
- **Audio Input (ASR)** — Speak your math questions via Whisper

### 🤖 AI Capabilities

- **Multi-Agent System** — Parser, Router, Solver, Verifier, Explainer
- **RAG Pipeline** — FAISS + Sentence Transformers (`all-MiniLM-L6-v2`)
- **Solution Verification** — Symbolic math validation
- **Memory System** — Learns from past problems and corrections

### 🛡️ Safety & Guardrails

| Check | Description | Action |
|-------|-------------|--------|
| Prompt Injection | Detects manipulation attempts | 🚫 Block |
| Off-Topic | Non-math queries | 🚫 Block |
| Harmful Content | Violence, weapons, drugs | 🚫 Block |
| PII Detection | Personal information | ⚠️ Warning |
| Length Validation | Too short/long inputs | 🚫 Block |
| Hallucination Detection | False citations in output | ⚠️ Flag + Remove |
| Completeness Check | Missing solution steps | ⚠️ Warning |

### 🧑‍🤝‍🧑 Human-in-the-Loop

| Input Type | HITL Threshold |
|------------|---------------|
| OCR | < 70% confidence |
| ASR | < 65% confidence |

- Editable OCR/ASR output before solving
- Learns correction patterns automatically
- Feedback-driven improvement loop

### 📚 Topics Covered

Algebra (Quadratics, Polynomials, Logs) · Calculus (Limits, Derivatives, Integrals) · Probability · Linear Algebra

---

## 🏗 Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                        FRONTEND                               │
│                   React + Vite → Nginx                        │
│         Text Input │ Image Upload │ Audio Record              │
└────────────┬─────────────────────────────────────────────────┘
             │ HTTP (/api/*)
             ▼
┌──────────────────────────────────────────────────────────────┐
│                     NGINX REVERSE PROXY                       │
│              Static files ← /                                 │
│              API proxy    ← /api/* → backend:8000             │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                           │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ Input Layer  │  │  Guardrails  │  │   Multi-Agent     │   │
│  │             │  │              │  │   System           │   │
│  │ • OCR       │  │ • Injection  │  │ • Parser           │   │
│  │ • ASR       │  │ • Topic      │  │ • Router           │   │
│  │ • Text      │  │ • Safety     │  │ • Solver           │   │
│  │ • HITL      │  │ • Output     │  │ • Verifier         │   │
│  └──────┬──────┘  └──────┬───────┘  │ • Explainer        │   │
│         │                │          └─────────┬───────────┘   │
│         ▼                ▼                    │               │
│  ┌─────────────────────────────────┐          │               │
│  │         RAG Engine              │◄─────────┘               │
│  │  FAISS + Sentence Transformers  │                          │
│  │  (all-MiniLM-L6-v2)            │                          │
│  └─────────────┬───────────────────┘                          │
│                │                                              │
│  ┌─────────────▼───────────────────┐  ┌──────────────────┐   │
│  │      Knowledge Base             │  │   Memory DB      │   │
│  │  algebra · calculus             │  │   SQLite +       │   │
│  │  probability · linear_algebra   │  │   Feedback Loop  │   │
│  │  common_mistakes               │  │                  │   │
│  └─────────────────────────────────┘  └──────────────────┘   │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              Groq LLM (Cloud API)                      │   │
│  └────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```
## System Flow

1. User submits problem (text / image / audio)
2. OCR / ASR extracts text (if applicable)
3. Confidence score is computed
4. Low confidence → HITL review
5. Guardrails validate input
6. RAG retrieves relevant formulas & methods
7. Multi-Agent system solves & verifies
8. Guardrails validate output
9. Solution displayed with step-by-step explanation
10. User feedback stored → Memory learns

## ⚙️ Tech Stack

| Layer              | Technology                                    |
|--------------------|----------------------------------------------|
| Frontend           | React 18, Vite, CSS                          |
| Backend            | Python 3.11, FastAPI, Uvicorn                |
| LLM                | Groq API                                     |
| Embeddings         | Sentence Transformers (all-MiniLM-L6-v2)     |
| Vector Store       | FAISS                                        |
| OCR                | Tesseract                                    |
| ASR                | Whisper (via Groq)                           |
| Database           | SQLite                                       |
| Reverse Proxy      | Nginx                                        |
| Containerisation   | Docker, Docker Compose                       |
| Registry           | Docker Hub                                   |
| Hosting            | AWS EC2                                      |

## Project Structure
```
math-mentor/
│
├── docker-compose.yml          # Local development orchestration
├── docker-compose.prod.yml     # Production orchestration
├── Dockerfile.backend          # Multi-stage backend build
├── Dockerfile.frontend         # Multi-stage frontend build
├── nginx.conf                  # Nginx reverse proxy config
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
├── .dockerignore               # Docker build exclusions
├── README.md                   # This file
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App entry point, routes, middleware
│   ├── middleware.py            # CORS, logging middleware
│   └── routes/
│       ├── __init__.py
│       ├── health.py           # GET /health
│       ├── solve.py            # POST /solve
│       ├── rag.py              # RAG endpoints
│       ├── guardrails.py       # Guardrails endpoints
│       ├── memory.py           # Memory/feedback endpoints
│       └── evaluation.py       # Evaluation metrics
│
├── src/                        # Core business logic
│   ├── __init__.py
│   │
│   ├── input_processing/
│   │   ├── __init__.py
│   │   ├── schemas.py          # Pydantic models
│   │   ├── ocr_processor.py    # Tesseract OCR
│   │   ├── asr_processor.py    # Whisper ASR
│   │   ├── math_normalizer.py  # LaTeX/Unicode normalisation
│   │   └── text_processor.py   # Text cleaning
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── crew_setup.py       # Multi-agent orchestration
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── calculator.py   # SymPy calculator tool
│   │
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guardrails.py
│   │   ├── output_guardrails.py
│   │   └── content_filter.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py        # FAISS retrieval logic
│   │   └── knowledge_base/
│   │       ├── algebra.json
│   │       ├── calculus.json
│   │       ├── probability.json
│   │       ├── linear_algebra.json
│   │       └── common_mistakes.json
│   │
│   └── memory/
│       ├── __init__.py
│       ├── database.py         # SQLite operations
│       └── memory_manager.py   # Feedback learning logic
│
├── frontend/                   # React application
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── public/
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── components/         # UI components
│       └── assets/             # Static assets
│
└── data/
    ├── vector_store/
    │   ├── faiss.index
    │   └── documents.json
    └── math_mentor.db          # SQLite database
```


## 📦 Modules

### 🔹 Module 1: Input Processing

- **OCR** — Tesseract-based extraction with confidence scoring  
- **ASR** — Whisper-based transcription with confidence scoring  
- **HITL (Human-in-the-Loop)** — Triggered when confidence falls below thresholds  
- **Normalizer** — LaTeX/Unicode math normalization  

---

### 🔹 Module 2: Multi-Agent System

| Agent      | Role                                      |
|------------|------------------------------------------|
| Parser     | Structures the raw problem               |
| Router     | Chooses solving strategy                |
| Solver     | Performs mathematical computation       |
| Verifier   | Validates the solution                  |
| Explainer  | Generates student-friendly steps        |

---

### 🔹 Module 3: RAG Pipeline

- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Vector Store**: FAISS index over curated math knowledge base  
- **Knowledge Domains**:
  - Algebra  
  - Calculus  
  - Probability  
  - Linear Algebra  
  - Common Mistakes  

---

### 🔹 Module 4: Memory System

- SQLite-backed persistent storage  
- Stores:
  - Solved problems  
  - Generated solutions  
  - User feedback  
- Features:
  - Reuses successful solution strategies  
  - Learns from user corrections over time  

---

### 🔹 Module 5: Guardrails

| Layer   | Component              | Purpose                                      |
|---------|----------------------|----------------------------------------------|
| Input   | `input_guardrails.py` | Sanitisation, length checks, injection detection |
| Input   | `content_filter.py`  | Topic enforcement, safety filtering          |
| Output  | `output_guardrails.py` | Hallucination detection, completeness check |

---


### Contributing
1. Fork the repo
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## License
MIT License © 2025


## 🙏 Acknowledgments

| Technology              | Purpose                          |
|------------------------|----------------------------------|
| Groq                   | Ultra-fast LLM inference         |
| FAISS                  | Vector similarity search         |
| Sentence Transformers  | Text embeddings                  |
| Tesseract              | Optical character recognition    |
| Whisper                | Speech recognition               |
| SymPy                  | Symbolic mathematics             |
| FastAPI                | Backend framework                |
| React                  | Frontend framework               |
| Docker                 | Containerisation                 |
