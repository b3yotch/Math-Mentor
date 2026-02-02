# 🧮 Math Mentor
AI-Powered JEE Mathematics Tutor with RAG, Multi-Agent System & Human-in-the-Loop

**Technologies:** Python, Streamlit  
**License:** MIT License

Math Mentor is an intelligent tutoring system designed to solve JEE-level mathematics problems with step-by-step reasoning, verification, and continuous learning from user feedback.

---

## 📌 Key Highlights
- **Multimodal input:** Text, Image (OCR), Audio (ASR)  
- **Retrieval-Augmented Generation (RAG)** for formula & method grounding  
- **Multi-Agent reasoning system**  
- **Human-in-the-Loop (HITL)** for low-confidence OCR/ASR  
- Learns from user corrections & feedback  
- Built with **Streamlit + Groq + FAISS**

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [System Flow](#-system-flow)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Modules](#-modules)
- [HITL Implementation](#-human-in-the-loop-hitl)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

---

## ✨ Features

### 🧾 Input Methods
- **Text Input** – Type math problems directly  
- **Image Input (OCR)** – Upload handwritten or printed problems  
- **Audio Input (ASR)** – Speak your math questions  

### 🤖 AI Capabilities
- **Multi-Agent System** (Parser, Router, Solver, Verifier, Explainer)
- **RAG Pipeline** using FAISS + Sentence Transformers
- **Solution Verification** using symbolic math
- **Memory System** for learning from past problems

### 🛡️ Safety & Guardrails ⬅️ NEW
- **Input Validation** – Sanitization & length checks
- **Prompt Injection Detection** – Blocks manipulation attempts
- **Topic Enforcement** – Math-only queries allowed
- **Content Safety Filter** – Blocks harmful/inappropriate content
- **Output Validation** – Ensures quality responses
- **Hallucination Detection** – Flags false citations

### 🧑‍🤝‍🧑 Human-in-the-Loop
- Confidence-based HITL triggering
- Editable OCR/ASR output
- Learns correction patterns automatically
- Feedback-driven improvement loop
- 
### 📚 Topics Covered
- Algebra (Quadratics, Polynomials, Logs)  
- Calculus (Limits, Derivatives, Integrals)  
- Probability  
- Linear Algebra  

---

## 🏗 Architecture Overview
High-level pipeline:

```
User Input (Text / Image / Audio)
↓
OCR / ASR / Text Processing
↓
Confidence Check → HITL (if required)
↓
Math Normalization
↓
RAG Retrieval + Memory Lookup
↓
Multi-Agent Reasoning (Groq LLM)
↓
Verification + Explanation
↓
User Feedback → Learning
```


---

## 🔄 System Flow
1. User submits a math problem  
2. OCR / ASR extracts text (if applicable)  
3. Confidence score is computed  
4. Low confidence → HITL review  
5. Canonical math input is generated  
6. RAG retrieves relevant formulas & methods  
7. Multi-agent system solves & verifies  
8. Solution is displayed  
9. User feedback is stored for learning  

---

## 📁 Project Structure

```
math-mentor/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
├── .env.example # Example environment file
├── README.md # This file
│
├── config/
│ ├── init.py
│ └── prompts.py # Agent prompts and backstories
│
├── src/
│ ├── init.py
│ │
│ ├── input_processing/
│ │ ├── init.py
│ │ ├── schemas.py
│ │ ├── ocr_processor.py
│ │ ├── asr_processor.py
│ │ ├── math_normalizer.py
│ │ └── text_processor.py
│ │
│ ├── agents/
│ │ ├── init.py
│ │ ├── crew_setup.py
│ │ └── tools/
│ │ ├── init.py
│ │ └── calculator.py
│ │
│ ├── guardrails/
│ │ ├── init.py
│ │ ├── input_guardrails.py
│ │ ├── output_guardrails.py
│ │ └── content_filter.py
│ │
│ ├── rag/
│ │ ├── init.py
│ │ ├── retriever.py
│ │ └── knowledge_base/
│ │ ├── algebra.json
│ │ ├── calculus.json
│ │ ├── probability.json
│ │ ├── linear_algebra.json
│ │ └── common_mistakes.json
│ │
│ └── memory/
│ ├── init.py
│ ├── database.py
│ └── memory_manager.py
│
├── data/
│ ├── vector_store/
│ │ ├── faiss.index
│ │ └── documents.json
│ └── math_mentor.db
```


---

## 🚀 Installation

### Prerequisites
- Python 3.10+  
- Tesseract OCR  
- Groq API Key  

### Step 1: Clone Repository

git clone https://github.com/b3yotch/Math-mentor.git

cd math-mentor

### Step 2: Create Virtual Environment
python -m venv venv


Windows:

.\venv\Scripts\activate


macOS / Linux:

source venv/bin/activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Install Tesseract OCR

Windows:
https://github.com/UB-Mannheim/tesseract/wiki

### ⚙️ Configuration

Create .env file:

cp .env.example .env


Add your API key:

GROQ_API_KEY=gsk_your_api_key_here

### ▶️ Usage

Run the application:

streamlit run app.py

## 📦 Modules

## Module 1: Input Processing

OCR (Tesseract)

ASR (Whisper)

Confidence scoring

HITL trigger

## Module 2: Multi-Agent System

Agent      Role
Parser     Structure problem
Router     Choose strategy
Solver     Perform math
Verifier   Validate solution
Explainer  Student-friendly steps


## Module 3: RAG Pipeline

Sentence Transformers

FAISS Vector Store

Curated math knowledge base

## Module 4: Memory System

SQLite storage

Learns corrections

Reuses successful solutions

## Module 5: Safety Guardrails 🛡️

The guardrails system provides comprehensive input/output validation and content safety.

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Input Guardrails | `input_guardrails.py` | Validate & sanitize user input |
| Output Guardrails | `output_guardrails.py` | Validate AI responses |
| Content Filter | `content_filter.py` | Topic enforcement & safety |
| Guardrails Manager | `guardrails_manager.py` | Unified interface |

### Input Protection

| Check | Description | Action |
|-------|-------------|--------|
| Prompt Injection | Detects manipulation attempts | 🚫 Block |
| Off-Topic | Non-math queries (geography, history, etc.) | 🚫 Block |
| Harmful Content | Violence, weapons, drugs | 🚫 Block |
| Profanity | Inappropriate language | 🚫 Block |
| PII Detection | Personal information | ⚠️ Warning |
| Length Validation | Too short/long inputs | 🚫 Block |

### Output Protection

| Check | Description | Action |
|-------|-------------|--------|
| Hallucination Detection | False citations/sources | ⚠️ Warning + Remove |
| Completeness Check | Missing solution steps | ⚠️ Warning |
| Safety Filter | Harmful content in response | 🚫 Block |
| Coherence Check | Truncated/broken responses | ⚠️ Warning |

## Human-in-the-Loop (HITL)
| Input Type | Threshold |
| ---------- | --------- |
| OCR        | < 70%     |
| ASR        | < 65%     |


## 🌐 Deployment

Hugging Face Spaces

SDK: Streamlit

Add GROQ_API_KEY as secret

🤝 Contributing

Fork the repo

Create a branch

Commit changes

Open a Pull Request

## 📄 License

MIT License © 2026

## 🙏 Acknowledgments

Groq – LLM inference

Whisper – Speech recognition

Tesseract – OCR

FAISS – Vector search

SymPy – Symbolic math

Streamlit – UI framework

