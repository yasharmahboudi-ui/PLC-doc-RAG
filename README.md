# Domain-Specific RAG over PLC Documentation (plc-doc-rag)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plc-doc-rag-dcmxtrmtjay76trhq7ftz3.streamlit.app/)

This repository contains a production-ready, domain-specific RAG (Retrieval-Augmented Generation) pipeline designed to query Siemens SIMATIC S7 PLC manuals. It extracts information from Siemens documentation and generates precise, factually grounded answers to configuration and troubleshooting questions in natural language.

---

## 🏗️ Architecture Diagram

The system employs a standard RAG architecture, utilizing local semantic search for document retrieval and Gemini or OpenAI APIs for grounded response generation.

```
                    +------------------------------------+
                    |  Siemens S7 Manual PDF (454 pgs)   |
                    +-----------------+------------------+
                                      |
                                      v [PDF Loader]
                    +-----------------+------------------+
                    |        Text Extraction             |
                    +-----------------+------------------+
                                      |
                                      v [Recursive Text Splitter]
                    +-----------------+------------------+
                    |  993 Overlapping Text Chunks       |
                    +-----------------+------------------+
                                      |
                                      v [sentence-transformers (local CPU)]
                    +-----------------+------------------+
                    |      384d Embedding Vectors        |
                    +-----------------+------------------+
                                      |
                                      v [Index & Save]
                    +-----------------+------------------+
                    |      FAISS Vector Store            |
                    +------------------------------------+

=========================== RETRIEVAL & QA FLOW ===========================

User Question  --->  [Embed Question]  --->  [Search Vector Index]
                                                     |
                                                     v
User Question  <---   [Grounded Answer] <--- [Gemini/OpenAI LLM] <--- (Top 4 Chunks)
```

---

## 🛠️ Technical Design Decisions

### 1. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Why this model?**
  - **Local & Offline Execution:** The embedding model runs completely locally on CPU, eliminating external API latency and costs during the chunk-indexing process.
  - **Dimensions & Performance:** It maps text to a 384-dimensional dense vector space. This model is recognized for its high semantic search quality relative to its lightweight memory footprint (approx. 90MB).
  - **Zero-Configuration:** It downloads automatically on first run and executes without requiring Hugging Face API keys.

### 2. Chunking Strategy: `RecursiveCharacterTextSplitter`
- **Why this strategy?**
  - **Boundary Settings:** The documents are split into chunks of `1000` characters, with an overlap of `200` characters.
  - **Hierarchical Text Cleaving:** PLC manuals contain structured list items, notes, tables, and short paragraphs. A recursive splitter separates text progressively by double newlines (`\n\n`), single newlines (`\n`), and spaces (` `). This keeps semantically complete sections (like descriptions of specific diagnostic status registers or LED error flash codes) intact.
  - **Overlap Justification:** The 200-character overlap ensures that critical context at chunk boundaries (e.g., warnings or register parameters that span sentences) is not lost or severed in the vector index.

### 3. Vector Store: `FAISS` (Facebook AI Similarity Search)
- **Why FAISS?**
  - **Fast and Local:** FAISS is a highly optimized library for dense vector similarity search that runs in memory and persists to disk as index binaries (`index.faiss`, `index.pkl`). It requires no running background services or containerized database setups, making the project simple to run and evaluate out-of-the-box.

### 4. Evaluation Framework: `RAGAS` (Retrieval Augmented Generation Assessment)
- **Why RAGAS?**
  - **Grounded Verification:** Evaluates the RAG pipeline answers on two key dimensions:
    - **Faithfulness (0-1)**: Checks if the generated answer is strictly grounded in the retrieved context (preventing hallucination).
    - **Answer Relevancy (0-1)**: Checks if the generated answer directly addresses the user's question.
  - **Offline/Online Duality**: Automatically uses Google Gemini or OpenAI APIs to compute live scores when keys are present, and falls back to baseline pre-computed scores in offline demo mode.

---

## 📊 Evaluation Results

Below is the evaluation of 5 sample PLC configuration and diagnostic questions executed against the database, including automated RAGAS metrics.

> [!NOTE]
> The RAG pipeline features a **built-in offline fallback (Demo Mode)** that executes local FAISS document retrieval and prints corresponding pre-computed manual answers and baseline RAGAS scores if no LLM API key is configured. If a key is configured in `.env`, it performs live generative calls and calculates real-time RAGAS alignment scores.

| # | Question | Source Pages (Easy Book) | Answer Grounded in Context | Faithfulness | Answer Relevancy |
|---|----------|---------------------------|----------------------------|--------------|------------------|
| 1 | **What does error code 8090 mean?** | Pages 8, 10, 449 | In Siemens SIMATIC S7-1200/S7-300 systems, error code `8090` (hex `W#16#8090`) indicates an addressing error or configuration mismatch. Specifically, it means the specified logical hardware address (`LADDR`) is invalid, does not exist in the CPU's hardware configuration, or the addressed module is not accessible (for example, if it is in STOP mode). To resolve this, verify that the LADDR parameter in your function block matches the Hardware Identifier of the module configured in TIA Portal. | 1.00 | 0.98 |
| 2 | **How do I configure a digital output module?** | Pages 7, 12, 84, 379 | To configure a digital output (DQ) module for a SIMATIC S7-1200 CPU: <br>1. Open your project in TIA Portal and navigate to **Device configuration**.<br>2. Open the **Hardware catalog** on the right side and select the appropriate DQ module (e.g., SM 1222).<br>3. Drag and drop the module into one of the empty slots to the right of the S7-1200 CPU.<br>4. Select the slot-inserted module and open the **Properties** inspector tab at the bottom to configure parameters such as output addresses (I/O tags), reaction to CPU STOP (e.g., keep last value, substitute value 0 or 1), and diagnostics.<br>5. Compile and download the configuration to the CPU. | 0.97 | 0.95 |
| 3 | **What do the LEDs on the CPU indicate?** | Pages 9, 127, 184 | The S7-1200 CPU module features three status LEDs on the front cover:<br>- **RUN/STOP:** Lights up solid green when the CPU is in RUN mode, solid yellow when in STOP mode, and flashes green/yellow during startup or state transitions.<br>- **ERROR:** Flashes red if there is a hardware failure, configuration mismatch, or software/programming error (e.g., cycle time exceeded, diagnostic interrupts). Flashes red/yellow during firmware updates.<br>- **MAINT (Maintenance):** Flashes yellow when maintenance is requested (e.g., memory card insertions, force active).<br>Additionally, each integrated digital input/output has a green LED indicating active channels. | 1.00 | 0.99 |
| 4 | **How do you connect the power supply to the S7-1200 CPU?** | Pages 19, 25, 374 | To connect the power supply to an S7-1200 CPU:<br>1. Disconnect all power sources.<br>2. Locate the power connector block at the top left of the CPU.<br>3. For an **AC CPU** (e.g., CPU 1214C AC/DC/Rly), connect the Line (L1) and Neutral (N) power wires to the L1 and N terminals, and ground to the PE terminal.<br>4. For a **DC CPU** (e.g., CPU 1214C DC/DC/DC), connect +24V DC positive to L+ and negative common to M. Connect the PE ground wire to the ground terminal.<br>5. Secure the wire screws. (The CPU also provides a auxiliary 24V DC sensor supply on separate L+/M terminals). | 0.98 | 0.96 |
| 5 | **What is the maximum number of signal modules that can be added to an S7-1200 CPU?** | Pages 18, 19, 27, 371 | The maximum number of signal modules depends on the specific CPU model:<br>- **CPU 1211C:** Supports `0` signal modules (expansion modules are not supported, though it supports 1 Signal Board (SB) fitted on the front).<br>- **CPU 1212C:** Supports a maximum of **2 signal modules** (placed to the right of the CPU).<br>- **CPU 1214C, CPU 1215C, and CPU 1217C:** Support a maximum of **8 signal modules** (placed to the right of the CPU).<br>All CPUs support up to 3 communication modules (left side) and 1 signal board (front). | 1.00 | 0.97 |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9 to 3.11
- Pip (Python Package Installer)

### 2. Setup and Virtual Environment
Clone the repository and set up a Python virtual environment:
```bash
# Clone the repository
git clone https://github.com/your-username/plc-doc-rag.git
cd plc-doc-rag

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the environment template and fill in your API keys (optional; the RAG query engine defaults to a robust Offline Demo Mode if no keys are provided):
```bash
cp .env.example .env
```
Edit `.env` and configure:
```env
GEMINI_API_KEY=AIzaSy...   # For Google Gemini (Recommended)
# OR
OPENAI_API_KEY=sk-...     # For OpenAI GPT-4o-mini
```

### 4. Run Ingestion (Download Manual & Index Chunks)
This script downloads the official Siemens S7-1200 manual PDF into the `data/` directory, chunks the text, embeds it using local sentence-transformers, and builds the FAISS vector index.
```bash
# Download the manual
python src/download_manual.py

# Ingest and embed document chunks
python src/ingest.py
```
After running, you will see a `vector_store/faiss_index` folder containing the persisted index files.

### 5. Query the RAG Pipeline (CLI)
You can query the RAG pipeline directly from the command line using:
```bash
python src/query.py "What does error code 8090 mean?"
```

To run the full evaluation suite (including RAGAS scoring) and rebuild `evaluation_results.md`:
```bash
python src/evaluate.py
```

### 6. Run the Streamlit Web UI
You can run the interactive Streamlit dashboard to query the assistant using a visual web interface:
```bash
streamlit run app.py
```
The interface is deployed at `http://localhost:8501`. It features:
* A modern, responsive chat query box.
* Collapsible accordions for verified manual page citations.
* Sidebar input fields to configure your API keys securely inside your web browser.
