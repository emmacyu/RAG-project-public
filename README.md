# 🛡️ Risk Auditor Agent (AIXel Case Assignment)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C5A?style=for-the-badge)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](#)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)](#)
[![LangSmith](https://img.shields.io/badge/LangSmith-1C3C5A?style=for-the-badge)](#)

A smart assistant that helps our risk officers understand and "talk" to complex risk management documents.

## 🌟 Core Technologies & Features

1. **Agentic Graph (LangGraph)**
   The core reasoning engine operates as a finite state machine rather than a linear chain. It utilizes a `PostgresSaver` checkpointer tied to user session IDs (`thread_id`), allowing true persistent conversational memory across server restarts. Each new question starts with a clean retry state, while every earlier step stays in the checkpoint history.

2. **Structure-Aware Document Ingestion**
   PDFs are cleaned (repeated page headers removed, hard line wraps joined back into paragraphs) and split along the document's own sections (`A.4 Key Terms > Model Owner`). Every chunk carries its heading path, its 1-based page or page range, and a deterministic ID, so citations point to the right page and re-uploading a file replaces its chunks instead of duplicating them. Documents without recognisable headings fall back to page-by-page chunking.

3. **Hybrid Retrieval with Reranking**
   Dense vectors (ChromaDB) find paraphrases; BM25 keyword search catches exact identifiers such as `B-10` or `E-21`. The two candidate lists are merged with Reciprocal Rank Fusion and reranked by a local cross-encoder, which reads the question and each passage together.

4. **Adaptive Context Size and Honest "Not Found"**
   The reranker scores decide how many passages reach the LLM: a narrow fact gets a few, a question spanning several sections gets more (2 to 8). When even the best passage is irrelevant, the agent answers "Cannot find related content in the database." directly, without calling the LLM.

5. **Contextual Query Rewrite**
   A secondary LLM node evaluates chat history, resolves pronouns (e.g. "what does _it_ mean?"), and rewrites follow-up questions into standalone search queries before retrieval.

6. **Self-Correction Hallucination Loop (Logic Auditor)**
   A dedicated "Judge Node" checks every risk-control answer against the retrieved context before it is returned. If it finds unsupported claims, its feedback rewrites the search and the answer is regenerated, up to three attempts in total (a circuit breaker prevents infinite loops).

7. **Decoupled Microservice Architecture**
   Embeddings (`all-MiniLM-L6-v2`) and the reranker (`ms-marco-MiniLM-L-6-v2`) run locally inside the backend container; vectors live in a standalone `ChromaDB` container and conversation state in a separate `PostgreSQL` instance.

8. **Premium Cyberpunk Glassmorphism UI**
   A dual-pane web application built on Vite + React + TailwindCSS v4 with document upload and chat interface.

## 🗂️ Repository Structure

```text
.
├── backend                          # Python backend service (FastAPI)
│   ├── app
│   │   ├── main.py                  # FastAPI entrypoint & HTTP routes
│   │   ├── config.py                # Pydantic environment configuration (incl. retrieval settings)
│   │   ├── prompts.py               # Core prompt templates for LLMs, separated for later prompt engineering
│   │   └── services
│   │       ├── agent.py             # LangGraph state machine & AI logic
│   │       ├── chunking.py          # PDF cleaning and structure-aware chunking
│   │       ├── retrieval.py         # BM25 + dense hybrid search, RRF, cross-encoder rerank, adaptive cutoff
│   │       ├── database.py          # PostgreSQL checkpointer
│   │       └── vector_store.py      # ChromaDB ingestion and the retriever factory
│   ├── Dockerfile                   # Backend container definition
│   └── tests
│       ├── test_chunking.py         # Offline: cleaning, headings, sections, page numbers, re-upload
│       ├── test_retrieval.py        # Offline: BM25, fusion, reranking, cutoff, "not found" shortcut
│       ├── test_agent_flow.py       # Offline: multi-turn graph runs with a scripted LLM
│       ├── test_upload.py           # Offline: uploads do not block other requests
│       ├── test_api.py              # Live: /chat endpoint
│       ├── test_eval_robot.py       # Live: intent routing and red-team hallucination test
│       ├── data/                    # Sample guideline used by the tests (OSFI E-23)
│       ├── retrieval_eval/          # Retrieval evaluation set, scripts and results
│       └── e2e_eval/                # End-to-end evaluation of the full agent
├── frontend                         # React web application (Vite)
│   ├── src
│   │   ├── App.jsx                  # Main UI interface
│   │   ├── App.css                  # styling
│   │   └── index.css                # styling
│   ├── Dockerfile                   # Frontend container definition
│   └── nginx.conf                   # NGINX configuration
└── docker-compose.yml               # Orchestrates the entire stack (Chroma, Postgres, Redis, etc.)
```

## ⚙️ AI Agent Workflow (LangGraph)

The internal reasoning engine leverages cyclical state machines to achieve self-reflection and hallucination prevention.

```mermaid
---
config:
  flowchart:
    curve: basis
---
graph LR;
    subgraph Core Audit Loop
        retrieve_node(🍁<br>Retriever)
        generate_node(⚙️<br>Generator)
        auditor_node(⚖️<br>Auditor)
    end
    
    __start__([START]):::first
    dispatcher(🧠<br>Dispatcher)
    __end__([END]):::last

    __start__ --> dispatcher;
    dispatcher -.Casual<br>Chat.-> generate_node;
    dispatcher -.Audit<br>Intent.-> retrieve_node;
    
    retrieve_node --> generate_node;
    generate_node --> auditor_node;
    
    auditor_node -.Hallucinate<br>under 3 attempts.-> retrieve_node;
    auditor_node -.Pass.-> __end__;



    classDef default fill:#f2f0ff,line-height:1.2,stroke:#bfb6fc,stroke-width:2px
    classDef first fill-opacity:0,stroke:#888
    classDef last fill:#bfb6fc
```

### Inside the Retriever

```mermaid
graph LR;
    q([Rewritten query]) --> dense(Dense top-30<br>ChromaDB);
    q --> bm25(BM25 top-30<br>keywords);
    dense --> rrf(Reciprocal<br>Rank Fusion);
    bm25 --> rrf;
    rrf --> rerank(Cross-encoder<br>rerank);
    rerank --> gate{Best score<br>≥ threshold?};
    gate -.Too low.-> nf([Cannot find related content<br>no LLM call]);
    gate -.Relevant.-> keep(Keep passages close<br>to the best: 2 to 8);
    keep --> gen([Generator]);

    classDef default fill:#f2f0ff,line-height:1.2,stroke:#bfb6fc,stroke-width:2px
```

## 🚀 Quick Start

The entire application stack (Frontend, Backend, Postgres, ChromaDB, Redis) is completely containerized. You do not need to install Node or Python locally.

### 1. Configure Environment
Create a `.env` file in the root directory and add your OpenRouter API key:
```bash
OPENROUTER_API_KEY="your-api-key"
LANGCHAIN_API_KEY="your-api-key" # optional
```

### 2. Start the Stack 
**Note**: do not forget to start the docker daemon in your local machine
```bash
# Spins up the entire microservice ecosystem in the background
docker compose up -d --build
```

### 3. Access the Application
- **Frontend UI**: `http://localhost:5173`
- **Backend API Docs**: `http://localhost:8000/docs`
- **PGAdmin (DB Viewer)**: `http://localhost:8080`
- **LangSmith**: `https://smith.langchain.com/`

The first upload and the first question download the embedding and reranker models (about 90 MB each), so they take longer than later ones.

![Risk Auditor Agent UI Showcase](assets/ui_showcase.png)

### 4. Retrieval Settings (optional)
All have defaults in `backend/app/config.py` and can be overridden in `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking; empty disables reranking |
| `USE_BM25` | `true` | Add keyword candidates to the dense ones |
| `RETRIEVAL_CANDIDATES` | `30` | Candidates taken from each of dense and BM25 |
| `RETRIEVAL_TOP_K` | `8` | Most passages passed to the LLM |
| `ADAPTIVE_TOP_K` | `true` | Let reranker scores decide the number of passages |
| `RERANK_MIN_SCORE` | `-6.0` | Below this best score the answer is "not found" |
| `RERANK_MARGIN` | `6.0` | Keep passages scoring within this of the best one |
| `RETRIEVAL_MIN_K` | `2` | Fewest passages once the best one passes the threshold |

The score thresholds are on the reranker's own scale and were calibrated for `ms-marco-MiniLM-L-6-v2`; re-run `tests/retrieval_eval/run_cutoff_eval.py` after changing the reranker.

## 🧪 Tests & Evaluation

### Automated tests
CI (GitHub Actions) builds the full stack on every push and runs two groups of tests inside the backend container:
- **Offline tests** (no LLM calls): chunking, retrieval, multi-turn graph behaviour with a scripted LLM, and non-blocking uploads.
- **Live tests**: the `/chat` endpoint and the "robot" evaluation of intent routing and hallucination defence, which call the real model.

```bash
docker exec risk_auditor_backend pytest tests/test_chunking.py tests/test_retrieval.py tests/test_agent_flow.py tests/test_upload.py -v
```

### Retrieval evaluation
`tests/retrieval_eval/` holds a hand-written evaluation set on the OSFI E-23 guideline: 36 single-fact questions, 6 questions whose answer spans several sections, and 14 questions the guideline does not answer. Each question is marked with the evidence passage and page it should find.

| Retrieval (on structure-aware chunks) | Answer ranked 1st | Answer in top 5 | MRR@15 | Time per query (CPU) |
|---|---|---|---|---|
| Dense only (previous setup) | 47% | 83% | 0.63 | ~25 ms |
| Hybrid BM25 + dense (RRF) | 53% | 94% | 0.68 | - |
| **Hybrid + MiniLM reranker (default)** | **75%** | **100%** | **0.86** | ~200 ms |
| Hybrid + bge-reranker-base | 83% | 100% | 0.90 | ~1.5 s |

| Passages passed to the LLM | Fixed 5 | Adaptive cutoff (default) |
|---|---|---|
| Single-fact answers found | 100% | 100% |
| Average passages, single-fact | 5 | 4.5 |
| Evidence covered, multi-section | 48% | 65% |
| Unanswerable questions refused before the LLM | 0 / 14 | 7 / 14 |
| Answerable questions wrongly refused | 0 | 0 |

The chunking changes also fixed citations: the original loader numbered pages from 0, so every cited page was off by one (0 of 34 correct before, 36 of 36 after).

These figures come from one document and a small question set, so they show direction rather than precise effect sizes.

```bash
cd backend
python tests/retrieval_eval/run_eval.py              # chunking strategies
python tests/retrieval_eval/run_retrieval_eval.py    # retrieval methods and rerankers
python tests/retrieval_eval/run_cutoff_eval.py       # calibrate the adaptive cutoff
```

### End-to-end evaluation
`tests/e2e_eval/run_e2e_eval.py` runs every evaluation question, plus two-turn follow-up conversations, through the whole agent and reports refusals, citation accuracy, LLM calls and latency; an optional LLM judge grades correctness and faithfulness to the retrieved context. `--dry-run` replaces the LLM with canned replies to check the harness and estimate token volume without cost.

## 📊 Observability & Telemetry (LangSmith)

To enable zero-code enterprise-grade tracing (Token usage, Latency tracking, and Node trace trees), this repository has native support for LangSmith telemetry:
1. Create a free account at [Langchain Smith](https://smith.langchain.com/) and generate an API key.
2. Edit your local `.env` file to include the telemetry flags:
```bash
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="lsv2_pt_your_api_key_here..."
```
3. Restart the background stack with `docker compose up -d`. All AI reasoning loops will now stream live to your cloud dashboard.

![LangSmith Telemetry Trace](assets/langsmith_trace.png)

## 🏢 FAQ (Architecture Decisions)

### 1. What are the key features in the system?
- **Self-Reflective AI-as-a-Judge**: A native Maker-Checker framework where an isolated `auditor_node` checks every generated draft against the retrieved context and sends unsupported drafts back for another attempt.
- **Precise, Grounded Retrieval**: Structure-aware chunks, hybrid keyword + vector search and cross-encoder reranking put the right passage first in 75% of evaluation questions and in the top 3 in all of them, with page citations that match the source document.
- **Honest "I Don't Know"**: When nothing relevant is retrieved, the agent says so directly instead of asking the LLM to answer from nothing.
- **Adaptive Intent Routing**: Automatically categorizes user intents to bypass RAG for casual chats, reducing wasted compute for non-domain interactions.
- **Advanced Contextual Query Expansion**: Features an active "Pronoun Resolution" module that rewrites user follow-up questions intelligently using previous memory state before hitting the RAG layer.
- **Resilient Failover & Caching**: Embedded Langchain routing that seamlessly falls back to a backup LLM (e.g. Claude) upon primary LLM (e.g. GPT-4o-mini) failure without bubbling errors (i.e., Http 500 or 429 errors).

### 2. What are the major components in the system?
- **The Agentic Orchestration Brain (LangGraph)**: The cyclic state machine routing logic among Dispatcher, Retriever, Generator, and Auditor nodes.
- **The API Gateway (FastAPI)**: The asynchronous web backbone carrying the persistent server lifespan, LLM caches, and backend HTTP endpoints.
- **The Retrieval Engine (ChromaDB + BM25 + Cross-Encoder)**: Dense vectors in a containerized ChromaDB, an in-memory BM25 keyword index, and a local reranker, all without external API calls.
- **The State Persistence Plane (PostgreSQL)**: The transactional ACID safety-net tracking deep conversation trees and checkpoints.
- **The CI/CD Safeguard (Pytest + Github Actions)**: Offline unit tests for ingestion, retrieval and graph behaviour, plus an automated "Red-Teaming Robot" that tries to provoke fabricated answers on every push to prevent regressed LLM behavior.

### 3. Why did you choose this specific way to handle memory?
Memory is managed by **LangGraph's AsyncPostgresSaver**. The system relies on this over simple in-memory arrays for 3 critical reasons:
1. **Scalability**: Permits scaling FastAPI horizontally across instances without "amnesia" since states exist on PG.
2. **Prerequisite for HITL**: Freezing an execution loop and waking it up hours later based solely on the tuple `(thread_id, state_checkpoint)` stored in Postgres is required for future interaction with asynchronous Human-In-The-Loop approvals. 
3. **Structured Audit Trails**: Beyond simple chat logs, Postgres natively preserves the 'inner cognitive state' of the agent (retry counts, flagged hallucination boolean flags) at every step, providing historical evidence for legal/compliance audits.

### 4. How did you ensure the system is fast?
- **Asynchronous I/O Concurrency**: FastAPI serves requests on a non-blocking event loop; document parsing and embedding run in a worker thread, so an upload does not hold up other users' questions.
- **Routing Bypasses**: Casual inputs skip retrieval entirely, and questions with no relevant passage skip both the generation and the audit LLM calls.
- **Local Retrieval Stack**: Embeddings, BM25 and the reranker run inside the backend container: a dense search takes about 25 ms and the full hybrid search with reranking about 200 ms on CPU, with no external API round trip.
- **Smaller Prompts**: Reranking lets the LLM receive 2 to 8 relevant passages instead of a fixed 15.

### 5. How did you ensure the system is stable?
- **High-Availability Fallbacks**: `with_fallbacks()` routes ensure model API downtime (429/500s) does virtually zero damage to application uptime.
- **Resilient Redis Caching (Thundering Herd Protection)**: Identical LLM requests are answered from an exact-match Redis cache, protecting provider rate limits under concurrent usage.
- **Circuit Breakers**: The audit loop allows at most three attempts per question (`retry_count < 3`), preventing infinitely cycling hallucination loops; the counter resets for every new question.
- **Idempotent Ingestion**: Deterministic chunk IDs make re-uploading a document safe: its previous chunks are replaced, never duplicated.

### 6. What would you do differently if you had a month instead of three days?
If granted a one-month roadmap, I would evolve this prototype into a complete Enterprise Swarm:
1. **GraphRAG and Query Decomposition**: Reranking is in place; next, break broad questions into sub-questions and add knowledge-graph links between sections, so answers spanning several sections reach full coverage (65% today).
2. **Optimized SSE Streaming**: Refactor `ainvoke()` endpoints into pure websockets / `astream_events()`, drastically slashing 'Time-to-First-Token' wait anxiety in the frontend.
3. **RLHF via HITL Logs**: Re-invest all human overrides and auditor rejection checkpoints into a continuous Direct Preference Optimization (DPO) pipeline, effectively creating a self-healing localized fine-tuned LLM policy.
4. **Multi-Agent Federation**: Dissolve the monolithic graph into explicitly disparate Swarm Agents (Legal Agent, Accounting Agent, Manager Agent) isolated via strict RBAC (Role-Based Access Controls).
5. **MCP Integration**: Integrate with MCP tools to enable the agent to interact with other tools and services when needed.
6. **Broader Evaluation**: Grow the evaluation set with questions written by risk officers and across several documents, run the end-to-end evaluation with an LLM judge on every release, and use the results to fine-tune and compare models.

## 📄 License

See [LICENSE](LICENSE) for the full terms.
