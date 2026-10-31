# AuditAgent

A **compliance-support prototype** that audits **one policy clause at a time** against a regulatory corpus using
hybrid retrieval (ChromaDB + BM25 with Reciprocal Rank Fusion) and a bounded LangGraph workflow
(retrieve, grade, rewrite, web fallback, audit, verify).

> **Not legal advice.** Output is generated from retrieved text and must be reviewed by a qualified person.
> The sample rules in this repo are **synthetic test text, not real law**.
> Only single-clause input is implemented. **Whole-document auditing is not implemented.**

Trust rules: the internal corpus is the primary source; web results are secondary, untrusted, and keep their URLs;
every citation and quote is checked by a deterministic verifier, and anything unsupported fails closed to
`INSUFFICIENT_EVIDENCE`. The frozen interface contract is in `docs/AUDITAGENT_SHARED_CONTRACT.md`.

## Flow

```
START -> retrieve -> grade --relevant--> audit -> verify -> END
                       |
                       +-- irrelevant, retry_count < MAX_RETRY_COUNT --> rewrite -> retrieve -> grade
                       +-- irrelevant, retries used up, no fallback yet --> web_fallback -> grade
                       +-- irrelevant after fallback --> audit (INSUFFICIENT_EVIDENCE) -> verify -> END
```
With `MAX_RETRY_COUNT=2`: at most 2 rewrites and 3 local retrievals, plus one web-fallback attempt.

## Setup

Requirements: Python 3.11+ (developed on 3.13.7 on Windows; the contract says 3.11, see "Status").

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env              # Mac/Linux: cp .env.example .env   then edit .env
```
`requirements.txt` has minimum constraints only (not a lock file). Never commit `.env`, keys, `chroma_db/`,
or private documents.

## Run the tests (offline: no keys, no internet)

```
python -m pytest -q
```

## Demos

**1. Offline scripted demo (no keys).** `streamlit run app.py`, keep mode "Offline scripted demo", pick a sample
clause, click **Run Audit**. Answers are pre-written for the synthetic samples: this demonstrates the UI, not a real audit.

**2. Offline evaluation.** `python evaluate.py` (add `--json out.json` to save). Uses a scripted fake model on six
synthetic cases; it checks pipeline wiring, **not** model quality or legal accuracy.

**3. Real pipeline on the synthetic corpus.**
```
python -m src.synthetic_corpus                      # writes data/synthetic_corpus/synthetic_rulebook.pdf
python -m src.ingestion data/synthetic_corpus       # builds the index in CHROMA_PERSIST_DIR (default ./chroma_db)
```
The first ingest or query downloads the embedding model (BAAI/bge-small-en-v1.5, roughly 130 MB), so it needs
internet once. Then put `GROQ_API_KEY=...` in `.env` (optional: `TAVILY_API_KEY` for web fallback), run
`streamlit run app.py` and choose **Live model**. If no index exists the app says so and uses the flagged
synthetic demo corpus instead.

**4. Your own PDFs.** `python -m src.ingestion path\to\file.pdf path\to\folder`
(`--replace` replaces the whole index; `--persist-dir DIR` picks another folder).
Re-ingesting unchanged files is a no-op. If the index is reported stale or partial: `python -m src.ingestion --rebuild`.
Corrupt or password-protected PDFs are reported and skipped, never silently ingested.

## Configuration (`.env`)

| Name | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | empty | needed only for live runs |
| `TAVILY_API_KEY` | empty | optional web fallback |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | verify it is still available before live use |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | index folder |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | changing it requires `--rebuild` |
| `RETRIEVAL_TOP_K` / `RRF_K` | 4 / 60 | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 150 | characters, not words |
| `MAX_RETRY_COUNT` | 2 | range 0 to 5 |

## Layout

```
app.py, evaluate.py            Streamlit UI, offline evaluation
src/schemas.py, config.py, retriever_interface.py     frozen shared contract
src/ingestion.py               PDF pages -> chunks (page/section provenance, stable ids) + CLI
src/retriever.py               tokenizer, RRF, BM25, HybridRetriever (persistent, recoverable)
src/nodes.py, graph.py         verifier, grader, rewriter, web fallback, auditor, LangGraph router
src/llm_clients.py             Groq + Tavily adapters (created lazily, never at import)
src/report.py                  UI view-model + safe error messages
src/synthetic_corpus.py        generates the synthetic demo PDF
tests/                         offline tests; fakes for LLM, search and embedder; MockRegulatoryRetriever
```

## Status: what is and is not verified

| Area | State |
|---|---|
| Offline test suite, mutation-style spot checks of key guards | passing (see the PR for the exact run) |
| Live Groq calls (JSON mode, model name) | **not verified** |
| Live Tavily search | **not verified** (API signature was checked offline only) |
| Real FastEmbed model download and embeddings | **not verified** (tests use a fake embedder) |
| Python 3.11 | **not verified** (developed on 3.13.7) |
| Dependency lock file | **not created yet** |
| Streamlit UI | headless `AppTest` only; check the layout in a browser |

## Troubleshooting

* **"An Application Control policy has blocked this file" when importing numpy (Windows).** Windows (Smart App
  Control or a managed policy) blocked a compiled file. Try `Unblock-File` on the venv's numpy files, or
  `pip install --force-reinstall --no-cache-dir numpy`; on a managed machine, ask IT or use WSL2.
* **"No index found ... Index documents first".** Run the ingest command above.
* **"The embedding model could not be loaded".** The first run needs internet to download the model.
* **Run git and pytest from the project folder** (the one containing `src/` and `tests/`).