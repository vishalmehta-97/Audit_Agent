# AuditAgent: handoff

Compliance-support prototype. Audits ONE policy clause at a time. Not legal advice. Sample rules are synthetic.

## What was built

| Area | Files |
|---|---|
| Frozen contract | `src/schemas.py`, `src/config.py`, `src/retriever_interface.py`, `docs/AUDITAGENT_SHARED_CONTRACT.md` |
| Ingestion | `src/ingestion.py` (PDF to chunks, CLI `python -m src.ingestion`), `src/synthetic_corpus.py` |
| Retrieval | `src/retriever.py` (tokenizer, RRF, BM25, `HybridRetriever`: ChromaDB dense + BM25 sparse) |
| Agent | `src/nodes.py` (verifier, retriever, grader, rewriter, web fallback, auditor), `src/graph.py`, `src/llm_clients.py` |
| App | `app.py`, `src/report.py`, `evaluate.py` |
| Tests | `tests/` (offline; fakes for LLM, search, embedder; `MockRegulatoryRetriever`) |

## Graph routing and retry counts

`START -> retrieve -> grade`, then: relevant -> `audit`; irrelevant and `retry_count < MAX_RETRY_COUNT` and no fallback yet
-> `rewrite -> retrieve -> grade`; irrelevant with retries used up and no fallback yet -> `web_fallback -> grade`;
irrelevant after fallback -> `audit` (INSUFFICIENT_EVIDENCE); then `verify -> END`.
With `MAX_RETRY_COUNT=2`: at most 2 rewrites, 3 local retrievals, 1 web-fallback attempt (always marked as attempted,
even with no key or a failed search). Routing reads `document_grade.is_relevant`, never `is_grounded` or the document count.
These counts are asserted exactly in `tests/test_graph.py` for limits 0 to 3.

## Decisions beyond the written contract (please review)

* `HybridRetriever` returns the nearest neighbours even when a query shares no words with the corpus (dense search
  has no "no match"); the grader decides relevance. A never-indexed retriever raises `RetrievalIndexError`.
* Verifier: a COMPLIANT / NON_COMPLIANT / PARTIALLY_COMPLIANT finding must cite at least one evidence id; an
  INSUFFICIENT_EVIDENCE finding with no citations passes with `is_grounded=True` (it makes no unsupported claim), so the UI
  keys its "insufficient" warning on the status, not on `is_grounded`. The clause identifier is matched case-insensitively;
  quotes are case-sensitive; only whitespace and typographic differences are normalized.
* Index write order: embed, corpus JSON, Chroma, manifest last. A crash in between is detected on next use and fixed with
  `python -m src.ingestion --rebuild`. No pickle files.
* Ingestion CLI: bad files are reported (exit code 1) while good files are still indexed; `--replace` is refused if any file failed.
* Default model is `openai/gpt-oss-120b` (`llama-3.3-70b-versatile` returned 404 on the author's account). Groq's catalog changes.
* The grader prompt was reworded after live runs showed it demanding proof of the organization's implementation (see below).

## What was actually run, and what happened

Offline (no keys, no internet):
* Linux, Python 3.12.3 (development sandbox): **204 passed**.
* Author's machine, Windows, Python 3.13.7: **198 passed**, later **203 passed** (screenshots), and the final suite **204 passed** (reported by the author).
* Key guards were spot-checked by deliberately breaking the code and confirming tests fail (routing, verifier, RRF, tokenizer,
  index consistency, ingestion ids and page numbers, CLI safety, error messages).

Live, on the author's machine (Windows, Python 3.13.7):
* Real FastEmbed model download (about 67 MB) and ingestion of the 4-page synthetic PDF: index of 4 chunks.
* Live audit in the UI with Groq `openai/gpt-oss-120b`: NON_COMPLIANT / HIGH, quote verified, cited chunk from the real index.
* Other four sample clauses in the UI: author reported results as expected, no errors (not captured in detail).
* `python evaluate.py --live` (5 synthetic cases): first run 4/5, grounding 100%, average 7.5 s; the "compliant" case
  came back INSUFFICIENT_EVIDENCE. Three repeat runs of that case: 1 INSUFFICIENT, 2 COMPLIANT, each needing rewrites
  because the grader rejected the evidence 2 to 4 times, once saying the evidence did not "prove" implementation.
* After the grader prompt rewording: compliant case `retrieve > grade > audit > verify` (no rewrites, 2.5 s); full live
  evaluation **5/5, grounding 100%, average 4.1 s, 0 errors**; out-of-scope still ended INSUFFICIENT_EVIDENCE.
  This is a small synthetic sample, not a measure of accuracy on real law.

## Assumptions about provider APIs

* Groq: `chat.completions.create(model, messages, temperature=0, response_format={"type": "json_object"})`; structured
  output is parsed and validated with Pydantic. Worked live with `openai/gpt-oss-120b`. Other models are untested.
* Tavily: `TavilyClient(api_key).search(query=..., max_results=...)` returning `results` with title, url, content.
  Checked against tavily-python 0.8.5 offline only. **Live web fallback has not been run.**
* FastEmbed `TextEmbedding` with `passage_embed` / `query_embed`; ChromaDB 1.5.9 persistent client, cosine space, telemetry off.

## Known limitations

* Single clause only; no whole-document auditing. No OCR (scanned PDFs return nothing and log a warning).
* Only the synthetic corpus has been ingested. No evaluation on real regulations.
* Live model output varies between runs. Bounded retries and the fail-closed verifier still apply.
* Live mode sends the clause and retrieved passages to Groq (and the query to Tavily if a key is set). The app has no login:
  run it with `streamlit run app.py --server.address localhost`.
* Not tested on Python 3.11 or a second machine. `constraints.txt` was captured on Windows with Python 3.13.7.
* The integration tests use a 4-chunk corpus, so they do not prove ranking quality; the retriever unit tests do.

## Integration steps

1. Merge `feat/data-retrieval` into `main` through a pull request (the agent-layer branch was merged earlier).
2. `git checkout main`, `git pull origin main`, activate the venv, `pip install -r requirements.txt -c constraints.txt`.
3. `python -m pytest -q` (all offline), then `python -m src.synthetic_corpus` and `python -m src.ingestion data/synthetic_corpus`.
4. Put `GROQ_API_KEY` (and optionally `TAVILY_API_KEY`) in `.env`, then `streamlit run app.py --server.address localhost`.
5. For your own regulations: `python -m src.ingestion path\to\file.pdf` (add `--replace` to drop the synthetic rules).