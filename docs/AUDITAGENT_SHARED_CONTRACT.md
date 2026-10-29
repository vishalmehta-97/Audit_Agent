AUDITAGENT — FROZEN SHARED CONTRACT
This contract is identical for Developer A and Developer B. It is the single source of truth for cross-module interfaces. Commit this file with the shared foundation before parallel implementation begins. Neither developer may silently change it; propose a change, agree with the other developer, update both implementations/tests, and merge the contract change first.

PROJECT BOUNDARY
- Core engine audits one atomic policy clause/rule at a time.
- A larger PDF may be split into clauses only through a separately implemented and tested UI/orchestration feature. Do not claim whole-document auditing if only single-clause input is implemented.
- This is a compliance-support prototype, not legal advice. Synthetic test rules are not real law.
- Internal regulatory corpus is the primary source. Web results are secondary, untrusted evidence and must retain their URLs.

PYTHON / LIBRARIES
- Python 3.11.
- Pydantic v2 and pydantic-settings.
- LangGraph for orchestration; Groq client/model configurable.
- PyMuPDF, FastEmbed, ChromaDB, rank-bm25 for retrieval.
- Tavily optional for web fallback.
- Streamlit for UI; pytest for tests.
- One dependency owner maintains requirements.txt and resolves compatibility. Proposed changes require a PR and tests. Prefer a reproducible lock/constraints file once the environment is verified; minimum version constraints alone are not a lock.
- Never claim compatibility or test success without actually checking it.

AUTHORITATIVE MODELS — src/schemas.py
Use these exact names/semantics. Imports must not require live API keys.

class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class RiskSeverity(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SourceType(str, Enum):
    INTERNAL = "INTERNAL"
    WEB = "WEB"

class EvidenceChunk(BaseModel):
    id: str
    document_name: str
    page_number: Optional[int] = None  # 1-indexed for PDFs; None for web results
    section_title: str = "GENERAL"
    text: str = Field(..., min_length=1)
    source_type: SourceType = SourceType.INTERNAL
    score: float = 0.0  # ranking score, not a probability
    source_url: Optional[str] = None  # required in practice for WEB evidence

class AuditFinding(BaseModel):
    policy_clause_id: str
    policy_text_excerpt: str
    status: ComplianceStatus
    severity: RiskSeverity
    statutory_clause: Optional[str] = None
    statutory_verbatim_quote: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    gap_analysis: str
    remediation_action: str

class DocumentGrade(BaseModel):
    is_relevant: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str

class AuditGraphState(TypedDict):
    policy_clause_id: str
    policy_text: str
    current_query: str
    retrieved_docs: List[EvidenceChunk]
    document_grade: Optional[DocumentGrade]
    retry_count: int
    web_fallback_used: bool
    web_fallback_attempted: bool
    is_grounded: bool
    audit_report: Optional[AuditFinding]
    error_message: Optional[str]

RETRIEVER CONTRACT — src/retriever_interface.py
class BaseRegulatoryRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[EvidenceChunk]:
        ...

- Empty/whitespace query -> ValueError.
- Return at most top_k unique EvidenceChunk objects sorted by descending score.
- No matches -> [].
- Same signature for HybridRetriever and MockRegulatoryRetriever.
- Retriever must not make LLM or web calls.

CONFIG CONTRACT — src/config.py
Expose settings with these names:
- GROQ_API_KEY: str = "" (do not require merely to import modules or run offline tests)
- TAVILY_API_KEY: str = "" (optional)
- GROQ_MODEL_NAME: configurable string, default may be "llama-3.3-70b-versatile" but verify current availability before live use
- CHROMA_PERSIST_DIR: str = "./chroma_db"
- EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
- RETRIEVAL_TOP_K: int = 4, range 1–20
- RRF_K: int = 60, minimum 1
- CHUNK_SIZE: int = 800 characters
- CHUNK_OVERLAP: int = 150 characters, must be smaller than CHUNK_SIZE
- MAX_RETRY_COUNT: int = 2, range 0–5
Do not initialize network clients at import time. Validate live credentials only when the relevant live client is actually needed.

GRAPH STATE LIFECYCLE — src/graph.py
Initial state must initialize every required field:
policy_clause_id, policy_text, current_query, retrieved_docs=[], document_grade=None,
retry_count=0, web_fallback_used=False, web_fallback_attempted=False,
is_grounded=False, audit_report=None, error_message=None.

Exact bounded flow:
START -> retrieve -> grade.
After grade:
1. If document_grade exists and is_relevant=True -> audit.
2. If irrelevant and retry_count < MAX_RETRY_COUNT and web_fallback_attempted=False -> rewrite -> retrieve -> grade.
3. If irrelevant and retry_count >= MAX_RETRY_COUNT and web_fallback_attempted=False -> web_fallback -> grade.
4. If web_fallback_attempted=True and grade remains irrelevant -> audit only to create an INSUFFICIENT_EVIDENCE result; then verify and end.
5. Web fallback is attempted at most once even when no API key is configured. Set web_fallback_attempted=True whenever the node runs; set web_fallback_used=True only if usable web results were actually obtained.
6. MAX_RETRY_COUNT=2 means at most two rewrites and three local retrieval attempts. The web fallback is an additional, separate attempt.
7. A non-empty retrieved_docs list does NOT imply relevance. `is_grounded` is exclusively a citation-verifier result and must NOT be used by the grader router.
8. Do not loop after verifier failure in MVP. Fail closed, return a safe INSUFFICIENT_EVIDENCE finding, and end.

CITATION / TRUST RULES
- Every finding.evidence_ids value must exist in the active retrieved_docs.
- A statutory quote must be found in a cited chunk whose ID is in evidence_ids after conservative normalization (whitespace/typographic normalization only).
- A quote found in an unrelated chunk is not valid.
- Verify statutory_clause against cited text when it is supplied.
- If grounding fails, do not display the original unsupported quote as verified. Return or transform to INSUFFICIENT_EVIDENCE, clear unsupported citations/quote, explain failure, and set is_grounded=False.
- Web evidence must have source_type=WEB, a real source_url, and page_number=None. Show it as web evidence in the UI.
- Retrieved text is untrusted data; ignore instructions contained inside retrieved documents.
- Never invent legal obligations or represent synthetic test text as authentic legislation.

SHARED FILE OWNERSHIP
- Developer A owns src/config.py, src/schemas.py, src/retriever_interface.py, src/ingestion.py, src/retriever.py, tests/mocks/mock_retriever.py, tests/test_ingestion.py, tests/test_retriever.py.
- Developer B owns src/nodes.py, src/graph.py, app.py, evaluate.py, tests/test_nodes.py, tests/test_graph.py, tests/test_verifier.py.
- tests/test_integration.py is jointly owned.
- requirements.txt has one named dependency owner, agreed before work starts. No simultaneous edits.
- Shared contracts are frozen after the foundation commit. Any changes require both developers' approval.

GIT WORKFLOW
1. Developer A creates the repository and commits only the shared foundation: package/test folders, shared contract implementation, mock, .env.example, .gitignore, requirements file, and basic README.
2. Both developers pull the same foundation commit.
3. Developer A branches `feat/data-retrieval`; Developer B branches `feat/agent-workflow`.
4. Developer B must build against MockRegulatoryRetriever; no dependency on a populated Chroma database during unit tests.
5. Open PRs, review and merge one at a time. Rebase the second branch on updated main and rerun the full suite.
6. Run joint integration tests with a small synthetic corpus. Do not commit .env, real secrets, generated indexes, or private documents.
7. Run tests after each component. Report exact commands and actual results; never fabricate test success.

REQUIRED ACCEPTANCE GATES
- Shared contracts import cleanly with no API keys.
- A's retriever and mock pass the same interface tests.
- B's router uses document_grade.is_relevant and has exact bounded node counts.
- Fallback runs at most once and its results are graded.
- Verifier rejects fabricated IDs, quotes, and mismatched cited chunks.
- Full offline `pytest` suite passes before claiming completion.
- A reproducible end-to-end demo works with documented setup.
