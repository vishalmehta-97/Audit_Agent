"""Frozen shared schemas (AUDITAGENT_SHARED_CONTRACT). Do not change without both developers' approval."""
from enum import Enum
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


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