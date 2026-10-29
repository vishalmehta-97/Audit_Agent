"""Foundation checks: contract imports, config validation, mock retriever behavior."""
import pytest
from pydantic import ValidationError

from src.config import Settings, settings
from src.retriever_interface import BaseRegulatoryRetriever
from src.schemas import (AuditFinding, ComplianceStatus, DocumentGrade, EvidenceChunk,
                         RiskSeverity, SourceType)
from tests.mocks.mock_retriever import MockRegulatoryRetriever


def _chunks():
    return [
        EvidenceChunk(id="c1", document_name="SYNTHETIC", page_number=1, text="Logs must be retained for 90 days."),
        EvidenceChunk(id="c2", document_name="SYNTHETIC", page_number=2, text="Data must be stored in region A. Logs also noted."),
        EvidenceChunk(id="c3", document_name="SYNTHETIC", text="Unrelated paragraph about parking."),
    ]


def test_defaults():
    s = Settings(_env_file=None)
    assert s.RETRIEVAL_TOP_K == 4 and s.MAX_RETRY_COUNT == 2
    assert s.CHUNK_SIZE == 800 and s.CHUNK_OVERLAP == 150 and s.RRF_K == 60
    assert settings is not None  # module-level settings import cleanly without keys


@pytest.mark.parametrize("kw", [{"RETRIEVAL_TOP_K": 0}, {"RETRIEVAL_TOP_K": 21}, {"RRF_K": 0},
                                {"MAX_RETRY_COUNT": 6}, {"MAX_RETRY_COUNT": -1},
                                {"CHUNK_SIZE": 100, "CHUNK_OVERLAP": 100}])
def test_config_validation(kw):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kw)


def test_enums_and_models():
    assert ComplianceStatus.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    assert RiskSeverity.INFORMATIONAL.value == "INFORMATIONAL"
    assert EvidenceChunk(id="x", document_name="d", text="t").source_type == SourceType.INTERNAL
    with pytest.raises(ValidationError):
        EvidenceChunk(id="x", document_name="d", text="")
    with pytest.raises(ValidationError):
        DocumentGrade(is_relevant=True, confidence_score=1.5, rationale="r")
    f = AuditFinding(policy_clause_id="P1", policy_text_excerpt="e", status=ComplianceStatus.COMPLIANT,
                     severity=RiskSeverity.LOW, gap_analysis="g", remediation_action="r")
    assert f.evidence_ids == []


def test_mock_is_a_retriever_and_ranks():
    r = MockRegulatoryRetriever(_chunks())
    assert isinstance(r, BaseRegulatoryRetriever)
    res = r.retrieve("logs retained 90 days", top_k=4)
    assert res[0].id == "c1"
    assert all(res[i].score >= res[i + 1].score for i in range(len(res) - 1))
    assert "c3" not in [c.id for c in res]


def test_mock_top_k_unique_empty_and_errors():
    r = MockRegulatoryRetriever(_chunks() + [_chunks()[0]])
    assert len(r.retrieve("logs", top_k=1)) == 1
    ids = [c.id for c in r.retrieve("logs data region days", top_k=10)]
    assert len(ids) == len(set(ids))
    assert r.retrieve("zzzz nomatch") == []
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            r.retrieve(bad)
    assert len(r.calls) == 3  # failed calls are not recorded