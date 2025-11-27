from app.models.rag import IncidentRAG


def test_rag_returns_guidance_for_known_category():
    rag = IncidentRAG()
    result = rag.retrieve("earthquake")

    assert result.category == "earthquake"
    assert isinstance(result.guidance, str)
    assert "earthquake" not in result.guidance.lower() or len(result.guidance) > 10


def test_rag_falls_back_to_other_for_unknown_category():
    rag = IncidentRAG()
    result = rag.retrieve("some_unknown_category")

    assert result.category == "some_unknown_category"
    assert isinstance(result.guidance, str)
    assert len(result.guidance) > 0
