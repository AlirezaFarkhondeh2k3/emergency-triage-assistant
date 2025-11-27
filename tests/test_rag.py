from app.models.rag import IncidentRAG


def test_rag_returns_guidance_for_known_category():
    rag = IncidentRAG()

    result = rag.retrieve(
        text="Strong shaking, people evacuating buildings in the city center.",
        category="earthquake",
    )

    # Should preserve the category we pass in
    assert result.category == "earthquake"
    # Should return some non empty guidance
    assert isinstance(result.guidance, str)
    assert len(result.guidance.strip()) > 0
    # Should have retrieved at least one document
    assert len(result.retrieved_docs) >= 1


def test_rag_handles_unknown_category_with_fallback_guidance():
    rag = IncidentRAG()

    result = rag.retrieve(
        text="Something strange is happening, not sure what type of incident this is.",
        category="some_unknown_category",
    )

    # Should preserve the unknown label we pass in
    assert result.category == "some_unknown_category"
    # Should still return some guidance (generic fallback is acceptable)
    assert isinstance(result.guidance, str)
    assert len(result.guidance.strip()) > 0
