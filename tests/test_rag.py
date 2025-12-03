from app.models.rag import RAGGuidance


def test_rag_returns_guidance_for_known_category():
    rag = RAGGuidance()

    summary = "Strong shaking, people evacuating buildings in the city center."
    guidance = rag.generate_guidance(summary, category="earthquake", severity="high")

    assert isinstance(guidance, str)
    assert guidance.strip()  # non-empty text
    # should at least mention severity or type in a generic way
    assert "severity" in guidance.lower() or "earthquake" in guidance.lower()


def test_rag_handles_unknown_category_with_fallback_guidance():
    rag = RAGGuidance()

    summary = "Something strange is happening, not sure what type of incident this is."
    guidance = rag.generate_guidance(summary, category="some_unknown_category", severity="high")

    assert isinstance(guidance, str)
    assert guidance.strip()
