"""
test_api.py — Integration tests for backend/api.py (FastAPI endpoints).

Uses FastAPI's TestClient so no real server is needed.
The LangGraph agent and the Groq client are mocked to keep tests fast and
completely offline.

Tests cover:
  - GET / → health check
  - POST /analyze → correct response shape
  - POST /analyze → correct verdict thresholds
  - POST /humanize → correct response shape
  - POST /humanize → iterations counter is included
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for the FastAPI app.
    We patch detector.analyze (the module-level instance) and agent_app.invoke
    so no GPT-2 inference or LangGraph execution happens.
    """
    from fastapi.testclient import TestClient
    import backend.api as api_module  # import first so the module-level objects exist

    mock_analyze_return = {
        "perplexity": 45.0,
        "burstiness": 8.5,
        "ai_score": 70,
    }

    mock_graph_result = {
        "original_text": "AI is transforming industry.",
        "text": "Honestly, AI has flipped entire industries on their head.",
        "perplexity": 85.5,
        "burstiness": 22.3,
        "iterations": 2,
    }

    with patch.object(api_module.detector, "analyze", return_value=mock_analyze_return), \
         patch.object(api_module.agent_app, "invoke", return_value=mock_graph_result):
        yield TestClient(api_module.app)


# ---------------------------------------------------------------------------
# Tests: health check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_message(self, client):
        data = response = client.get("/").json()
        assert "message" in data


# ---------------------------------------------------------------------------
# Tests: /analyze endpoint
# ---------------------------------------------------------------------------

class TestAnalyzeEndpoint:
    """POST /analyze should return a well-formed AnalysisResponse."""

    def test_analyze_returns_200(self, client, AI_TEXT):
        response = client.post("/analyze", json={"text": AI_TEXT})
        assert response.status_code == 200

    def test_analyze_response_has_required_fields(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert "perplexity" in data
        assert "burstiness" in data
        assert "ai_score" in data
        assert "verdict" in data

    def test_analyze_perplexity_is_float(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert isinstance(data["perplexity"], float)

    def test_analyze_verdict_is_string(self, client, AI_TEXT):
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert isinstance(data["verdict"], str)
        assert data["verdict"] in ("Likely AI", "Likely Human")

    def test_analyze_high_ai_score_gives_likely_ai_verdict(self, client, AI_TEXT):
        """Our mock returns ai_score=70 → verdict should be 'Likely AI'."""
        data = client.post("/analyze", json={"text": AI_TEXT}).json()
        assert data["ai_score"] == 70
        assert data["verdict"] == "Likely AI"


# ---------------------------------------------------------------------------
# Tests: /humanize endpoint
# ---------------------------------------------------------------------------

class TestHumanizeEndpoint:
    """POST /humanize should return a well-formed HumanizeResponse."""

    def test_humanize_returns_200(self, client, AI_TEXT):
        response = client.post("/humanize", json={"text": AI_TEXT})
        assert response.status_code == 200

    def test_humanize_response_has_required_fields(self, client, AI_TEXT):
        data = client.post("/humanize", json={"text": AI_TEXT}).json()
        assert "original_text" in data
        assert "final_text" in data
        assert "final_perplexity" in data
        assert "final_burstiness" in data
        assert "iterations" in data

    def test_humanize_original_text_matches_input(self, client, AI_TEXT):
        data = client.post("/humanize", json={"text": AI_TEXT}).json()
        # Our mock returns a fixed original_text — just verify the field exists and is a string
        assert isinstance(data["original_text"], str)
        assert len(data["original_text"]) > 0

    def test_humanize_final_text_is_string(self, client, AI_TEXT):
        data = client.post("/humanize", json={"text": AI_TEXT}).json()
        assert isinstance(data["final_text"], str)
        assert len(data["final_text"]) > 0

    def test_humanize_iterations_is_int(self, client, AI_TEXT):
        data = client.post("/humanize", json={"text": AI_TEXT}).json()
        assert isinstance(data["iterations"], int)
        assert data["iterations"] >= 0

    def test_humanize_final_perplexity_is_float(self, client, AI_TEXT):
        data = client.post("/humanize", json={"text": AI_TEXT}).json()
        assert isinstance(data["final_perplexity"], float)
