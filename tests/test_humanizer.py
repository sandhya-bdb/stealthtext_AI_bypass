"""
test_humanizer.py — Unit tests for logic/humanizer.py (TextHumanizer class).

Uses monkeypatching / mocking to avoid real Groq API calls during CI.

Tests cover:
  - No-API-key graceful fallback (returns original text)
  - Successful rewrite returns a non-empty string
  - AI stop-words are replaced in post-processing
  - rewrite() on API failure falls back to original text
  - advanced_rewrite() delegates to rewrite()
"""
import re
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_groq_client():
    """Returns a mock Groq client whose completions.create() returns 'MOCKED_OUTPUT'."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "MOCKED_OUTPUT"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def humanizer_with_mock(mock_groq_client):
    """TextHumanizer instance with the Groq client replaced by a mock."""
    from logic.humanizer import TextHumanizer
    h = TextHumanizer.__new__(TextHumanizer)
    h.client = mock_groq_client
    h.model = "llama-3.3-70b-versatile"
    return h


@pytest.fixture
def humanizer_no_client():
    """TextHumanizer instance where client is None (simulates missing API key)."""
    from logic.humanizer import TextHumanizer
    h = TextHumanizer.__new__(TextHumanizer)
    h.client = None
    h.model = "llama-3.3-70b-versatile"
    return h


# ---------------------------------------------------------------------------
# Tests: no-API-key fallback
# ---------------------------------------------------------------------------

class TestNoClientFallback:
    """When client is None, rewrite() must return the original text unchanged."""

    def test_rewrite_returns_original_when_no_client(self, humanizer_no_client, AI_TEXT):
        result = humanizer_no_client.rewrite(AI_TEXT)
        assert result == AI_TEXT, "Should return original text when client is None"

    def test_advanced_rewrite_returns_original_when_no_client(self, humanizer_no_client, AI_TEXT):
        result = humanizer_no_client.advanced_rewrite(AI_TEXT)
        assert result == AI_TEXT


# ---------------------------------------------------------------------------
# Tests: successful rewrite
# ---------------------------------------------------------------------------

class TestSuccessfulRewrite:
    """When the Groq client works, rewrite() should return non-empty string."""

    def test_rewrite_returns_string(self, humanizer_with_mock, AI_TEXT):
        result = humanizer_with_mock.rewrite(AI_TEXT)
        assert isinstance(result, str)

    def test_rewrite_returns_non_empty(self, humanizer_with_mock, AI_TEXT):
        result = humanizer_with_mock.rewrite(AI_TEXT)
        assert len(result) > 0, "Rewrite output must not be empty"

    def test_advanced_rewrite_delegates_to_rewrite(self, humanizer_with_mock, AI_TEXT):
        """advanced_rewrite() is just an alias — should call the same Groq path."""
        r1 = humanizer_with_mock.rewrite(AI_TEXT)
        r2 = humanizer_with_mock.advanced_rewrite(AI_TEXT)
        assert r1 == r2

    def test_groq_api_called_once(self, humanizer_with_mock, AI_TEXT):
        humanizer_with_mock.rewrite(AI_TEXT)
        humanizer_with_mock.client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: AI stop-word post-processing
# ---------------------------------------------------------------------------

class TestStopWordReplacement:
    """Post-processing should strip known AI buzzwords from the output."""

    def _make_humanizer_returning(self, text: str):
        """Helper: humanizer whose Groq mock returns a specific string."""
        from logic.humanizer import TextHumanizer
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = text
        mock_client.chat.completions.create.return_value = mock_response

        h = TextHumanizer.__new__(TextHumanizer)
        h.client = mock_client
        h.model = "llama-3.3-70b-versatile"
        return h

    def test_delve_is_replaced(self):
        h = self._make_humanizer_returning("We need to delve into this topic.")
        result = h.rewrite("This is a sufficiently long input sentence for validation.")
        assert "delve" not in result.lower(), f"'delve' should be replaced, got: {result}"

    def test_utilize_is_replaced(self):
        h = self._make_humanizer_returning("Companies utilize AI to leverage their data.")
        result = h.rewrite("This is a sufficiently long input sentence for validation.")
        assert "utilize" not in result.lower(), f"'utilize' should be replaced"

    def test_leverage_is_replaced(self):
        h = self._make_humanizer_returning("We can leverage the power of AI.")
        result = h.rewrite("This is a sufficiently long input sentence for validation.")
        assert "leverage" not in result.lower(), f"'leverage' should be replaced"

    def test_multiple_stop_words_replaced(self):
        h = self._make_humanizer_returning(
            "Furthermore, it is crucial to utilize and leverage these pivotal tools."
        )
        result = h.rewrite("This is a sufficiently long input sentence for validation.")
        for word in ["furthermore", "crucial", "utilize", "leverage", "pivotal"]:
            assert word not in result.lower(), f"'{word}' should have been replaced in: {result}"


# ---------------------------------------------------------------------------
# Tests: API error fallback
# ---------------------------------------------------------------------------

class TestAPIErrorFallback:
    """If the Groq API raises an exception, rewrite() should return original text."""

    def test_api_exception_returns_original(self, AI_TEXT):
        from logic.humanizer import TextHumanizer
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")

        h = TextHumanizer.__new__(TextHumanizer)
        h.client = mock_client
        h.model = "llama-3.3-70b-versatile"

        result = h.rewrite(AI_TEXT)
        assert result == AI_TEXT, "Should fall back to original on API failure"
