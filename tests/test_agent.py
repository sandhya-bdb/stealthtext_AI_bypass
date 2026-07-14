"""
test_agent.py — Unit tests for agent/graph.py (LangGraph state machine logic).

We test the pure Python logic (should_continue, detector_node, rewriter_node)
WITHOUT running the full compiled LangGraph — this keeps tests fast and
avoids GPU/API calls.

Tests cover:
  - should_continue() routing logic for all branches
  - detector_node() returns expected keys
  - rewriter_node() increments iteration counter and stores history
"""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests: should_continue routing
# ---------------------------------------------------------------------------

class TestShouldContinue:
    """should_continue() must route correctly based on state values."""

    def _make_state(self, iterations=0, perplexity=50.0, burstiness=10.0):
        return {
            "text": "some text",
            "original_text": "some text",
            "perplexity": perplexity,
            "burstiness": burstiness,
            "verdict": "Likely AI",
            "iterations": iterations,
            "history": [],
        }

    def test_first_iteration_forces_rewrite(self):
        """iterations==0 → always 'rewrite' regardless of scores."""
        from agent.graph import should_continue
        state = self._make_state(iterations=0, perplexity=999.0, burstiness=999.0)
        assert should_continue(state) == "rewrite"

    def test_max_iterations_ends_graph(self):
        """iterations >= 3 → 'end'."""
        from agent.graph import should_continue
        state = self._make_state(iterations=3)
        assert should_continue(state) == "end"

    def test_above_max_iterations_ends_graph(self):
        """iterations > 3 → 'end'."""
        from agent.graph import should_continue
        state = self._make_state(iterations=10)
        assert should_continue(state) == "end"

    def test_good_scores_end_graph(self):
        """ppl > 80 AND burstiness > 20 → 'end'."""
        from agent.graph import should_continue
        state = self._make_state(iterations=1, perplexity=85.0, burstiness=25.0)
        assert should_continue(state) == "end"

    def test_low_perplexity_continues_rewrite(self):
        """ppl <= 80 → continue rewriting even if burstiness is high."""
        from agent.graph import should_continue
        state = self._make_state(iterations=1, perplexity=60.0, burstiness=30.0)
        assert should_continue(state) == "rewrite"

    def test_low_burstiness_continues_rewrite(self):
        """ppl > 80 but burstiness <= 20 → still rewrite."""
        from agent.graph import should_continue
        state = self._make_state(iterations=1, perplexity=90.0, burstiness=15.0)
        assert should_continue(state) == "rewrite"

    def test_both_scores_borderline_low(self):
        """Exactly at threshold — ppl=80 is NOT > 80, so should rewrite."""
        from agent.graph import should_continue
        state = self._make_state(iterations=1, perplexity=80.0, burstiness=20.0)
        assert should_continue(state) == "rewrite"


# ---------------------------------------------------------------------------
# Tests: detector_node
# ---------------------------------------------------------------------------

class TestDetectorNode:
    """detector_node() should return a dict with the expected keys."""

    def test_detector_node_returns_expected_keys(self):
        from agent.graph import detector_node
        state = {
            "text": "AI is transforming industry leveraging advanced algorithms.",
            "original_text": "AI is transforming industry.",
            "perplexity": 0.0,
            "burstiness": 0.0,
            "verdict": "",
            "iterations": 0,
            "history": [],
        }
        result = detector_node(state)
        assert "perplexity" in result
        assert "burstiness" in result
        assert "verdict" in result
        assert "iterations" in result

    def test_detector_node_verdict_is_string(self):
        from agent.graph import detector_node
        state = {
            "text": "Some sample text for testing the detector node.",
            "original_text": "Some sample text.",
            "perplexity": 0.0,
            "burstiness": 0.0,
            "verdict": "",
            "iterations": 0,
            "history": [],
        }
        result = detector_node(state)
        assert isinstance(result["verdict"], str)
        assert result["verdict"] in ("Likely Human", "Likely AI")

    def test_detector_node_preserves_iterations(self):
        from agent.graph import detector_node
        state = {
            "text": "Testing that iteration count passes through.",
            "original_text": "original",
            "perplexity": 0.0,
            "burstiness": 0.0,
            "verdict": "",
            "iterations": 2,
            "history": [],
        }
        result = detector_node(state)
        assert result["iterations"] == 2


# ---------------------------------------------------------------------------
# Tests: rewriter_node
# ---------------------------------------------------------------------------

class TestRewriterNode:
    """rewriter_node() should increment iterations and update history.
    
    We patch humanizer.rewrite so these tests focus purely on state management
    (iteration count, history list) without touching Groq or validation.
    """

    # Use a text that passes _validate() (>= 10 chars) as a safe default
    _DEFAULT_TEXT = "This is the original text for testing purposes."

    def _make_state(self, text=None, iterations=0, history=None):
        t = text if text is not None else self._DEFAULT_TEXT
        return {
            "text": t,
            "original_text": t,
            "perplexity": 50.0,
            "burstiness": 10.0,
            "verdict": "Likely AI",
            "iterations": iterations,
            "history": history or [],
        }

    def test_rewriter_increments_iterations(self):
        from agent.graph import rewriter_node
        import agent.graph as ag
        with patch.object(ag.humanizer, "rewrite", return_value="mocked rewrite output"):
            state = self._make_state(iterations=1)
            result = rewriter_node(state)
        assert result["iterations"] == 2

    def test_rewriter_stores_old_text_in_history(self):
        from agent.graph import rewriter_node
        import agent.graph as ag
        with patch.object(ag.humanizer, "rewrite", return_value="mocked rewrite output"):
            state = self._make_state(history=[])
            result = rewriter_node(state)
        assert self._DEFAULT_TEXT in result["history"]

    def test_rewriter_accumulates_history(self):
        from agent.graph import rewriter_node
        import agent.graph as ag
        second_text = "Second piece of text for testing accumulation."
        with patch.object(ag.humanizer, "rewrite", return_value="mocked rewrite output"):
            state = self._make_state(
                text=second_text,
                history=[self._DEFAULT_TEXT]
            )
            result = rewriter_node(state)
        assert self._DEFAULT_TEXT in result["history"]
        assert second_text in result["history"]

    def test_rewriter_returns_text_key(self):
        from agent.graph import rewriter_node
        import agent.graph as ag
        with patch.object(ag.humanizer, "rewrite", return_value="mocked rewrite output"):
            state = self._make_state()
            result = rewriter_node(state)
        assert "text" in result
        assert isinstance(result["text"], str)
