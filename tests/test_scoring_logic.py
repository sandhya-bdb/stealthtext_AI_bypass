"""
test_scoring_logic.py — Tests for the AI score heuristic logic in detector.py.

These tests verify the scoring thresholds in analyze() without running the
real GPT-2 model — they mock calculate_perplexity() and calculate_burstiness()
to inject known values and confirm the heuristic score computation is correct.

Tests cover:
  - Very low ppl + low burstiness → high ai_score
  - Medium ppl → medium ai_score contribution
  - High ppl → low ai_score contribution
  - High burstiness → no burstiness penalty added
  - Score never exceeds 100
  - Score is always >= 0
  - All expected score combinations from the heuristic table
"""
import pytest
from unittest.mock import patch, MagicMock


def run_analyze_with(detector_instance, ppl: float, burstiness: float) -> dict:
    """
    Patch calculate_perplexity, calculate_burstiness, and _validate on the
    given detector instance to return fixed values, then call analyze().

    We mock _validate too so scoring tests are not coupled to the validation
    rules — they test only the heuristic math.
    """
    with patch.object(detector_instance, "_validate", return_value=None), \
         patch.object(detector_instance, "calculate_perplexity", return_value=ppl), \
         patch.object(detector_instance, "calculate_burstiness", return_value=burstiness):
        return detector_instance.analyze("any text")


# ---------------------------------------------------------------------------
# Scoring heuristic table (derived from detector.py lines 79–95):
#
#   ppl < 30  → +60     ppl < 50  → +40     else → +10
#   burst < 10 → +30
#
# Expected combos:
#   ppl=20, burst=5   → 60 + 30 = 90
#   ppl=20, burst=15  → 60 + 0  = 60
#   ppl=40, burst=5   → 40 + 30 = 70
#   ppl=40, burst=15  → 40 + 0  = 40
#   ppl=70, burst=5   → 10 + 30 = 40
#   ppl=70, burst=15  → 10 + 0  = 10
# ---------------------------------------------------------------------------

class TestScoringHeuristic:
    """Verify the exact score values produced by the scoring heuristic."""

    def test_very_low_ppl_low_burstiness_gives_90(self, detector):
        result = run_analyze_with(detector, ppl=20.0, burstiness=5.0)
        assert result["ai_score"] == 90, f"Expected 90, got {result['ai_score']}"

    def test_very_low_ppl_high_burstiness_gives_60(self, detector):
        result = run_analyze_with(detector, ppl=20.0, burstiness=15.0)
        assert result["ai_score"] == 60, f"Expected 60, got {result['ai_score']}"

    def test_medium_ppl_low_burstiness_gives_70(self, detector):
        result = run_analyze_with(detector, ppl=40.0, burstiness=5.0)
        assert result["ai_score"] == 70, f"Expected 70, got {result['ai_score']}"

    def test_medium_ppl_high_burstiness_gives_40(self, detector):
        result = run_analyze_with(detector, ppl=40.0, burstiness=15.0)
        assert result["ai_score"] == 40, f"Expected 40, got {result['ai_score']}"

    def test_high_ppl_low_burstiness_gives_40(self, detector):
        result = run_analyze_with(detector, ppl=70.0, burstiness=5.0)
        assert result["ai_score"] == 40, f"Expected 40, got {result['ai_score']}"

    def test_high_ppl_high_burstiness_gives_10(self, detector):
        result = run_analyze_with(detector, ppl=70.0, burstiness=15.0)
        assert result["ai_score"] == 10, f"Expected 10, got {result['ai_score']}"


class TestScoreBounds:
    """ai_score must always stay in [0, 100]."""

    def test_score_never_exceeds_100(self, detector):
        # Worst-case: both penalty branches fire → 60 + 30 = 90 (already < 100)
        # But if thresholds change in future, ensure cap holds
        result = run_analyze_with(detector, ppl=1.0, burstiness=0.0)
        assert result["ai_score"] <= 100

    def test_score_is_never_negative(self, detector):
        result = run_analyze_with(detector, ppl=500.0, burstiness=999.0)
        assert result["ai_score"] >= 0


class TestScoreRounding:
    """Perplexity and burstiness in the result dict should be rounded to 2dp."""

    def test_perplexity_rounded_to_2dp(self, detector):
        result = run_analyze_with(detector, ppl=45.6789, burstiness=5.0)
        assert result["perplexity"] == round(45.6789, 2)

    def test_burstiness_rounded_to_2dp(self, detector):
        result = run_analyze_with(detector, ppl=40.0, burstiness=12.3456)
        assert result["burstiness"] == round(12.3456, 2)


class TestVerdictThresholds:
    """Test the API-level verdict thresholds (ai_score > 50 → Likely AI)."""

    def test_score_above_50_is_likely_ai(self, detector):
        result = run_analyze_with(detector, ppl=20.0, burstiness=5.0)
        verdict = "Likely AI" if result["ai_score"] > 50 else "Likely Human"
        assert verdict == "Likely AI"

    def test_score_at_or_below_50_is_likely_human(self, detector):
        result = run_analyze_with(detector, ppl=70.0, burstiness=15.0)
        verdict = "Likely AI" if result["ai_score"] > 50 else "Likely Human"
        assert verdict == "Likely Human"
