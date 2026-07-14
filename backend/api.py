import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from agent.graph import app as agent_app
from logic.detector import AIDetector, MAX_CHARS as DETECTOR_MAX_CHARS
from logic.humanizer import MAX_CHARS as HUMANIZER_MAX_CHARS

# LangSmith: zero effect if LANGCHAIN_TRACING_V2 is not set.
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        def decorator(fn):
            return fn
        return decorator

logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="StealthText API",
    version="1.0.0",
    description="AI text humanizer — detect AI writing and rewrite it to sound human.",
)

# Initialize logic once at startup (GPT-2 is ~500 MB — don't reload per request)
detector = AIDetector()

# ── Shared constants ─────────────────────────────────────────────────────────
_MAX_CHARS  = min(DETECTOR_MAX_CHARS, HUMANIZER_MAX_CHARS)  # 8 000
_MIN_CHARS  = 10


# ── Pydantic models ──────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < _MIN_CHARS:
            raise ValueError(
                f"Text is too short ({len(stripped)} chars). "
                f"Provide at least {_MIN_CHARS} characters."
            )
        if len(stripped) > _MAX_CHARS:
            raise ValueError(
                f"Text is too long ({len(stripped)} chars). "
                f"Maximum is {_MAX_CHARS} characters (~2 000 words)."
            )
        return stripped  # return stripped so downstream never sees leading/trailing whitespace


class AnalysisResponse(BaseModel):
    perplexity: float
    burstiness: float
    ai_score: int
    verdict: str


class HumanizeResponse(BaseModel):
    original_text: str
    final_text: str
    final_perplexity: float
    final_burstiness: float
    iterations: int
    score_history: list[dict] = []


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "StealthText API is running", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest):
    """
    Score the supplied text with the GPT-2 detector.
    Returns perplexity, burstiness, an AI probability score (0–100), and a verdict.
    """
    logger.info("/analyze — %d chars", len(request.text))
    try:
        result  = detector.analyze(request.text)
        verdict = "Likely AI" if result["ai_score"] > 50 else "Likely Human"
        logger.info(
            "/analyze done — ppl=%.1f  burst=%.1f  score=%d  verdict=%s",
            result["perplexity"], result["burstiness"], result["ai_score"], verdict,
        )
        return {
            "perplexity": result["perplexity"],
            "burstiness": result["burstiness"],
            "ai_score":   result["ai_score"],
            "verdict":    verdict,
        }
    except ValueError as exc:
        # Validation errors that slipped past Pydantic (shouldn't happen, belt-and-suspenders)
        logger.warning("/analyze validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("/analyze unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")


@traceable(
    name="humanize_pipeline",
    run_type="chain",
    tags=["stealthtext", "api"],
)
def _run_humanize_graph(text: str) -> dict:
    """
    Thin wrapper so LangSmith captures the full agent pipeline
    (detect → rewrite loop) as a single named top-level trace.
    """
    initial_state = {
        "text": text,
        "original_text": text,
        "iterations": 0,
        "history": [],
    }
    return agent_app.invoke(initial_state)


@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: TextRequest):
    """
    Run the LangGraph detect→rewrite agent on the supplied text.
    Returns the humanized text along with final scores and iteration count.
    """
    logger.info("/humanize — %d chars", len(request.text))
    try:
        final_state = _run_humanize_graph(request.text)
        logger.info(
            "/humanize done — %d iteration(s), final ppl=%.1f",
            final_state["iterations"], final_state["perplexity"],
        )
        return {
            "original_text":    final_state["original_text"],
            "final_text":       final_state["text"],
            "final_perplexity": final_state["perplexity"],
            "final_burstiness": final_state["burstiness"],
            "iterations":       final_state["iterations"],
            "score_history":    final_state.get("score_history", []),
        }
    except ValueError as exc:
        logger.warning("/humanize validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("/humanize unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during humanization.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
