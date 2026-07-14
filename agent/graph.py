from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator
from logic.detector import AIDetector
from logic.humanizer import TextHumanizer
from logic.external_detector import GPTZeroClient
from dotenv import load_dotenv

load_dotenv()

# LangSmith: @traceable makes each function a named span in the trace.
# If LANGCHAIN_TRACING_V2 is not set this import is a no-op decorator.
try:
    from langsmith import traceable
except ImportError:
    def traceable(**kwargs):
        """Fallback no-op decorator when langsmith is not installed."""
        def decorator(fn):
            return fn
        return decorator

# 1. State Definition
class AgentState(TypedDict):
    text: str
    original_text: str
    perplexity: float
    burstiness: float
    verdict: str
    iterations: int
    history: List[str]
    score_history: List[dict]
    tone: str
    gptzero_score: float

# 2. Logic Initialization
detector = AIDetector()
humanizer = TextHumanizer()
gptzero = GPTZeroClient()

# 3. Nodes
@traceable(name="detector_node", run_type="chain", tags=["stealthtext", "detector"])
def detector_node(state: AgentState):
    print(f"--- DETECTOR NODE (Iteration {state.get('iterations', 0)}) ---")
    text = state['text']
    iterations = state.get('iterations', 0)
    
    # Calculate scores
    result = detector.analyze(text)
    
    # Call external GPTZero API only if key is configured, we're past iteration 0,
    # and the local pre-screen looks promising (perplexity > 55, burstiness > 15)
    gptzero_score = 0.0
    if gptzero.api_key and iterations > 0:
        if result['perplexity'] > 55 and result['burstiness'] > 15:
            gptzero_score = gptzero.check(text)
    
    return {
        "perplexity": result['perplexity'],
        "burstiness": result['burstiness'],
        "verdict": "Likely Human" if result['ai_score'] < 50 else "Likely AI", # Simplified threshold
        "iterations": iterations,
        "gptzero_score": gptzero_score,
        "score_history": state.get('score_history', []) + [{
            "iteration": iterations,
            "perplexity": result['perplexity'],
            "burstiness": result['burstiness'],
            "gptzero_score": gptzero_score
        }]
    }

@traceable(name="rewriter_node", run_type="chain", tags=["stealthtext", "rewriter"])
def rewriter_node(state: AgentState):
    print("--- REWRITER NODE ---")
    current_text = state['text']
    tone = state.get('tone', 'casual')
    
    # Rewrite text
    new_text = humanizer.rewrite(current_text, tone=tone)
    
    return {
        "text": new_text,
        "iterations": state['iterations'] + 1,
        "history": state.get('history', []) + [current_text]
    }

# 4. Edges
def should_continue(state: AgentState):
    if state['iterations'] >= 3:
        return "end"
    
    # Force at least one rewrite if the user clicked "Humanize"
    if state['iterations'] == 0:
        return "rewrite"
        
    local_passed = state['perplexity'] > 80 and state['burstiness'] > 20
    
    if gptzero.api_key:
        # If external verification is enabled, we require both local pass and external pass (< 40% AI)
        if local_passed and state.get('gptzero_score', 100.0) < 40.0:
            return "end"
    else:
        # If external verification is disabled, rely purely on local scores
        if local_passed:
            return "end"
        
    return "rewrite"

# 5. Graph Construction
workflow = StateGraph(AgentState)

workflow.add_node("detector", detector_node)
workflow.add_node("rewriter", rewriter_node)

workflow.set_entry_point("detector")

workflow.add_conditional_edges(
    "detector",
    should_continue,
    {
        "rewrite": "rewriter",
        "end": END
    }
)

workflow.add_edge("rewriter", "detector")

# 6. Compilation
app = workflow.compile()
