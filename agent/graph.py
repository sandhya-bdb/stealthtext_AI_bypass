from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator
from logic.detector import AIDetector
from logic.humanizer import TextHumanizer
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

# 2. Logic Initialization
detector = AIDetector()
humanizer = TextHumanizer()

# 3. Nodes
@traceable(name="detector_node", run_type="chain", tags=["stealthtext", "detector"])
def detector_node(state: AgentState):
    print(f"--- DETECTOR NODE (Iteration {state.get('iterations', 0)}) ---")
    text = state['text']
    
    # Calculate scores
    result = detector.analyze(text)
    
    return {
        "perplexity": result['perplexity'],
        "burstiness": result['burstiness'],
        "verdict": "Likely Human" if result['ai_score'] < 50 else "Likely AI", # Simplified threshold
        "iterations": state.get('iterations', 0),
        "score_history": state.get('score_history', []) + [{
            "iteration": state.get('iterations', 0),
            "perplexity": result['perplexity'],
            "burstiness": result['burstiness']
        }]
    }

@traceable(name="rewriter_node", run_type="chain", tags=["stealthtext", "rewriter"])
def rewriter_node(state: AgentState):
    print("--- REWRITER NODE ---")
    current_text = state['text']
    
    # Rewrite text
    new_text = humanizer.rewrite(current_text)
    
    return {
        "text": new_text,
        "iterations": state['iterations'] + 1,
        "history": state.get('history', []) + [current_text]
    }

# 4. Edges
def should_continue(state: AgentState):
    # Thresholds: PPL > 60 and Burstiness > 15 usually indicates Human
    # Or stop if max iterations reached
    
    if state['iterations'] >= 3:
        return "end"
    
    # Force at least one rewrite if the user clicked "Humanize"
    if state['iterations'] == 0:
        return "rewrite"
        
    if state['perplexity'] > 80 and state['burstiness'] > 20: # Increased thresholds
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
