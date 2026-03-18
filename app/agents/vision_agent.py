import asyncio
import logging
from functools import lru_cache

from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.schemas.agent import AgentResult
from app.schemas.websocket import ImageRequest
from app.services.image_processing import validate_and_prepare_image
from app.services.session_memory import session_memory
from app.services.vision import identify_image

logger = logging.getLogger(__name__)

VALID_TASK_TYPES = {"describe_scene", "identify_component", "read_text", "analyze_circuit"}


class IntentClassification(BaseModel):
    """Classified intent of a student's prompt."""
    task_type: str = Field(
        description=(
            "One of: describe_scene, identify_component, read_text, analyze_circuit. "
            "describe_scene = student wants a general overview of what they see. "
            "identify_component = student wants to know what a specific part/component is. "
            "read_text = student wants text read from labels, screens, or documents. "
            "analyze_circuit = student wants to verify if wiring/connections are correct."
        ),
    )
    reasoning: str = Field(
        description="Brief one-sentence explanation for why this task type was chosen.",
    )


def _load_memory(state: AgentState) -> AgentState:
    client_id = state["client_id"]
    return {"memory_context": session_memory.recent_summary(client_id)}


async def _prepare_image(state: AgentState) -> AgentState:
    image_b64, media_type = await asyncio.to_thread(
        validate_and_prepare_image,
        state["image_bytes"],
    )
    return {"image_b64": image_b64, "media_type": media_type}


async def _infer_intent(state: AgentState) -> AgentState:
    """Use a fast LLM with structured output to classify the student's intent."""
    from app.services.model_client import get_llm

    prompt = state["prompt"]
    try:
        llm = get_llm("groq-intent")
        structured_llm = llm.with_structured_output(IntentClassification)
        result: IntentClassification = await structured_llm.ainvoke(
            f"Classify the following student prompt into one of the task types.\n\n"
            f"Student prompt: \"{prompt}\""
        )
        task_type = result.task_type.strip().lower()
        if task_type not in VALID_TASK_TYPES:
            logger.warning(
                "Intent model returned invalid task_type=%r for prompt=%r, falling back to describe_scene",
                task_type, prompt,
            )
            task_type = "describe_scene"
        logger.info(
            "Intent: task_type=%s | reasoning=%s | prompt=%r",
            task_type, result.reasoning, prompt,
        )
        return {"task_type": task_type}
    except Exception:
        logger.exception("Intent classification failed for prompt=%r, falling back to describe_scene", prompt)
        return {"task_type": "describe_scene"}


async def _run_vision_model(state: AgentState) -> AgentState:
    response_text, structured_data = await identify_image(
        image_b64=state["image_b64"],
        media_type=state["media_type"],
        prompt=state["prompt"],
        task_type=state["task_type"],
        memory_context=state["memory_context"],
    )
    return {"response_text": response_text, "structured_data": structured_data}


def _update_memory(state: AgentState) -> AgentState:
    session_memory.append(
        client_id=state["client_id"],
        prompt=state["prompt"],
        task_type=state["task_type"],
        response_text=state["response_text"],
    )
    return {}


@lru_cache(maxsize=1)
def _compile_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ValueError("LangGraph is not installed. Add the `langgraph` package to your environment.") from exc

    graph = StateGraph(AgentState)
    graph.add_node("load_memory", _load_memory)
    graph.add_node("prepare_image", _prepare_image)
    graph.add_node("infer_intent", _infer_intent)
    graph.add_node("run_vision_model", _run_vision_model)
    graph.add_node("update_memory", _update_memory)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "prepare_image")
    graph.add_edge("prepare_image", "infer_intent")
    graph.add_edge("infer_intent", "run_vision_model")
    graph.add_edge("run_vision_model", "update_memory")
    graph.add_edge("update_memory", END)
    return graph.compile()


async def run_vision_agent(request: ImageRequest) -> AgentResult:
    graph = _compile_graph()
    result = await graph.ainvoke(
        {
            "client_id": request.client_id,
            "request_id": request.request_id,
            "prompt": request.prompt,
            "image_bytes": request.image_bytes,
        }
    )
    return AgentResult(
        request_id=request.request_id,
        task_type=result["task_type"],
        response_text=result["response_text"],
        structured_data=result.get("structured_data"),
    )
