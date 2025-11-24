"""FastAPI wrapper for the enrichment graph."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from enrichment_agent.graph import graph
from langchain_core.messages import BaseMessage, message_to_dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI(
    title="Enrichment Agent API",
    version="0.1.0",
    description="Run the enrichment LangGraph over HTTP",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigurablePayload(BaseModel):
    """Optional configuration overrides for the graph."""

    model: Optional[str] = Field(
        default=None,
        description="Override the model, e.g. openai/gpt-4o-mini or anthropic/claude-3-haiku",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt template with {info} and {topic} placeholders",
    )
    max_search_results: Optional[int] = Field(
        default=None, ge=1, description="Maximum search results returned per query"
    )
    max_info_tool_calls: Optional[int] = Field(
        default=None, ge=1, description="Maximum number of Info tool calls"
    )
    max_loops: Optional[int] = Field(
        default=None, ge=1, description="Maximum loop iterations before stopping"
    )


class EnrichmentRequest(BaseModel):
    """Request body for the enrichment endpoint."""

    topic: str = Field(..., description="Topic to research")
    extraction_schema: Dict[str, Any] = Field(
        ..., description="JSON schema describing the structured output"
    )
    configurable: Optional[ConfigurablePayload] = Field(
        default=None, description="Optional overrides for the graph configuration"
    )


class EnrichmentResponse(BaseModel):
    """Response body for the enrichment endpoint."""

    info: Dict[str, Any]
    trace: list[Dict[str, Any]]
    steps: Optional[list[Dict[str, Any]]] = None


def _serialize_messages(messages: List[BaseMessage]) -> list[Dict[str, Any]]:
    """Convert LangChain messages to JSON-safe dicts with useful fields."""
    serialized = []
    for msg in messages:
        as_dict = message_to_dict(msg)
        tool_calls = getattr(msg, "tool_calls", None) or msg.additional_kwargs.get(
            "tool_calls"
        )
        serialized.append(
            {
                "type": as_dict.get("type"),
                "data": {
                    **(as_dict.get("data") or {}),
                    "tool_calls": tool_calls,
                },
            }
        )
    return serialized


async def _invoke_with_steps(
    inputs: Dict[str, Any], config: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run the graph while collecting node-level steps."""
    steps: List[Dict[str, Any]] = []
    final_output: Dict[str, Any] = {}
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        event_type = event.get("event")
        name = event.get("name")
        if event_type in {"on_node_start", "on_node_end"}:
            payload = event.get("data") or {}
            steps.append(
                {
                    "event": event_type,
                    "node": name,
                    "output_keys": list((payload.get("output") or {}).keys()),
                }
            )
        if event_type == "on_chain_end":
            final_output = (event.get("data") or {}).get("output") or {}
    return final_output, steps


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness endpoint."""
    return {"status": "ok"}


@app.post("/enrich", response_model=EnrichmentResponse)
async def run_enrichment(request: EnrichmentRequest) -> EnrichmentResponse:
    """Run the enrichment graph for a topic and schema."""
    config = {}
    if request.configurable:
        config["configurable"] = request.configurable.model_dump(exclude_none=True)
    try:
        logger.info("Enrichment request received", extra={"topic": request.topic})
        result, steps = await _invoke_with_steps(
            {"topic": request.topic, "extraction_schema": request.extraction_schema},
            config=config or None,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=f"Failed to run enrichment: {exc}")

    info = result.get("info")
    if info is None:
        raise HTTPException(status_code=500, detail="Graph completed without returning info")
    messages = result.get("messages", [])
    if not messages:
        logger.warning("No messages returned in graph output", extra={"topic": request.topic})
    trace = _serialize_messages(messages)
    logger.info(
        "Enrichment complete",
        extra={
            "topic": request.topic,
            "messages_count": len(messages),
            "trace_keys": list(info.keys()),
        },
    )
    return EnrichmentResponse(info=info, trace=trace, steps=steps or None)
