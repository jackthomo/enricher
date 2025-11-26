"""FastAPI wrapper for the enrichment graph."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

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
    max_time_seconds: Optional[int] = Field(
        default=None, ge=1, description="Maximum wall-clock seconds before wrapping up"
    )


class EnrichmentRequest(BaseModel):
    """Request body for the enrichment endpoint."""

    topic: str = Field(..., description="Topic to research")
    extraction_schema: Dict[str, Any] = Field(
        ..., description="JSON schema describing the structured output"
    )
    example_rows: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional few-shot example rows to illustrate the expected format. These guide the model but are not treated as the target batch.",
    )
    input_rows: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional input rows (e.g., from CSV) that should be completed according to the schema",
    )
    configurable: Optional[ConfigurablePayload] = Field(
        default=None, description="Optional overrides for the graph configuration"
    )


class EnrichmentResponse(BaseModel):
    """Response body for the enrichment endpoint."""

    info: Dict[str, Any]
    trace: list[Dict[str, Any]]
    steps: Optional[list[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _has_type(schema: Dict[str, Any], target: str) -> bool:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return target in schema_type
    return schema_type == target


def _coerce_decimal(value: Any) -> Optional[Decimal]:
    """Best-effort conversion of user/LLM-provided values into Decimal."""
    if isinstance(value, bool):  # bool is a subclass of int; guard explicitly
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        match = _NUMBER_RE.search(cleaned)
        if not match:
            return None
        try:
            return Decimal(match.group(0))
        except (InvalidOperation, ValueError):
            return None
    return None


def _round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round a decimal to the nearest multiple of `step`."""
    if step == 0:
        return value
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        rounded = (value / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
        try:
            return rounded.quantize(step, rounding=ROUND_HALF_UP)
        except InvalidOperation:
            # Fall back to the computed rounded value if quantize fails on exotic steps
            return rounded


def _sanitize_number(value: Any, schema: Dict[str, Any]) -> Any:
    """Coerce and round numeric values according to schema hints."""
    decimal_value = _coerce_decimal(value)
    if decimal_value is None:
        return value

    multiple_of = schema.get("multipleOf")
    if multiple_of is not None:
        try:
            step = Decimal(str(multiple_of))
            decimal_value = _round_to_step(decimal_value, step)
        except (InvalidOperation, ValueError):
            pass

    try:
        return float(decimal_value)
    except (InvalidOperation, ValueError):
        return value


def _sanitize_against_schema(data: Any, schema: Dict[str, Any]) -> Any:
    """
    Walk output data and enforce numeric formatting based on the extraction schema.

    Currently enforces:
    - For number types with `multipleOf`, rounds to the nearest multiple and coerces to a number.
    """
    if not isinstance(schema, dict):
        return data

    if _has_type(schema, "object") and isinstance(data, dict):
        properties = schema.get("properties") or {}
        sanitized = dict(data)
        for key, prop_schema in properties.items():
            if key in sanitized:
                sanitized[key] = _sanitize_against_schema(sanitized[key], prop_schema)
        return sanitized

    if _has_type(schema, "array") and isinstance(data, list):
        items_schema = schema.get("items")
        if not items_schema:
            return data
        return [_sanitize_against_schema(item, items_schema) for item in data]

    if _has_type(schema, "number") or ("multipleOf" in schema):
        return _sanitize_number(data, schema)

    return data


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
    start_time = time.monotonic()
    config = {}
    if request.configurable:
        config["configurable"] = request.configurable.model_dump(exclude_none=True)
    try:
        logger.info("Enrichment request received", extra={"topic": request.topic})
        result, steps = await _invoke_with_steps(
            {
                "topic": request.topic,
                "extraction_schema": request.extraction_schema,
                "example_rows": request.example_rows,
                "input_rows": request.input_rows,
            },
            config=config or None,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=f"Failed to run enrichment: {exc}")

    info = result.get("info")
    if info is None:
        raise HTTPException(status_code=500, detail="Graph completed without returning info")
    info = _sanitize_against_schema(info, request.extraction_schema)
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
    duration_ms = int((time.monotonic() - start_time) * 1000)
    metrics = {"duration_ms": duration_ms}
    # Token usage may be available in messages' response metadata; include if present.
    usage = result.get("usage") or result.get("token_usage") or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in usage:
            metrics[key] = usage[key]
    return EnrichmentResponse(
        info=info,
        trace=trace,
        steps=steps or None,
        metrics=metrics,
        evidence=result.get("evidence"),
        plan=result.get("plan"),
    )
