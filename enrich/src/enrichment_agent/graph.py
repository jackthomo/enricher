"""Define a data enrichment agent.

Works with a chat model with tool calling support.
"""

import logging
import json
import time
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from enrichment_agent import prompts
from enrichment_agent.configuration import Configuration
from enrichment_agent.state import InputState, OutputState, State
from enrichment_agent.tools import scrape_website, search
from enrichment_agent.utils import init_model


logger = logging.getLogger(__name__)


async def call_agent_model(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Call the primary Language Model (LLM) to decide on the next research action.

    This asynchronous function performs the following steps:
    1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
    2. Prepares the prompt and message history for the LLM.
    3. Initializes and configures the LLM with available tools.
    4. Invokes the LLM and processes its response.
    5. Handles the LLM's decision to either continue research or submit final info.
    """
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)

    # Define the 'Info' tool, which is the user-defined extraction schema
    info_tool = {
        "name": "Info",
        "description": "Call this when you have gathered all the relevant info",
        "parameters": state.extraction_schema,
    }

    # Format the prompt defined in prompts.py with the extraction schema and topic
    p = configuration.prompt.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    elapsed = time.monotonic() - state.start_time
    remaining_loops = max(configuration.max_loops - state.loop_step, 0)
    remaining_info_calls = max(configuration.max_info_tool_calls - state.info_call_count, 0)
    remaining_seconds = max(configuration.max_time_seconds - int(elapsed), 0)
    constraint_lines = [
        f"Remaining loops before wrap-up: {remaining_loops} (max {configuration.max_loops}).",
        f"Remaining Info submissions: {remaining_info_calls} (max {configuration.max_info_tool_calls}).",
        f"Time budget: {configuration.max_time_seconds}s total; elapsed ~{int(elapsed)}s; remaining ~{remaining_seconds}s.",
        f"Search results per query are capped at {configuration.max_search_results}. Minimize tool calls; prefer reusing gathered info.",
        "If remaining loops or time are low, prioritize calling the Info tool with best-effort output instead of additional searches.",
    ]
    p = "\n".join(
        [
            "RUN CONSTRAINTS:",
            *constraint_lines,
            "",
            p,
        ]
    )
    if state.input_rows:
        try:
            rows_text = json.dumps(state.input_rows, indent=2)
        except TypeError:
            rows_text = str(state.input_rows)
        p += (
            f"\n\nThe user provided {len(state.input_rows)} input row(s) (from CSV). "
            "Use these as the batch to enrich: keep provided values and fill missing fields according to the schema. "
            "Do not add or remove rows unless explicitly asked."
            f"\n<input_rows>\n{rows_text}\n</input_rows>"
        )
    wrap_up_notes: list[str] = []
    if state.info_call_count >= configuration.max_info_tool_calls:
        wrap_up_notes.append(
            "You have reached the maximum allowed Info tool submissions. Provide your best-effort final answer now. Avoid extra searches or scrapes unless absolutely critical. Call the Info tool exactly once to return what you have."
        )
    if state.loop_step >= configuration.max_loops:
        wrap_up_notes.append(
            "You are at the loop limit. Provide your best-effort final answer now using the Info tool and avoid further tool calls."
        )
    if elapsed >= configuration.max_time_seconds:
        wrap_up_notes.append(
            f"You have reached the time limit ({configuration.max_time_seconds}s). Provide your best-effort final answer now. Avoid any additional tool calls unless necessary to complete the Info response."
        )
    if wrap_up_notes:
        p += "\n\nWRAP UP INSTRUCTIONS:\n- " + "\n- ".join(wrap_up_notes)

    # Create the messages list with the formatted prompt and the previous messages
    messages = [HumanMessage(content=p)] + state.messages

    # Initialize the raw model with the provided configuration and bind the tools
    raw_model = init_model(config)
    model = raw_model.bind_tools([scrape_website, search, info_tool], tool_choice="any")
    logger.info(
        "call_agent_model: invoking LLM",
        extra={"topic": state.topic, "loop_step": state.loop_step},
    )
    response = cast(AIMessage, await model.ainvoke(messages))  # type: ignore[redundant-cast]

    # Initialize info to None
    info = None

    # Check if the response has tool calls
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "Info":
                info = tool_call["args"]
                break
    if info is not None:
        # The agent is submitting their answer;
        # ensure it isn't erroneously attempting to simultaneously perform research
        response.tool_calls = [
            next(tc for tc in response.tool_calls if tc["name"] == "Info")
        ]
    response_messages: List[BaseMessage] = [response]
    tool_names = [tc["name"] for tc in (response.tool_calls or [])]
    logger.info(
        "call_agent_model: received response",
        extra={
            "topic": state.topic,
            "loop_step": state.loop_step,
            "tool_calls": tool_names,
            "info_submitted": info is not None,
        },
    )
    if not response.tool_calls:  # If LLM didn't respect the tool_choice
        response_messages.append(
            HumanMessage(content="Please respond by calling one of the provided tools.")
        )
    return {
        "messages": response_messages,
        "info": info,
        # Add 1 to the step count
        "loop_step": 1,
        "info_call_count": 1 if info is not None else 0,
    }


class InfoIsSatisfactory(BaseModel):
    """Validate whether the current extracted info is satisfactory and complete."""

    reason: List[str] = Field(
        description="First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons."
    )
    is_satisfactory: bool = Field(
        description="After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching."
    )
    improvement_instructions: Optional[str] = Field(
        description="If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory."
        " This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
        default=None,
    )


async def reflect(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Validate the quality of the data enrichment agent's output.

    This asynchronous function performs the following steps:
    1. Prepares the initial prompt using the main prompt template.
    2. Constructs a message history for the model.
    3. Prepares a checker prompt to evaluate the presumed info.
    4. Initializes and configures a language model with structured output.
    5. Invokes the model to assess the quality of the gathered information.
    6. Processes the model's response and determines if the info is satisfactory.
    """
    configuration = Configuration.from_runnable_config(config)
    p = prompts.MAIN_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), topic=state.topic
    )
    elapsed = time.monotonic() - state.start_time
    remaining_loops = max(configuration.max_loops - state.loop_step, 0)
    remaining_info_calls = max(configuration.max_info_tool_calls - state.info_call_count, 0)
    remaining_seconds = max(configuration.max_time_seconds - int(elapsed), 0)
    constraint_lines = [
        f"Remaining loops before wrap-up: {remaining_loops} (max {configuration.max_loops}).",
        f"Remaining Info submissions: {remaining_info_calls} (max {configuration.max_info_tool_calls}).",
        f"Time budget: {configuration.max_time_seconds}s total; elapsed ~{int(elapsed)}s; remaining ~{remaining_seconds}s.",
    ]
    p = "\n".join(
        [
            "RUN CONSTRAINTS:",
            *constraint_lines,
            "",
            p,
        ]
    )
    if state.input_rows:
        try:
            rows_text = json.dumps(state.input_rows, indent=2)
        except TypeError:
            rows_text = str(state.input_rows)
        p += (
            f"\n\nThe user provided {len(state.input_rows)} input row(s) (from CSV). "
            "Ensure the final output covers these rows, preserves any existing values, and fills missing fields."
            f"\n<input_rows>\n{rows_text}\n</input_rows>"
        )
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"{reflect.__name__} expects the last message in the state to be an AI message with tool calls."
            f" Got: {type(last_message)}"
        )
    messages = [HumanMessage(content=p)] + state.messages[:-1]
    presumed_info = state.info
    checker_prompt = """I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}"""
    p1 = checker_prompt.format(presumed_info=json.dumps(presumed_info or {}, indent=2))
    messages.append(HumanMessage(content=p1))
    raw_model = init_model(config)
    bound_model = raw_model.with_structured_output(InfoIsSatisfactory)
    logger.info(
        "reflect: evaluating presumed info",
        extra={"topic": state.topic, "loop_step": state.loop_step},
    )
    response = cast(InfoIsSatisfactory, await bound_model.ainvoke(messages))
    if response.is_satisfactory and presumed_info:
        logger.info(
            "reflect: info satisfactory",
            extra={"topic": state.topic, "loop_step": state.loop_step},
        )
        return {
            "info": presumed_info,
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content="\n".join(response.reason),
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="success",
                )
            ],
        }
    else:
        logger.info(
            "reflect: info unsatisfactory",
            extra={
                "topic": state.topic,
                "loop_step": state.loop_step,
                "improvement": response.improvement_instructions,
            },
        )
        return {
            "messages": [
                ToolMessage(
                    tool_call_id=last_message.tool_calls[0]["id"],
                    content=f"Unsatisfactory response:\n{response.improvement_instructions}",
                    name="Info",
                    additional_kwargs={"artifact": response.model_dump()},
                    status="error",
                )
            ]
        }


def route_after_agent(
    state: State,
) -> Literal["reflect", "tools", "call_agent_model", "__end__"]:
    """Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:

    1. Error recovery: If the last message is unexpectedly not an AIMessage.
    2. Info submission: If the agent has called the "Info" tool to submit findings.
    3. Continued research: If the agent has called any other tool.
    """
    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code),
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "call_agent_model"
    # If the "Into" tool was called, then the model provided its extraction output. Reflect on the result
    if last_message.tool_calls and last_message.tool_calls[0]["name"] == "Info":
        return "reflect"
    # The last message is a tool call that is not "Info" (extraction output)
    else:
        return "tools"


def route_after_checker(
    state: State, config: RunnableConfig
) -> Literal["__end__", "call_agent_model"]:
    """Schedule the next node after the checker's evaluation.

    This function determines whether to continue the research process or end it
    based on the checker's evaluation and the current state of the research.
    """
    configurable = Configuration.from_runnable_config(config)
    last_message = state.messages[-1]
    info_cap_reached = state.info_call_count >= configurable.max_info_tool_calls
    loop_cap_reached = state.loop_step >= configurable.max_loops
    time_cap_reached = (time.monotonic() - state.start_time) >= configurable.max_time_seconds

    if info_cap_reached and not state.info:
        logger.info(
            "route_after_checker: forcing wrap-up (info cap reached, no info yet)",
            extra={
                "topic": state.topic,
                "loop_step": state.loop_step,
                "info_calls": state.info_call_count,
            },
        )
        return "call_agent_model"

    if loop_cap_reached and not state.info:
        logger.info(
            "route_after_checker: forcing wrap-up (loop cap reached, no info yet)",
            extra={"topic": state.topic, "loop_step": state.loop_step},
        )
        return "call_agent_model"

    if time_cap_reached and not state.info:
        logger.info(
            "route_after_checker: forcing wrap-up (time cap reached, no info yet)",
            extra={
                "topic": state.topic,
                "loop_step": state.loop_step,
                "elapsed": time.monotonic() - state.start_time,
            },
        )
        return "call_agent_model"

    if state.loop_step < configurable.max_loops:
        if info_cap_reached and state.info:
            logger.info(
                "route_after_checker: ending (info cap reached, returning best-effort)",
                extra={
                    "topic": state.topic,
                    "loop_step": state.loop_step,
                    "info_calls": state.info_call_count,
                },
            )
            return "__end__"
        if time_cap_reached and state.info:
            logger.info(
                "route_after_checker: ending (time cap reached, returning best-effort)",
                extra={
                    "topic": state.topic,
                    "loop_step": state.loop_step,
                    "elapsed": time.monotonic() - state.start_time,
                },
            )
            return "__end__"
        if not state.info:
            logger.debug(
                "route_after_checker: continuing (no info)",
                extra={"topic": state.topic, "loop_step": state.loop_step},
            )
            return "call_agent_model"
        if not isinstance(last_message, ToolMessage):
            raise ValueError(
                f"{route_after_checker.__name__} expected a tool messages. Received: {type(last_message)}."
            )
        if last_message.status == "error":
            # Research deemed unsatisfactory
            logger.debug(
                "route_after_checker: continuing (checker flagged error)",
                extra={"topic": state.topic, "loop_step": state.loop_step},
            )
            return "call_agent_model"
        # It's great!
        logger.info(
            "route_after_checker: ending (satisfactory)",
            extra={"topic": state.topic, "loop_step": state.loop_step},
        )
        return "__end__"
    else:
        logger.info(
            "route_after_checker: ending (max loops reached)",
            extra={"topic": state.topic, "loop_step": state.loop_step},
        )
        return "__end__"


# Create the graph
workflow = StateGraph(
    State, input_schema=InputState, output_schema=OutputState, context_schema=Configuration
)
workflow.add_node(call_agent_model)
workflow.add_node(reflect)
workflow.add_node("tools", ToolNode([search, scrape_website]))
workflow.add_edge("__start__", "call_agent_model")
workflow.add_conditional_edges("call_agent_model", route_after_agent)
workflow.add_edge("tools", "call_agent_model")
workflow.add_conditional_edges("reflect", route_after_checker)

graph = workflow.compile()
graph.name = "ResearchTopic"
