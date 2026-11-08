"""Glue code used by chat.py and run_scenarios.py:
storage paths, choosing the model, and running one user message with a trace."""

import json
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from helpdesk.agent_graph import build_helpdesk_graph, checkpoint_config, make_input
from helpdesk.data import PROJECT_DIR
from helpdesk.fake_model import FakeHelpdeskModel
from helpdesk.user_memory import UserMemory

load_dotenv(PROJECT_DIR / ".env")


def storage_dir():
    folder = Path(os.getenv("HELPDESK_STORAGE_DIR", "local-storage"))
    if not folder.is_absolute():
        folder = PROJECT_DIR / folder
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_model(fake=False):
    """Real OpenAI model from .env, or the clearly labelled fake model."""
    if fake:
        model = FakeHelpdeskModel()
        return model, "FAKE MODEL - " + model.name
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    if not api_key or not model_name:
        raise SystemExit("OPENAI_API_KEY and MODEL_NAME must be set in .env for the real model. "
                         "Use --fake-model for an offline run with the scripted fake model.")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=0), f"REAL MODEL - OpenAI {model_name}"


class HelpdeskSession:
    """Opens the storage, builds the graph, and runs messages."""

    def __init__(self, model, folder=None, ticket_dir=None):
        folder = Path(folder) if folder else storage_dir()
        folder.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(folder / "checkpoints.sqlite", check_same_thread=False)
        self.user_memory = UserMemory(folder / "user_memory.json")
        self.graph = build_helpdesk_graph(model, self.user_memory, SqliteSaver(self.connection),
                                          ticket_dir=ticket_dir)

    def close(self):
        self.connection.close()

    def send(self, user_id, thread_id, status_profile, message):
        """Run one user message and return a trace dict (used for printing and logs)."""
        config = checkpoint_config(user_id, thread_id)
        before = self.graph.get_state(config).values.get("messages", [])
        trace = {
            "user_id": user_id, "thread_id": thread_id, "status_profile": status_profile,
            "message": message,
            "conversation_memory": [f"{type(m).__name__}: {short(m)}" for m in before
                                    if isinstance(m, (HumanMessage, AIMessage)) and m.content],
            "steps": [],
        }
        for chunk in self.graph.stream(make_input(user_id, thread_id, status_profile, message),
                                       config, stream_mode="updates"):
            for node, update in chunk.items():
                trace["steps"].append({"node": node, "update": describe_update(update)})

        final = self.graph.get_state(config).values
        trace["reply"] = final["messages"][-1].content
        trace["memory_note"] = final.get("memory_note")
        trace["user_language"] = final.get("user_language")
        trace["tickets"] = final.get("tickets", [])
        trace["loop_limit_reached"] = final.get("loop_limit_reached", False)
        trace["model_calls"] = final.get("model_calls", 0)
        return trace


def short(message, limit=160):
    text = message.content if isinstance(message.content, str) else str(message.content)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def describe_update(update):
    """Make a node's state update readable (messages become short descriptions)."""
    result = {}
    for key, value in (update or {}).items():
        if key == "messages":
            result[key] = [describe_message(m) for m in value]
        else:
            result[key] = value
    return result


def describe_message(message):
    if isinstance(message, AIMessage) and message.tool_calls:
        calls = [f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in message.tool_calls]
        return "MODEL REQUESTS TOOLS: " + ", ".join(calls)
    if isinstance(message, ToolMessage):
        return f"TOOL RESULT {message.name}: {message.content}"
    if isinstance(message, AIMessage):
        return "MODEL ANSWER: " + message.content
    return f"{type(message).__name__}: {message.content}"


def format_trace(trace):
    """Human-readable text for the terminal and for outputs/ logs."""
    lines = [
        f"user_id={trace['user_id']}  thread_id={trace['thread_id']}  status_profile={trace['status_profile']}",
        f"USER: {trace['message']}",
        "",
        f"Read from CONVERSATION MEMORY (checkpoint of this thread): {len(trace['conversation_memory'])} earlier messages",
    ]
    lines += [f"    - {m}" for m in trace["conversation_memory"]]
    lines.append(f"Read from USER MEMORY: {trace['memory_note']}")
    lines.append("Nodes visited and state updates:")
    for number, step in enumerate(trace["steps"], 1):
        lines.append(f"  {number}. {step['node']}")
        for key, value in step["update"].items():
            if key == "messages":
                for item in value:
                    lines.append(f"       {item}")
            elif key not in ("service_checks",):
                lines.append(f"       {key} = {value}")
    sources = []
    for step in trace["steps"]:
        for item in step["update"].get("messages", []):
            if item.startswith("TOOL RESULT find_instructions") and '"found": true' in item:
                sources.append("REFERENCE FILE data/helpdesk.json (find_instructions)")
            if item.startswith("TOOL RESULT check_service_status"):
                sources.append(f"STATUS SNAPSHOT data/service_status.json profile={trace['status_profile']}")
    if trace["conversation_memory"]:
        sources.append("CONVERSATION MEMORY (SQLite checkpoint)")
    if trace["user_language"]:
        sources.append(f"USER MEMORY (language={trace['user_language']})")
    lines.append("Information sources used: " + (", ".join(dict.fromkeys(sources)) or "none"))
    lines.append(f"Model calls this turn: {trace['model_calls']}   loop limit reached: {trace['loop_limit_reached']}")
    lines.append(f"Local ticket files in this conversation: {trace['tickets'] or 'none'}")
    lines.append(f"HELPDESK: {trace['reply']}")
    return "\n".join(lines)
