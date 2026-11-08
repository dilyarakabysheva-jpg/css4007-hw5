"""Graph tests for Parts 2-3. They use the clearly labelled FAKE model
(helpdesk/fake_model.py), so no API key is needed. They check the graph wiring,
the tool loop and its limit, and memory; they do NOT measure real LLM quality."""

import os
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

from helpdesk.fake_model import FakeHelpdeskModel
from helpdesk.session import HelpdeskSession

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONDITIONAL_REQUEST = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not connect. "
                       "Check whether Wi-Fi is down. If it is operating, prepare a ticket about this problem.")


def tool_names(trace):
    return [line.split("(")[0].replace("MODEL REQUESTS TOOLS: ", "")
            for step in trace["steps"] for line in step["update"].get("messages", [])
            if line.startswith("MODEL REQUESTS TOOLS")]


def new_session(tmp_path, model=None):
    return HelpdeskSession(model or FakeHelpdeskModel(), folder=tmp_path / "storage",
                           ticket_dir=tmp_path / "tickets")


# ---------- Part 2: the tool loop ----------

def test_operational_status_leads_to_ticket(tmp_path):
    session = new_session(tmp_path)
    trace = session.send("demo-user-a", "t1", "normal", CONDITIONAL_REQUEST)
    session.close()
    assert tool_names(trace) == ["check_service_status", "prepare_ticket"]
    assert len(list((tmp_path / "tickets").glob("*.json"))) == 1


def test_outage_means_no_ticket(tmp_path):
    session = new_session(tmp_path)
    trace = session.send("demo-user-a", "t1", "wifi_outage", CONDITIONAL_REQUEST)
    session.close()
    assert tool_names(trace) == ["check_service_status"]
    assert not (tmp_path / "tickets").exists()


def test_unknown_status_means_no_ticket_and_a_question(tmp_path):
    session = new_session(tmp_path)
    trace = session.send("demo-user-a", "t1", "status_unavailable", CONDITIONAL_REQUEST)
    session.close()
    assert "prepare_ticket" not in tool_names(trace)
    assert trace["reply"].endswith("?")


class AlwaysWantsToolsModel:
    """Fake model that never stops asking for tools - used to test the loop limit."""
    def bind_tools(self, schemas):
        return self

    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[{"name": "find_instructions", "args": {"topic": "wifi"},
                                                  "id": f"call_{len(messages)}", "type": "tool_call"}])


def test_loop_limit_stops_the_graph(tmp_path):
    session = new_session(tmp_path, AlwaysWantsToolsModel())
    trace = session.send("demo-user-a", "t1", "normal", "wifi")
    session.close()
    assert trace["loop_limit_reached"] is True
    assert trace["steps"][-1]["node"] == "stop_at_limit"
    assert trace["model_calls"] == 5


# ---------- Part 3: memory ----------

def test_conversation_and_preference_survive_reopening_storage(tmp_path):
    first = new_session(tmp_path)
    first.send("demo-user-a", "t1", "normal", "Please remember that I prefer replies in Kazakh. "
                                              "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    first.close()                                  # like closing the program

    second = new_session(tmp_path)                 # new graph, new database connection
    trace = second.send("demo-user-a", "t1", "normal", "Which problem and steps did we discuss?")
    second.close()
    assert len(trace["conversation_memory"]) == 2  # old messages came from the checkpoint
    assert "tried reconnecting" in trace["reply"]
    assert trace["user_language"] == "Kazakh"


def test_new_thread_has_preference_but_not_old_messages(tmp_path):
    session = new_session(tmp_path)
    session.send("demo-user-a", "a1", "normal", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    trace = session.send("demo-user-a", "a2", "normal", "Which language did I ask you to use?")
    session.close()
    assert trace["conversation_memory"] == []
    assert trace["user_language"] == "Kazakh"
    assert "Kazakh" in trace["reply"]


def test_users_do_not_share_memory(tmp_path):
    session = new_session(tmp_path)
    session.send("demo-user-a", "same-thread", "normal", "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    trace = session.send("demo-user-b", "same-thread", "normal",
                         "Have I told you a preferred language or a previous problem?")
    session.close()
    assert trace["user_language"] is None
    assert trace["conversation_memory"] == []       # same thread name, but a different user
    assert "Kazakh" not in trace["reply"]


def test_memory_survives_a_real_process_restart(tmp_path):
    """Runs chat.py twice as two separate processes against the same storage folder."""
    env = dict(os.environ, HELPDESK_STORAGE_DIR=str(tmp_path / "storage"),
               HELPDESK_TICKET_DIR=str(tmp_path / "tickets"), PYTHONIOENCODING="utf-8")
    base = [sys.executable, "chat.py", "--user", "demo-user-a", "--thread", "restart", "--fake-model", "--quiet"]

    def run(message):
        return subprocess.run(base + ["--message", message], cwd=PROJECT_DIR, env=env,
                              capture_output=True, text=True, encoding="utf-8", check=True).stdout

    run("My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    output = run("Which problem and steps did we discuss?")
    assert "Wi-Fi problem" in output and "tried reconnecting" in output
