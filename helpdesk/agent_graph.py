"""Parts 2 and 3: the helpdesk graph where a model chooses tools, with memory.

    START -> load_user_memory -> agent --(tool calls?)--> tools -> agent -> ...
                                   |
                                   +--(no tool calls)--> END
                                   +--(too many model calls)--> stop_at_limit -> END

* agent           : the MODEL decides: answer now, or request tool(s).
* tools           : OUR PROGRAM executes the requested tools and adds the results.
* load_user_memory: reads/saves the user's language preference (user memory).
* The checkpointer (SQLite) saves the whole state per thread (conversation memory).
"""

import json
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from helpdesk import tools
from helpdesk.user_memory import detect_language_preference

# The model may be called at most this many times while answering ONE user message.
MAX_MODEL_CALLS = 5


class HelpdeskState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # conversation history (appended, not replaced)
    user_id: str             # supplied by the application, never chosen by the model
    thread_id: str           # which conversation this is
    status_profile: str      # which service_status.json snapshot is selected
    user_language: str       # read from user memory (None if nothing saved)
    memory_note: str         # what load_user_memory read/saved (for the trace)
    model_calls: int         # model calls during the current user message (loop limit)
    loop_limit_reached: bool
    service_checks: dict     # service -> last status result in this conversation
    reference_ids: list      # instruction ids retrieved in this conversation
    tickets: list            # local ticket files created in this conversation


def build_system_prompt(state):
    language = state.get("user_language")
    if language:
        memory_text = f"Saved preferred reply language for this user: {language}. Reply in {language}."
    else:
        memory_text = "No language preference or other information is saved for this user."
    scope = ", ".join(tools.HELPDESK["scope"])
    return f"""You are a university IT helpdesk assistant in a PRACTICE exercise. All data is fictional.
You can help with: {scope}.

Rules:
- Use find_instructions to get Wi-Fi or password instructions. Never invent instructions.
- Use check_service_status when the user asks whether a service is down or has a known problem.
  Wi-Fi -> service "wifi". Password / account recovery -> service "account_recovery".
- Status "unknown" means the status could not be determined. It is NOT operational and NOT an outage.
- Follow the user's conditions exactly. Example: "if it is operating, prepare a ticket":
  * operational -> call prepare_ticket;
  * outage -> do NOT prepare the ticket, explain why;
  * unknown -> do NOT prepare the ticket; ask whether they want a draft explaining that the status could not be checked.
- prepare_ticket needs a problem description and the steps already tried, taken from the user's own words
  in this conversation. If either is missing, ask "What problem are you having, and what have you already tried?"
  Never add steps the user did not mention.
- Tickets are LOCAL DRAFTS only. Never say a ticket was sent. Tell the user the local file name.
- Never invent repair times and never claim a real university outage.
- Never ask for or store passwords, recovery codes or API keys.
- Use the earlier messages of this conversation to understand references such as "that" or "we discussed".
- If the user asks what they told you before and this conversation has no such information, say that nothing has been recorded.

User memory (loaded by the application for user_id={state.get('user_id')}): {memory_text}
"""


def build_helpdesk_graph(model, user_memory, checkpointer=None, ticket_dir=None,
                         max_model_calls=MAX_MODEL_CALLS):
    model_with_tools = model.bind_tools(tools.TOOL_SCHEMAS)

    # ---------------- nodes ----------------

    def load_user_memory(state):
        user_id = state["user_id"]
        saved = user_memory.get(user_id)
        note = f"read user memory for {user_id}: {saved if saved else 'nothing saved'}"
        latest = state["messages"][-1].content
        new_language = detect_language_preference(latest)
        if new_language:
            user_memory.save(user_id, "language", new_language)
            saved = user_memory.get(user_id)
            note += f"; saved new preference language={new_language}"
        return {
            "user_language": saved.get("language"),
            "memory_note": note,
            "model_calls": 0,             # new user message -> reset the loop counter
            "loop_limit_reached": False,
        }

    def agent(state):
        prompt = [SystemMessage(content=build_system_prompt(state))] + state["messages"]
        response = model_with_tools.invoke(prompt)
        return {"messages": [response], "model_calls": state.get("model_calls", 0) + 1}

    def run_tools(state):
        """Execute every tool the model requested and return the results as ToolMessages."""
        request = state["messages"][-1]
        service_checks = dict(state.get("service_checks") or {})
        reference_ids = list(state.get("reference_ids") or [])
        tickets = list(state.get("tickets") or [])
        results = []

        for call in request.tool_calls:
            name, args = call["name"], call["args"]
            try:
                if name == "find_instructions":
                    result = tools.find_instructions(args.get("topic"))
                    if result["found"] and result["instruction_id"] not in reference_ids:
                        reference_ids.append(result["instruction_id"])
                elif name == "check_service_status":
                    result = tools.check_service_status(args.get("service"), state["status_profile"])
                    service_checks[result["service"]] = result
                elif name == "prepare_ticket":
                    category = tools.normalize_topic(args.get("category")) or args.get("category")
                    service = tools.HELPDESK["topic_services"].get(category)
                    checked = service_checks.get(service)
                    result = tools.prepare_ticket(
                        category=category,
                        summary=args.get("summary", ""),
                        steps_tried=args.get("steps_tried", []),
                        user_id=state["user_id"],
                        thread_id=state["thread_id"],
                        service_status=checked["status"] if checked else "not checked",
                        status_profile=state["status_profile"],
                        reference_ids=reference_ids,
                        ticket_dir=ticket_dir,
                    )
                    if result["created"]:
                        tickets.append(result["file"])
                else:
                    result = {"error": f"Unknown tool {name}"}
            except Exception as error:  # the model gets the error instead of the program crashing
                result = {"error": str(error)}
            results.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                       tool_call_id=call["id"], name=name))

        return {"messages": results, "service_checks": service_checks,
                "reference_ids": reference_ids, "tickets": tickets}

    def stop_at_limit(state):
        """Loop limit reached: answer every open tool request, then stop with a message."""
        request = state["messages"][-1]
        closing = [ToolMessage(content='{"error": "not executed: loop limit reached"}',
                               tool_call_id=call["id"], name=call["name"])
                   for call in request.tool_calls]
        closing.append(AIMessage(content="I stopped because the limit of model/tool steps for one message "
                                         "was reached. Please rephrase or ask a simpler question."))
        return {"messages": closing, "loop_limit_reached": True}

    # ---------------- conditional edge ----------------

    def after_agent(state):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "end"                      # the model answered -> finish
        if state.get("model_calls", 0) >= max_model_calls:
            return "limit"                    # the model still wants tools, but the limit is reached
        return "tools"                        # run the requested tools

    builder = StateGraph(HelpdeskState)
    builder.add_node("load_user_memory", load_user_memory)
    builder.add_node("agent", agent)
    builder.add_node("tools", run_tools)
    builder.add_node("stop_at_limit", stop_at_limit)

    builder.add_edge(START, "load_user_memory")
    builder.add_edge("load_user_memory", "agent")
    builder.add_conditional_edges("agent", after_agent,
                                  {"tools": "tools", "limit": "stop_at_limit", "end": END})
    builder.add_edge("tools", "agent")
    builder.add_edge("stop_at_limit", END)
    return builder.compile(checkpointer=checkpointer)


def checkpoint_config(user_id, thread_id):
    """The checkpointer key combines user and thread, so one user can never open another user's thread."""
    return {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}


def make_input(user_id, thread_id, status_profile, message):
    return {"messages": [HumanMessage(content=message)], "user_id": user_id,
            "thread_id": thread_id, "status_profile": status_profile}
