"""Part 1: a small helpdesk graph WITHOUT an LLM.

Original graph:
    START -> read_request -> wifi_help | password_help | other_help -> END

Changed graph (with_clarification=True) adds one more branch:
    read_request -> clarify_problem -> END   (when Wi-Fi AND password are mentioned)

The routing decision is made by plain Python functions (read_request and the
router). LangGraph only runs the nodes in the order given by the edges.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from helpdesk.data import format_instructions, load_helpdesk

HELPDESK = load_helpdesk()


class Part1State(TypedDict, total=False):
    message: str    # the user's original message (never changed)
    category: str   # set by read_request: wifi / password / both / other
    reply: str      # set by one of the help nodes


def mentions_wifi(text):
    text = text.lower()
    return any(alias in text for alias in HELPDESK["topic_aliases"]["wifi"])


def mentions_password(text):
    text = text.lower()
    return any(alias in text for alias in HELPDESK["topic_aliases"]["password"])


# ----- nodes: each one reads the state and returns ONLY the fields it changes -----

def read_request(state):
    """Original rules: wifi is checked first, then password, otherwise other."""
    message = state["message"]
    if mentions_wifi(message):
        category = "wifi"
    elif mentions_password(message):
        category = "password"
    else:
        category = "other"
    return {"category": category}


def read_request_with_clarification(state):
    """Changed rules: if both topics appear, the category becomes 'both'."""
    message = state["message"]
    if mentions_wifi(message) and mentions_password(message):
        category = "both"
    elif mentions_wifi(message):
        category = "wifi"
    elif mentions_password(message):
        category = "password"
    else:
        category = "other"
    return {"category": category}


def wifi_help(state):
    return {"reply": format_instructions(HELPDESK["instructions"]["wifi"])}


def password_help(state):
    return {"reply": format_instructions(HELPDESK["instructions"]["password"])}


def other_help(state):
    return {"reply": "I can only help with Wi-Fi or password problems. "
                     "Does your problem concern Wi-Fi or a password?"}


def clarify_problem(state):
    return {"reply": "Your message mentions both Wi-Fi and a password. "
                     "Is the problem connecting to Wi-Fi, or recovering your account password? "
                     "Which one should we solve first?"}


# ----- conditional edge: reads the state and returns the NAME of the next node -----

def choose_next_node(state):
    return {
        "wifi": "wifi_help",
        "password": "password_help",
        "both": "clarify_problem",
        "other": "other_help",
    }[state["category"]]


def build_part1_graph(with_clarification=False):
    builder = StateGraph(Part1State)
    if with_clarification:
        builder.add_node("read_request", read_request_with_clarification)
    else:
        builder.add_node("read_request", read_request)
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)

    targets = ["wifi_help", "password_help", "other_help"]
    if with_clarification:
        builder.add_node("clarify_problem", clarify_problem)
        targets.append("clarify_problem")

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", choose_next_node, targets)
    for node in targets:
        builder.add_edge(node, END)
    return builder.compile()


def run_with_trace(message, with_clarification=False):
    """Run the graph and record the node order and the full state after each node."""
    graph = build_part1_graph(with_clarification)
    state = {"message": message}
    steps = []
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node_name, update in chunk.items():
            state.update(update)
            steps.append({"node": node_name, "update": dict(update), "state_after": dict(state)})
    return {"route": [step["node"] for step in steps], "steps": steps, "final_state": state}
