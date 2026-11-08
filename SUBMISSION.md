# HW5 — My LangGraph helpdesk

**Student:** Dilyara Kabysheva · **GitHub:** [dilyarakabysheva-jpg](https://github.com/dilyarakabysheva-jpg) · Narxoz University · CSS-4007 Artificial Intelligence
**Repository:** https://github.com/dilyarakabysheva-jpg/css4007-hw5

**Where the results come from**
- S01–S05 (Part 1): no model, logs in `outputs/S01.log` … `S05.log`.
- S06–S14 and P3-D1 (Parts 2–3): real runs with OpenAI **gpt-4o-mini** (`gpt-4o-mini-2024-07-18`) on 8 November 2026. Logs are in `outputs/`, ticket drafts in `outputs/tickets/`.
- `outputs/fake_model/` holds offline runs with a scripted fake model (not an LLM). They are used only for tests, not as results.

## Use of AI

There are two different kinds of AI in this project:

1. **gpt-4o-mini is part of the program.** In Parts 2–3 the assignment requires a model that decides which tools to call. All the results for S06–S14 come from this model.
2. **Claude Code (Anthropic) was used during development.** It wrote most of the code, the tests and the documentation, ran the scenarios, and fixed the problems found in the first real run (see the end of this file). Commits made with Claude end with `Co-Authored-By: Claude`.

## Part 1 — How the graph works

### My graph and state

```text
START → read_request ──(choose_next_node)──┬─ wifi_help     → END
                                           ├─ password_help → END
                                           └─ other_help    → END
```

Mermaid diagram, simplified from the output of `python part1.py --draw` (dashed arrows are the conditional edge):

```mermaid
graph TD;
  __start__ --> read_request;
  read_request -.-> wifi_help;
  read_request -.-> password_help;
  read_request -.-> other_help;
  wifi_help --> __end__;
  password_help --> __end__;
  other_help --> __end__;
```

State (`Part1State` in `helpdesk/part1_graph.py`):

| Field | Set by | Meaning |
| --- | --- | --- |
| `message` | the caller (input) | the user's original message; no node changes it |
| `category` | `read_request` | `wifi`, `password` or `other` (`both` in the changed graph) |
| `reply` | one help node | instructions from `data/helpdesk.json`, or a question |

| Node | Reads | Changes |
| --- | --- | --- |
| `read_request` | `message` | `category` |
| `wifi_help` | nothing from the state; it reads `helpdesk.json` → `instructions.wifi` | `reply` |
| `password_help` | nothing from the state; it reads `helpdesk.json` → `instructions.password` | `reply` |
| `other_help` | nothing | `reply` (asks: Wi-Fi or password?) |
| `choose_next_node` (conditional edge, not a node) | `category` | nothing; it returns the name of the next node |

**Answers to the four questions**

1. *What does each node read and change?* See the table above. Each node returns only the fields it changes, for example `{"category": "wifi"}`.
2. *Node update vs conditional edge.* A node does work and returns an **update** that LangGraph merges into the state. A conditional edge does not change the state. It is a function that reads the state and returns the **name of the next node**. In my graph `read_request` decides the category, and `choose_next_node` turns that category into a route.
3. *What does the next node receive?* The **whole current state** after the update has been merged. `wifi_help` could still read `message`, which it did not receive from `read_request`'s update. My traces show this: after each node, the state contains all fields so far.
4. *START, END, compile.* `START` is the entry point, so the first node to run is the one connected from `START` (`read_request`). `END` means the run stops and the final state is returned. `compile()` checks the structure (for example, that every edge points to a node that exists) and turns the builder into a runnable object with `invoke` and `stream`.

### Predictions before running

The routing rules: the `wifi` rule is checked first, then `password`, otherwise `other`. Case is ignored.

| Message | Predicted route | Actual route | What I learned |
| --- | --- | --- | --- |
| I cannot connect to Wi-Fi. | read_request → wifi_help | read_request → wifi_help (`outputs/S01.log`) | `wi-fi` matches after lower-casing |
| I forgot my PASSWORD. | read_request → password_help | read_request → password_help (`outputs/S02.log`) | Upper case works because I compare `message.lower()` |
| My laptop screen is broken. | read_request → other_help | read_request → other_help (`outputs/S03.log`) | Unsupported requests get a question back, not a guess |
| My Wi-Fi password does not work. | read_request → wifi_help | read_request → wifi_help (`outputs/S04.log`) | Both topics appear, but the **wifi rule is checked first**, so the password part is ignored |

All four predictions matched the actual runs.

**Both topics:** in S04 the state after `read_request` is `{'message': 'My Wi-Fi password does not work.', 'category': 'wifi'}`. The `if mentions_wifi(...)` branch comes before `elif mentions_password(...)`, so the first matching rule wins. The user may actually have a password problem, but they get Wi-Fi instructions.

### My change to the graph

```text
START → read_request ──┬─ wifi_help       → END
                       ├─ password_help   → END
                       ├─ other_help      → END
                       └─ clarify_problem → END   (new: Wi-Fi AND password)
```

```mermaid
graph TD;
  __start__ --> read_request;
  read_request -.-> clarify_problem;
  read_request -.-> wifi_help;
  read_request -.-> password_help;
  read_request -.-> other_help;
  clarify_problem --> __end__;
  wifi_help --> __end__;
  password_help --> __end__;
  other_help --> __end__;
```

`read_request_with_clarification` now checks "both topics" first and sets `category = "both"`. The conditional edge maps `both` to the new node `clarify_problem`.
New trace (`outputs/S05.log`, command `python part1.py "My Wi-Fi password does not work." --variant with_clarification`):

```text
Route: START -> read_request -> clarify_problem -> END
  1. read_request     update={'category': 'both'}
  2. clarify_problem  update={'reply': 'Your message mentions both Wi-Fi and a password. Is the problem connecting
                              to Wi-Fi, or recovering your account password? Which one should we solve first?'}
```

The difference is a **different path through the graph**: another node runs. The `wifi_help` node itself was not edited.

**What LangGraph controls vs what my functions decide.** LangGraph controls execution. It starts at `START`, runs nodes in edge order, merges each update into the state, calls the conditional edge function, follows the name it returns, and stops at `END`. My Python functions make every decision. `read_request` picks the category with keyword rules, and `choose_next_node` maps it to a node. There is no model in Part 1, so every route can be predicted from the code.

## Part 2 — Model and tools

### The loop

```text
START → load_user_memory → agent ──after_agent──┬─ "tools" → tools ─┐
                             ▲                   │                   │
                             └───────────────────┼───────────────────┘
                                                 ├─ "end"   → END            (no tool calls: final answer)
                                                 └─ "limit" → stop_at_limit → END
```

```mermaid
graph TD;
  __start__ --> load_user_memory --> agent;
  agent -.->|tool calls| tools;
  tools --> agent;
  agent -.->|no tool calls| __end__;
  agent -.->|model_calls >= 5| stop_at_limit --> __end__;
```

1. The model (node `agent`) receives the system prompt and the conversation. It returns either a normal answer or an `AIMessage` with `tool_calls`.
2. The conditional edge `after_agent` reads the last message.
3. The node `tools` (my function `run_tools`) runs the real Python functions from `helpdesk/tools.py` and appends one `ToolMessage` per call.
4. Control returns to `agent`, which sees the results and either requests more tools or answers.

**Loop limit.** `load_user_memory` resets `model_calls` to 0 for each new user message, and `agent` adds 1 each time it runs. In `after_agent`, if the model still asks for tools when `model_calls >= MAX_MODEL_CALLS` (5), the graph goes to `stop_at_limit`. That node answers the open tool calls with "not executed" and ends with an explanation. `test_loop_limit_stops_the_graph` checks this with a fake model that never stops asking for tools. LangGraph also has its own `recursion_limit`, but in the installed version (langgraph 1.2.20) the default is very large (10007 steps). So my `MAX_MODEL_CALLS` check is what actually stops the loop, and it gives a friendly message instead of an error.

**Answers**

1. *How does the graph decide between a tool and an answer?* The conditional edge `after_agent` checks whether the model's last message contains `tool_calls`. If it does (and the limit is not reached), the route is `tools`. Otherwise it is `END`.
2. *Who requests and who executes?* The **model** only *requests* a tool by returning a name and arguments in JSON. **My program**, the `tools` node, *executes* it. The model never runs code and never chooses `status_profile`, `user_id` or `thread_id`. My program adds those from the state.
3. *What changes in the state after a tool result?* `messages` gets the `ToolMessage`s. Depending on the tool, `service_checks` (last status per service), `reference_ids` (instruction ids used) and `tickets` (local files created) are updated too. `prepare_ticket` takes its `service_status` from `service_checks`, or uses `"not checked"`, so the model cannot invent a status.
4. *Difference from Part 1?* In Part 1 the route is fixed by my keyword rules and can be predicted. In Part 2 the next step is chosen by the model at run time, based on the message and on earlier tool results. The graph only provides the possible paths and the limit.

### Model and provider

- Real model: **OpenAI** through `langchain-openai` (`ChatOpenAI`, `temperature=0`). The model name comes from `MODEL_NAME` in `.env`.
  Model used for the real runs: `MODEL_NAME=gpt-4o-mini`; the API reported the version `gpt-4o-mini-2024-07-18`.
- Tests and offline runs: **FAKE MODEL** `fake-helpdesk-model (scripted rules, not an LLM)`. Every reply starts with `[FAKE MODEL]`.

### Direct tool calls (no model, no graph)

These calls are covered by `tests/test_tools.py`:

```python
find_instructions("Wi-Fi")              # -> {"found": True, "instruction_id": "GUIDE-WIFI", "steps": [...]}
check_service_status("wifi", "wifi_outage")            # -> {"status": "outage", ...}
check_service_status("password", "recovery_outage")    # -> service "account_recovery", status "outage"
prepare_ticket("wifi", "Campus-Demo will not connect", ["enabled Wi-Fi"],
               user_id="demo-user-a", thread_id="t1", ticket_dir=...)  # -> writes TKT-....json (draft)
prepare_ticket("wifi", "", [], ...)     # -> {"created": False, "missing": ["problem description", "steps already tried"]}
```

### A run that uses several tools (S06, second message)

Real run with gpt-4o-mini, from `outputs/S06.log` (shortened):

```text
1. load_user_memory   read user memory for demo-user-a: nothing saved
2. agent  MODEL REQUESTS TOOLS: check_service_status({"service": "wifi"})
3. tools  TOOL RESULT check_service_status: {"service": "wifi", "status": "operational",
          "message": "The demo Wi-Fi service has no reported outage.", "status_profile": "normal", ...}
4. agent  MODEL REQUESTS TOOLS: prepare_ticket({"category": "wifi", "summary": "Unable to connect to Campus-Demo network.",
                                                "steps_tried": ["Enabled Wi-Fi and forgot the saved network."]})
5. tools  TOOL RESULT prepare_ticket: {"created": true, "file": "outputs/tickets/TKT-20261108-203300-DC91.json", ...}
6. agent  MODEL ANSWER: The Wi-Fi service is operational, so I have prepared a ticket for you ...
          The ticket has been saved as outputs/tickets/TKT-20261108-203300-DC91.json.
```

The model requested `prepare_ticket` only **after** it saw the `operational` result. The ticket uses only the user's own steps.
The first message of S06 used `find_instructions` (GUIDE-WIFI), so the saved ticket has `reference_ids: ["GUIDE-WIFI"]` and `service_status: "operational"`. My code fills `service_status`, not the model.

### Same request, three snapshots (S06 / S07 / S08)

Real runs with gpt-4o-mini:

| Snapshot | `check_service_status` result | What the model did next | Log |
| --- | --- | --- | --- |
| `normal` | operational | called `prepare_ticket`; draft `outputs/tickets/TKT-20261108-203300-DC91.json` | `outputs/S06.log` |
| `wifi_outage` | outage | no ticket: "there is currently an outage with the Wi-Fi service across campus, so I cannot prepare a ticket for you", and said no repair time is given | `outputs/S07.log` |
| `status_unavailable` | unknown | no ticket: "Would you like me to prepare a draft ticket explaining that the status could not be checked?" | `outputs/S08.log` |

The request text is the same, and only the snapshot passed with `--profile` changes. Because the tool result changes, the next step changes too.

## Part 3 — Memory

**Two kinds of memory**

- **Conversation memory** comes from LangGraph's `SqliteSaver` checkpointer (`local-storage/checkpoints.sqlite`). After every node the whole state of the thread is saved: `messages` (all user messages, model answers, tool requests and results), `service_checks`, `reference_ids` and `tickets`. The current problem and the steps already tried are in these saved messages.
- **User memory** is `local-storage/user_memory.json`, saved per `user_id`, for example `{"demo-user-a": {"language": "Kazakh"}}`. The node `load_user_memory` reads it at the start of every turn. If the new message says "I prefer replies in X", the node saves X. The preference is learned from the message. It is not hard-coded and not read from `demo_users.json`.

**`thread_id` vs user identifier.** A `thread_id` identifies **one conversation**, and the checkpointer stores and loads state by it. A `user_id` identifies **a person**, who can have many threads. My checkpointer key is `"{user_id}:{thread_id}"` (`checkpoint_config`), so two users who use the same thread name still get separate conversations. User memory is keyed only by `user_id`, so it is shared across that user's threads and is never visible to another user. The model never chooses which user's data is loaded: `user_id` comes from the command line.

**Retrieval vs memory.** Looking up Wi-Fi instructions is **retrieval**. The answer comes from a fixed reference file (`helpdesk.json`) that is the same for everyone and does not change because of the conversation. Recalling the chosen language is **memory**. It is information the user gave in an earlier conversation, stored for that one user and read back later.

**Memory must survive a restart.** Both stores are files on disk. `run_scenarios.py` runs each part between `restart_process` events as a **new Python process**, and the logs show different pids. `test_memory_survives_a_real_process_restart` runs `chat.py` twice as separate processes.

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
All four are real runs with gpt-4o-mini:

| Situation | Command used | Log file | What happened |
| --- | --- | --- | --- |
| Refer to an earlier message | `python run_scenarios.py --only P3-D1` | `outputs/P3-D1.log` | Turn 1: `find_instructions` → GUIDE-WIFI steps (from the reference file). Turn 2, "I already tried that": 2 earlier messages were read from the checkpoint. The model treated "that" as the steps it had just given, so it did not repeat them. It called `check_service_status` (operational) and offered a ticket. |
| Close, restart and resume | `python run_scenarios.py --only S11` (process 1, then process 2) | `outputs/S11.log` | Process 1 saved `language=Kazakh` to user memory. Process 2 had a different pid and read 2 earlier messages from the SQLite checkpoint. It answered in Kazakh that we discussed a Wi-Fi problem, and that enabling Wi-Fi and reconnecting did not help. |
| Same user, new conversation | `python run_scenarios.py --only S12` (2 processes) | `outputs/S12.log` | Thread `preference-a2` read 0 earlier messages, but user memory returned `{'language': 'Kazakh'}`. The model answered in Kazakh that the user asked for Kazakh, and listed what it can help with. |
| Different user | `python run_scenarios.py --only S13` | `outputs/S13.log` | demo-user-b: `read user memory for demo-user-b: nothing saved`, 0 earlier messages. Reply in English: "Nothing has been recorded regarding a preferred language or a previous problem." |

## Part 4 — Running and checking my work

- Setup, run commands and Python version: [README.md](README.md).
- Exact package versions: [requirements.txt](requirements.txt). Python 3.11.1.
- Memory location and reset: README section 5 (`python reset_memory.py`).
- CI: [.github/workflows/tests.yml](.github/workflows/tests.yml) runs `pytest` on every push, with no API key.

| Test file | What it would catch if the code broke |
| --- | --- |
| `test_part1_routing.py` | a wrong rule order, case-sensitive matching, a missing `wi-fi` alias, `clarify_problem` not wired, the original message being overwritten |
| `test_tools.py` | wrong instruction ids, the wrong snapshot or the password→account_recovery mapping broken, tickets missing template fields, tickets created without details or with secrets |
| `test_user_memory.py` | a question about language mistaken for a new preference, preferences mixed between users, a preference not saved to disk |
| `test_agent_and_memory.py` | the tool loop not running tools, a ticket made during an outage or unknown status, a ticket the user never asked for, the loop never stopping, conversation not restored after reopening storage or after a real process restart, a new thread seeing old messages, one user seeing another user's preference |

Tests that run the graph use the **fake model**, so they test my program and not the intelligence of an LLM. The real-model demonstration is kept separate (`python run_scenarios.py` without `--fake-model`).

## Choices and limitations

- **Storage:** SQLite (`SqliteSaver`) for conversation checkpoints, because it is supplied, needs no server, survives restarts, and is one file. A small JSON file for user preferences, because it is simple and a person can open it and see what is remembered.
- **Preference detection** uses a regular expression (`prefer replies in X`, `reply in X`) with a list of known languages. Unusual wording like "talk to me in Kazakh" would be missed. A model could extract it instead, but then the model would be writing user memory.
- **Tool execution** is my own `run_tools` node instead of LangGraph's prebuilt `ToolNode`. This keeps "who executes the tool" visible and lets my program add `status_profile` and the checked service status.
- **Fake model** follows keyword rules. It is only for tests and offline demos and is not a replacement for the real LLM run.
- **Real LLM:** results depend on the model and are not fully deterministic, even at `temperature=0`. The system prompt states the rules, and two things are enforced in my code: the ticket's `service_status` is filled by the program, and `prepare_ticket` is refused if the user never mentioned a ticket.
- **Known imperfections in the final real run:**
  - In S14 the model added "I cannot prepare a ticket for this issue since it's not operational" although no ticket was requested. The status answer itself is correct (outage, no repair time invented).
  - In S09 the ticket summary is short ("Wi-Fi issue"), but it does not invent anything.

## Supplied scenarios — my actual results

All rows are real runs. S01–S05 use the Part 1 graph (no model). S06–S14 use gpt-4o-mini, through `python run_scenarios.py --only <ID>`, which starts a new process for each `restart_process`. Fake-model versions of S06–S14 are only in `outputs/fake_model/`.

| Scenario | Command or manual steps | Log file | What happened |
| --- | --- | --- | --- |
| S01 — Simple Wi-Fi route | `python part1.py "I cannot connect to Wi-Fi."` | `outputs/S01.log` | read_request (category=wifi) → wifi_help; GUIDE-WIFI steps |
| S02 — Case-insensitive password route | `python part1.py "I forgot my PASSWORD."` | `outputs/S02.log` | category=password → password_help; works because of `.lower()` |
| S03 — Unsupported request | `python part1.py "My laptop screen is broken."` | `outputs/S03.log` | category=other → other_help asks "Wi-Fi or a password?" |
| S04 — Two topics before the graph change | `python part1.py "My Wi-Fi password does not work." --variant original` | `outputs/S04.log` | category=wifi → wifi_help; the wifi rule is first, so the password part is lost |
| S05 — Two topics after adding clarification | `python part1.py "My Wi-Fi password does not work." --variant with_clarification` | `outputs/S05.log` | category=both → new node clarify_problem asks which problem first |
| S06 — Instructions then local ticket | `python run_scenarios.py --only S06` | `outputs/S06.log` | Turn 1: find_instructions → GUIDE-WIFI. Turn 2: check_service_status = operational → prepare_ticket with the user's own steps; reply names `outputs/tickets/TKT-20261108-203300-DC91.json` |
| S07 — The same request during a Wi-Fi outage | `python run_scenarios.py --only S07` | `outputs/S07.log` | check_service_status = outage → no ticket; says it cannot prepare the ticket because of the outage and that no repair time is given |
| S08 — Status source unavailable | `python run_scenarios.py --only S08` | `outputs/S08.log` | check_service_status = unknown → no ticket; asks whether to prepare a draft explaining that the status could not be checked |
| S09 — Missing ticket details | `python run_scenarios.py --only S09` | `outputs/S09.log` | Turn 1: no tool, asks "What problem are you having, and what have you already tried?". Turn 2: prepare_ticket with steps ["Enabled Wi-Fi", "Tried reconnecting"]; ticket service_status "not checked" |
| S10 — Use the current conversation | `python run_scenarios.py --only S10` | `outputs/S10.log` | Turn 1: find_instructions. Turn 2: 2 earlier messages from the checkpoint; listed "enabled Wi-Fi and forgot the saved network" as already tried |
| S11 — Persist and recall after restart | `python run_scenarios.py --only S11` (2 processes) | `outputs/S11.log` | Different pid after restart; problem and steps recalled from the SQLite checkpoint; reply in Kazakh from user memory |
| S12 — Preference across threads without replay | `python run_scenarios.py --only S12` (2 processes) | `outputs/S12.log` | New thread: 0 earlier messages; Kazakh from user memory; answered in Kazakh |
| S13 — Do not share memory between users | `python run_scenarios.py --only S13` | `outputs/S13.log` | demo-user-b: nothing in user memory, 0 earlier messages; "Nothing has been recorded ..." |
| S14 — Password recovery outage | `python run_scenarios.py --only S14` | `outputs/S14.log` | check_service_status("account_recovery") = outage under recovery_outage; no repair time invented (extra sentence about a ticket, see limitations) |

**What changed after the first real run.** The first gpt-4o-mini run (same day; its logs were replaced by the final run) had three problems:
- In **S10** and **S11** the model made a ticket although the user never asked for one.
- In **S11** the reply named a ticket file that does not exist (`wifi_issue_draft.txt`).
- In **P3-D1** the model answered "I cannot connect to Wi-Fi." with a question instead of the instructions, so "I already tried that" could not be understood.

Two fixes were made:
1. The system prompt became stricter: give instructions for a reported problem, prepare a ticket only when asked, and copy the file name exactly from the tool result.
2. The `tools` node now refuses `prepare_ticket` if no user message mentions a ticket. Test: `test_no_ticket_unless_user_asked`.

After that, all model scenarios (S06–S14, P3-D1) were run again, because they all use the same system prompt. The table above shows this final run.
