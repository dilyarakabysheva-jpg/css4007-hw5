# HW5 — University IT helpdesk with LangGraph

CSS-4007 Artificial Intelligence · Narxoz University · Dilyara Kabysheva ([dilyarakabysheva-jpg](https://github.com/dilyarakabysheva-jpg))

A terminal helpdesk for a practice university IT service. It uses only the fictional data in `data/` and never sends real tickets.
The assignment is in [ASSIGNMENT.md](ASSIGNMENT.md). Answers, diagrams and results are in [SUBMISSION.md](SUBMISSION.md).

Python 3.11.1 (Windows 11). CI runs on Python 3.11 (Ubuntu).

## Files

| File | What it does |
| --- | --- |
| `helpdesk/part1_graph.py` | Part 1: graph with keyword routing, no LLM |
| `helpdesk/tools.py` | the three tools: `find_instructions`, `check_service_status`, `prepare_ticket` |
| `helpdesk/agent_graph.py` | Parts 2–3: model + tools loop, loop limit, user memory node |
| `helpdesk/user_memory.py` | saves the preferred language per user (JSON file) |
| `helpdesk/session.py` | opens the storage, picks the model, prints the trace |
| `helpdesk/fake_model.py` | scripted fake model for tests (not an LLM; replies start with `[FAKE MODEL]`) |
| `part1.py`, `chat.py` | command lines for Part 1 and Parts 2–3 |
| `run_scenarios.py` | runs S01–S14 (+ extra demo P3-D1) and writes logs to `outputs/` |
| `reset_memory.py` | deletes saved memory |
| `tests/`, `.github/workflows/tests.yml` | tests and GitHub Actions |

## Install

```bash
git clone https://github.com/dilyarakabysheva-jpg/css4007-hw5.git
cd css4007-hw5
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Parts 2–3 with a real model, copy `.env.example` to `.env` and set:

```
OPENAI_API_KEY=your key
MODEL_NAME=gpt-4o-mini
```

`.env` is in `.gitignore`. Part 1, the tests and `--fake-model` runs need no key.

## Run

```bash
python part1.py "I cannot connect to Wi-Fi."
python part1.py "My Wi-Fi password does not work." --variant with_clarification
python part1.py --draw                                              # print both graphs

python chat.py --user demo-user-a --thread t1 --profile normal    # interactive, type quit to exit
python chat.py --user demo-user-a --thread t1 --profile wifi_outage --message "Is Wi-Fi down?"
```

`--profile` picks a snapshot from `data/service_status.json` (`normal`, `wifi_outage`, `recovery_outage`, `status_unavailable`).
`--fake-model` runs without a key. `--quiet` prints only the replies.

All scenarios:

```bash
python run_scenarios.py                 # real model -> outputs/S01.log ... S14.log, P3-D1.log, outputs/tickets/
python run_scenarios.py --fake-model    # no key -> outputs/fake_model/
python run_scenarios.py --only S06 S07  # only some scenarios
```

For every `restart_process` event, the runner stops the Python process and starts a new one on the same storage. Each log shows the process ids.

## Memory files

| What | Where |
| --- | --- |
| Conversation memory (LangGraph SQLite checkpointer, key `user_id:thread_id`) | `local-storage/checkpoints.sqlite` |
| User memory (preferred language per `user_id`) | `local-storage/user_memory.json` |
| Storage used by `run_scenarios.py` | `local-storage/scenarios/`, `local-storage/scenarios-fake/` |

`local-storage/` is ignored by git. To start again, run `python reset_memory.py` or delete the folder. The scenario runner clears its own folder before every scenario with `reset_before: true`.

## Tests

```bash
python -m pytest -v
```

The tests check routing, each tool on its own, the tool loop and its limit, the ticket checks, memory after reopening storage and after a real process restart, and that users and threads are kept apart. Graph tests use the fake model, so they need no API key. GitHub Actions runs them on every push.
