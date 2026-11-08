# HW5 — University IT helpdesk with LangGraph

CSS-4007 Artificial Intelligence · Narxoz University · Dilyara Kabysheva ([dilyarakabysheva](https://github.com/dilyarakabysheva))

This is a terminal helpdesk for a practice university IT service. It uses only fictional data from `data/`, and no real tickets are sent.
The original assignment brief is in [ASSIGNMENT.md](ASSIGNMENT.md). My answers and results are in [SUBMISSION.md](SUBMISSION.md).

**Python version:** 3.11.1 (Windows 11). The GitHub Actions workflow is set up for Python 3.11 on Ubuntu.

## Project layout

| File | What it contains |
| --- | --- |
| `helpdesk/part1_graph.py` | Part 1: rule-based graph, no LLM (original and `with_clarification` versions) |
| `helpdesk/tools.py` | Part 2: `find_instructions`, `check_service_status`, `prepare_ticket` and the tool schemas the model sees |
| `helpdesk/agent_graph.py` | Parts 2–3: model + tools loop with a loop limit, and the user-memory node |
| `helpdesk/user_memory.py` | Part 3: user preference store (JSON file per storage folder) |
| `helpdesk/session.py` | Opens the storage (SQLite checkpointer + user memory), picks the model and formats traces |
| `helpdesk/fake_model.py` | **Fake** scripted model for tests and offline runs. It is not an LLM, and every reply starts with `[FAKE MODEL]` |
| `part1.py` | Command line for Part 1 |
| `chat.py` | Command line for Parts 2–3 (user id, thread id, status snapshot) |
| `run_scenarios.py` | Runs S01–S14 from `data/scenarios.json` (plus extra demo `P3-D1`) and writes logs to `outputs/` |
| `reset_memory.py` | Deletes saved memory |
| `tests/` | pytest tests (no API key needed) |
| `.github/workflows/tests.yml` | GitHub Actions CI |

## 1. Install

Get the code:

```bash
git clone https://github.com/dilyarakabysheva/css4007-hw5.git
```
```bash
cd css4007-hw5
```

Then, in Windows PowerShell:

```bash
python -m venv .venv
```
```bash
.venv\Scripts\Activate.ps1
```
```bash
python -m pip install -r requirements.txt
```

macOS/Linux: use `source .venv/bin/activate` instead of the second command.

## 2. Configure the model (only for Parts 2–3 with a real LLM)

Copy `.env.example` to `.env`, then fill in:

```
OPENAI_API_KEY=sk-...your key...
MODEL_NAME=gpt-4o-mini
```

Any tool-capable OpenAI chat model works. `.env` is listed in `.gitignore`, so never commit it.
Part 1, the tests and `--fake-model` runs need no key.

## 3. Run

Part 1 (no model):

```bash
python part1.py "I cannot connect to Wi-Fi."
```
```bash
python part1.py "My Wi-Fi password does not work." --variant with_clarification
```
```bash
python part1.py --draw
```

Parts 2–3, interactive chat (type `quit` to exit):

```bash
python chat.py --user demo-user-a --thread t1 --profile normal
```

Send messages without typing them interactively:

```bash
python chat.py --user demo-user-a --thread t1 --profile wifi_outage --message "Check whether Wi-Fi is down. If it is operating, prepare a ticket. I already enabled Wi-Fi and reconnected, with no success."
```

`--profile` selects a snapshot from `data/service_status.json`: `normal`, `wifi_outage`, `recovery_outage` or `status_unavailable`.
Add `--fake-model` to use the scripted fake model without a key, and `--quiet` to print only the replies.

## 4. Run all supplied scenarios

```bash
python run_scenarios.py
```

This uses the real model. It writes `outputs/S01.log` … `outputs/S14.log` and `outputs/P3-D1.log`, and saves ticket drafts to `outputs/tickets/`.

```bash
python run_scenarios.py --fake-model
```

This runs without a key. Part 2–3 logs and tickets go to `outputs/fake_model/` and are marked FAKE.
Part 1 logs (S01–S05) never use a model, so they always go to `outputs/`.

Run only some scenarios with `--only S06 S07 S08`.
For each `restart_process` event, the runner ends the current Python process and starts a new one on the same storage. Each log shows the process ids.

## 5. Where memory is saved and how to clear it

| What | File |
| --- | --- |
| Conversation memory (LangGraph SQLite checkpointer, one thread per `user_id:thread_id`) | `local-storage/checkpoints.sqlite` |
| User memory (saved preferences per `user_id`) | `local-storage/user_memory.json` |
| Scenario runner storage | `local-storage/scenarios/` (real), `local-storage/scenarios-fake/` (fake) |

You can change the folder with `HELPDESK_STORAGE_DIR` in `.env`. `local-storage/` is ignored by git.
To start again:

```bash
python reset_memory.py
```

You can also delete the `local-storage` folder by hand. The scenario runner resets its own folder before each scenario that has `reset_before: true`.

## 6. Tests and CI

```bash
python -m pytest -v
```

The tests cover routing, the three tools called directly, the tool loop and its limit, memory after reopening storage, memory after a real process restart (`chat.py` run twice), and separation between users and threads.
Graph tests use the **fake model** (`helpdesk/fake_model.py`), so they need no API key. They check the program's wiring, not the quality of a real LLM.
GitHub Actions (`.github/workflows/tests.yml`) runs these tests on every push.
