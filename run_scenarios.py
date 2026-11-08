"""Run the supplied scenarios S01-S14 (data/scenarios.json) and save one log per scenario.

    python run_scenarios.py                 # real model (needs OPENAI_API_KEY and MODEL_NAME in .env)
    python run_scenarios.py --fake-model    # scripted FAKE model, logs go to outputs/fake_model/
    python run_scenarios.py --only S06 S11  # only some scenarios

How restarts work: the events of a scenario are split at every "restart_process"
event. Each part runs in a NEW Python process (subprocess) that opens the same
storage folder, so memory really has to come from disk. The log shows each
process id (pid) so you can see that the process changed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from helpdesk.data import PROJECT_DIR, load_scenarios
from part1 import format_part1_trace

# Extra Part 3 demonstration that is required by the brief but is not in scenarios.json:
# "A user reports a Wi-Fi problem, then says 'I already tried that.'"
EXTRA_SCENARIOS = [
    {
        "id": "P3-D1",
        "part": 3,
        "title": "Refer to an earlier message ('I already tried that')",
        "status_profile": "normal",
        "reset_before": True,
        "events": [
            {"action": "message", "user_id": "demo-user-a", "thread_id": "tried-that",
             "message": "I cannot connect to Wi-Fi."},
            {"action": "message", "user_id": "demo-user-a", "thread_id": "tried-that",
             "message": "I already tried that."},
        ],
    }
]


def all_scenarios():
    return load_scenarios() + EXTRA_SCENARIOS


def split_at_restarts(events):
    segments = [[]]
    for event in events:
        if event["action"] == "restart_process":
            segments.append([])
        else:
            segments[-1].append(event)
    return segments


def mode_paths(fake):
    if fake:
        return (PROJECT_DIR / "outputs" / "fake_model", PROJECT_DIR / "local-storage" / "scenarios-fake",
                PROJECT_DIR / "outputs" / "fake_model" / "tickets")
    return (PROJECT_DIR / "outputs", PROJECT_DIR / "local-storage" / "scenarios",
            PROJECT_DIR / "outputs" / "tickets")


# ---------------- child process: runs one segment ----------------

def run_child(scenario_id, segment_index, result_file, fake):
    from helpdesk.session import HelpdeskSession, format_trace, make_model

    scenario = next(s for s in all_scenarios() if s["id"] == scenario_id)
    events = split_at_restarts(scenario["events"])[segment_index]
    model, model_label = make_model(fake=fake)
    session = HelpdeskSession(model)
    turns = []
    try:
        for event in events:
            trace = session.send(event["user_id"], event["thread_id"], scenario["status_profile"],
                                 event["message"])
            turns.append(format_trace(trace))
    finally:
        session.close()
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "model": model_label, "turns": turns}, f, ensure_ascii=False)


# ---------------- parent process ----------------

def run_part1_scenario(scenario, out_dir):
    variant = scenario.get("graph_variant", "original")
    lines = header(scenario, "none (Part 1 has no model)")
    for event in scenario["events"]:
        command = f'python part1.py "{event["message"]}" --variant {variant}'
        text, _ = format_part1_trace(event["message"], variant)
        lines += [f"Command: {command}", f"user_id={event['user_id']}  thread_id={event['thread_id']}", text]
    write_log(out_dir, scenario["id"], lines)


def run_model_scenario(scenario, out_dir, storage, ticket_dir, fake):
    if scenario.get("reset_before") and storage.exists():
        shutil.rmtree(storage)          # reset conversation AND user memory before this scenario
    env = dict(os.environ, HELPDESK_STORAGE_DIR=str(storage), HELPDESK_TICKET_DIR=str(ticket_dir))
    segments = split_at_restarts(scenario["events"])
    lines = None
    for index, _ in enumerate(segments):
        with tempfile.TemporaryDirectory() as tmp:
            result_file = Path(tmp) / "result.json"
            command = [sys.executable, "run_scenarios.py", "--child", scenario["id"],
                       "--segment", str(index), "--result-file", str(result_file)]
            if fake:
                command.append("--fake-model")
            completed = subprocess.run(command, cwd=PROJECT_DIR, env=env, capture_output=True,
                                       text=True, encoding="utf-8")
            if completed.returncode != 0:
                print(completed.stdout, completed.stderr)
                raise SystemExit(f"{scenario['id']} segment {index} failed")
            result = json.loads(result_file.read_text(encoding="utf-8"))
        if lines is None:
            lines = header(scenario, result["model"])
            lines.append(f"Storage folder (reset before scenario: {scenario.get('reset_before')}): "
                         f"{storage.relative_to(PROJECT_DIR).as_posix()}")
        if index > 0:
            lines.append("=== restart_process: the previous process exited; a NEW process opens the same storage ===")
        lines.append(f"--- Process {index + 1} (pid {result['pid']}) ---")
        lines.append("Command: python run_scenarios.py --child " + scenario["id"] + f" --segment {index}"
                     + (" --fake-model" if fake else ""))
        for turn in result["turns"]:
            lines += ["", turn]
        lines.append("")
    write_log(out_dir, scenario["id"], lines)


def header(scenario, model_label):
    lines = [
        f"Scenario: {scenario['id']} - {scenario['title']}",
        f"Part: {scenario['part']}   graph_variant: {scenario.get('graph_variant', 'default')}",
        f"Status snapshot: {scenario['status_profile']}",
        f"Model: {model_label}",
        f"Run at: {datetime.now().isoformat(timespec='seconds')}",
    ]
    if model_label.startswith("FAKE"):
        lines.append("NOTE: this log was produced by the scripted FAKE model. It is NOT a real LLM run.")
    return lines + [""]


def write_log(out_dir, scenario_id, lines):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario_id}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(PROJECT_DIR).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="scenario ids to run, e.g. S01 S06")
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--child")
    parser.add_argument("--segment", type=int)
    parser.add_argument("--result-file")
    args = parser.parse_args()

    if args.child:
        run_child(args.child, args.segment, args.result_file, args.fake_model)
        return

    out_dir, storage, ticket_dir = mode_paths(args.fake_model)
    for scenario in all_scenarios():
        if args.only and scenario["id"] not in args.only:
            continue
        if scenario["part"] == 1:
            run_part1_scenario(scenario, PROJECT_DIR / "outputs")
        else:
            run_model_scenario(scenario, out_dir, storage, ticket_dir, args.fake_model)


if __name__ == "__main__":
    main()
