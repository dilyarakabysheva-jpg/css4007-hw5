"""Load the fixture JSON files supplied in the data/ folder."""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_json(file_name):
    with open(DATA_DIR / file_name, encoding="utf-8") as f:
        return json.load(f)


def load_helpdesk():
    return load_json("helpdesk.json")


def load_service_status():
    return load_json("service_status.json")


def load_ticket_template():
    return load_json("ticket_template.json")["ticket"]


def load_scenarios():
    return load_json("scenarios.json")["scenarios"]


def format_instructions(entry):
    """Turn one instructions entry from helpdesk.json into readable text."""
    lines = [f"{entry['title']} ({entry['instruction_id']}):"]
    for number, step in enumerate(entry["steps"], start=1):
        lines.append(f"  {number}. {step}")
    return "\n".join(lines)
