"""Part 2: the three helpdesk tools.

They are ordinary Python functions, so each one can be called directly
(for example in tests) without the model or the graph.
The graph's "tools" node calls these same functions when the model asks for them.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from helpdesk.data import PROJECT_DIR, load_helpdesk, load_service_status, load_ticket_template

HELPDESK = load_helpdesk()
SERVICE_STATUS = load_service_status()

DEFAULT_TICKET_DIR = os.getenv("HELPDESK_TICKET_DIR", str(PROJECT_DIR / "outputs" / "tickets"))

# Words that suggest secret information that must never be stored in a ticket.
SECRET_WORDS = ["password is", "password:", "recovery code", "api key", "api_key"]


def normalize_topic(topic):
    """Map 'Wi-Fi', 'wifi', 'password' ... to the keys used in helpdesk.json."""
    text = (topic or "").strip().lower()
    for key, aliases in HELPDESK["topic_aliases"].items():
        if text == key or text in aliases:
            return key
    if "recovery" in text or "account" in text:
        return "password"
    return None


def find_instructions(topic):
    """Look up Wi-Fi or password instructions in data/helpdesk.json."""
    key = normalize_topic(topic)
    if key is None:
        return {"found": False, "topic": topic,
                "message": "No instructions for this topic. Supported topics: wifi, password."}
    entry = HELPDESK["instructions"][key]
    return {
        "found": True,
        "topic": key,
        "instruction_id": entry["instruction_id"],
        "title": entry["title"],
        "steps": entry["steps"],
        "source": "data/helpdesk.json",
    }


def service_for(name):
    """Accept a service name (wifi, account_recovery) or a topic (password)."""
    text = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("wifi", "wi_fi"):
        return "wifi"
    if text in ("account_recovery", "password", "password_recovery", "recovery"):
        return "account_recovery"
    return None


def check_service_status(service, status_profile):
    """Look up a service in the selected snapshot of data/service_status.json.

    status_profile is chosen by the application (normal, wifi_outage,
    recovery_outage, status_unavailable), never by the model.
    """
    if status_profile not in SERVICE_STATUS["profiles"]:
        raise ValueError(f"Unknown status profile: {status_profile}")
    service_key = service_for(service)
    if service_key is None:
        return {"service": service, "status": "unknown", "status_profile": status_profile,
                "message": "This service is not in the status data. Known services: wifi, account_recovery."}
    entry = SERVICE_STATUS["profiles"][status_profile][service_key]
    return {
        "service": service_key,
        "status": entry["status"],   # operational / outage / unknown
        "message": entry["message"],
        "status_profile": status_profile,
        "source": "data/service_status.json",
    }


def make_ticket_id():
    return "TKT-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4].upper()


def prepare_ticket(category, summary, steps_tried, user_id, thread_id,
                   service_status="not checked", status_profile=None,
                   reference_ids=None, ticket_dir=None):
    """Save a LOCAL ticket draft (JSON file). Nothing is sent anywhere.

    Required information (helpdesk.json ticket_policy): a problem description
    and the steps already tried. If something is missing, no file is written and
    the result says what to ask the user.
    """
    if isinstance(steps_tried, str):
        steps_tried = [steps_tried]
    steps_tried = [step.strip() for step in (steps_tried or []) if step and step.strip()]

    missing = []
    if not summary or not summary.strip():
        missing.append("problem description")
    if not steps_tried:
        missing.append("steps already tried")
    if missing:
        return {"created": False, "missing": missing,
                "message": "Ask the user for: " + ", ".join(missing) + ". Do not invent them."}

    all_text = (summary + " " + " ".join(steps_tried)).lower()
    if any(word in all_text for word in SECRET_WORDS):
        return {"created": False, "missing": [],
                "message": "The ticket text looks like it contains a secret. "
                           "Never include passwords, recovery codes or API keys."}

    ticket = load_ticket_template()
    ticket["ticket_id"] = make_ticket_id()
    ticket["status"] = "draft"
    ticket["user_id"] = user_id
    ticket["thread_id"] = thread_id
    ticket["category"] = normalize_topic(category) or category
    ticket["summary"] = summary.strip()
    ticket["steps_tried"] = steps_tried
    ticket["service_status"] = service_status
    ticket["status_profile"] = status_profile
    ticket["reference_ids"] = list(reference_ids or [])

    folder = Path(ticket_dir or DEFAULT_TICKET_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{ticket['ticket_id']}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2, ensure_ascii=False)

    try:  # show a short path such as outputs/tickets/TKT-....json when possible
        shown_path = file_path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        shown_path = str(file_path)
    return {"created": True, "ticket_id": ticket["ticket_id"], "file": shown_path,
            "ticket": ticket, "message": "Local draft saved. Nothing was sent to a real helpdesk."}


# What the MODEL sees: name, description and the arguments it may fill in.
# status_profile, user_id, thread_id and service_status are added by our program.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_instructions",
            "description": "Look up the local helpdesk instructions for a topic.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "enum": ["wifi", "password"]}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_service_status",
            "description": "Check whether a service is operational, has an outage, or its status is unknown. "
                           "Use 'wifi' for Wi-Fi and 'account_recovery' for password recovery.",
            "parameters": {
                "type": "object",
                "properties": {"service": {"type": "string", "enum": ["wifi", "account_recovery"]}},
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_ticket",
            "description": "Save a LOCAL support ticket draft (nothing is sent). Only use the user's own words "
                           "for the summary and the steps already tried. The service status is filled in "
                           "automatically from earlier check_service_status results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["wifi", "password"]},
                    "summary": {"type": "string", "description": "Short description of the problem."},
                    "steps_tried": {"type": "array", "items": {"type": "string"},
                                    "description": "Steps the user said they already tried."},
                },
                "required": ["category", "summary", "steps_tried"],
            },
        },
    },
]
