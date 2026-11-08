"""Direct tool tests (no model, no graph). They catch wrong data lookups, a wrong
snapshot or service mapping, and tickets that invent details or miss template fields."""

import json

import pytest

from helpdesk import tools
from helpdesk.data import load_ticket_template


def test_find_instructions_wifi_and_password():
    assert tools.find_instructions("Wi-Fi")["instruction_id"] == "GUIDE-WIFI"
    assert tools.find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"


def test_find_instructions_unknown_topic():
    assert tools.find_instructions("printer")["found"] is False


@pytest.mark.parametrize("profile, service, expected", [
    ("normal", "wifi", "operational"),
    ("wifi_outage", "wifi", "outage"),
    ("status_unavailable", "wifi", "unknown"),
    ("recovery_outage", "account_recovery", "outage"),
    ("recovery_outage", "password", "outage"),     # password topic -> account_recovery service
    ("recovery_outage", "wifi", "operational"),
])
def test_check_service_status_uses_selected_snapshot(profile, service, expected):
    assert tools.check_service_status(service, profile)["status"] == expected


def test_check_service_status_rejects_unknown_profile():
    with pytest.raises(ValueError):
        tools.check_service_status("wifi", "no_such_profile")


def test_prepare_ticket_writes_local_draft_with_template_fields(tmp_path):
    result = tools.prepare_ticket("wifi", "Campus-Demo will not connect", ["enabled Wi-Fi"],
                                  user_id="demo-user-a", thread_id="t1", service_status="operational",
                                  status_profile="normal", reference_ids=["GUIDE-WIFI"], ticket_dir=tmp_path)
    assert result["created"] is True
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert set(saved) == set(load_ticket_template())          # same fields as data/ticket_template.json
    assert saved["status"] == "draft"
    assert saved["steps_tried"] == ["enabled Wi-Fi"]           # exactly the user's steps
    assert saved["ticket_id"].startswith("TKT-")


def test_prepare_ticket_refuses_missing_details(tmp_path):
    result = tools.prepare_ticket("wifi", "", [], user_id="demo-user-a", thread_id="t1", ticket_dir=tmp_path)
    assert result["created"] is False
    assert result["missing"] == ["problem description", "steps already tried"]
    assert list(tmp_path.iterdir()) == []                     # no file was written


def test_prepare_ticket_refuses_secrets(tmp_path):
    result = tools.prepare_ticket("password", "my password is hunter2", ["tried"],
                                  user_id="demo-user-a", thread_id="t1", ticket_dir=tmp_path)
    assert result["created"] is False
