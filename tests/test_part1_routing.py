"""Part 1 routing tests (no model). They catch broken keyword rules, a wrong rule
order, case-sensitivity bugs, and a missing or mis-wired clarify_problem node."""

import pytest

from helpdesk.part1_graph import run_with_trace


@pytest.mark.parametrize("message, expected_node", [
    ("I cannot connect to Wi-Fi.", "wifi_help"),
    ("I forgot my PASSWORD.", "password_help"),          # uppercase must still work
    ("My laptop screen is broken.", "other_help"),
    ("My Wi-Fi password does not work.", "wifi_help"),   # original graph: the wifi rule is checked first
    ("my wifi is slow", "wifi_help"),                     # 'wifi' without a hyphen
])
def test_original_graph_routes(message, expected_node):
    result = run_with_trace(message)
    assert result["route"] == ["read_request", expected_node]
    assert result["final_state"]["message"] == message   # original message is kept


def test_help_nodes_return_instructions_from_helpdesk_json():
    assert "GUIDE-WIFI" in run_with_trace("wifi")["final_state"]["reply"]
    assert "GUIDE-PASSWORD" in run_with_trace("password")["final_state"]["reply"]


def test_other_help_asks_wifi_or_password():
    reply = run_with_trace("printer jam")["final_state"]["reply"]
    assert "Wi-Fi or a password" in reply


def test_both_topics_go_to_clarify_problem_in_changed_graph():
    result = run_with_trace("My Wi-Fi password does not work.", with_clarification=True)
    assert result["route"] == ["read_request", "clarify_problem"]
    assert result["final_state"]["category"] == "both"
    assert "?" in result["final_state"]["reply"]


def test_changed_graph_keeps_single_topic_routes():
    assert run_with_trace("I forgot my PASSWORD.", with_clarification=True)["route"][-1] == "password_help"
    assert run_with_trace("I cannot connect to Wi-Fi.", with_clarification=True)["route"][-1] == "wifi_help"
