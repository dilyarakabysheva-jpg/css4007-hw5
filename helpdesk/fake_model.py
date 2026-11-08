"""A FAKE model for tests and offline demonstrations. It is NOT a real LLM.

It follows simple keyword rules written by hand, so tests and CI can run
without an API key. It has the same interface the graph needs from a chat
model: bind_tools(...) and invoke(messages) -> AIMessage.
Every reply it writes starts with "[FAKE MODEL]" so it is never confused with a real LLM.
"""

import json
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

LABEL = "[FAKE MODEL] "


def find_topic(text):
    text = text.lower()
    if "wifi" in text or "wi-fi" in text:
        return "wifi"
    if "password" in text or "recovery" in text:
        return "password"
    return None


def extract_steps(text):
    """Take the user's own words describing what they tried, e.g.
    'I enabled Wi-Fi and forgot the saved network, but ...' -> ['I enabled Wi-Fi', 'forgot the saved network']."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lower = sentence.lower()
        if any(word in lower for word in ["enabled", "tried", "forgot the", "reconnect"]):
            part = re.split(r",?\s+(?:but|with no)\b", sentence)[0].strip(" .")
            part = re.sub(r"^(my issue is wi-fi\.\s*|the issue is wi-fi\.\s*)", "", part, flags=re.I)
            return [step.strip() for step in re.split(r"\s+and\s+", part) if step.strip()]
    return []


def extract_problem(text, topic):
    match = re.search(r"\bbut\s+([^.]+)", text, flags=re.I)
    if match:
        return f"{'Wi-Fi' if topic == 'wifi' else 'Password'} problem: {match.group(1).strip()}"
    return None


def tool_call(name, args):
    return {"name": name, "args": args, "id": "call_" + uuid.uuid4().hex[:8], "type": "tool_call"}


class FakeHelpdeskModel:
    name = "fake-helpdesk-model (scripted rules, not an LLM)"

    def bind_tools(self, tool_schemas):
        return self

    def invoke(self, messages):
        system = messages[0].content if isinstance(messages[0], SystemMessage) else ""
        language = re.search(r"Saved preferred reply language for this user: (\w+)", system)
        self.language = language.group(1) if language else None
        history = [m for m in messages if not isinstance(m, SystemMessage)]

        # The last user message and everything after it belong to the current turn.
        last_human_index = max(i for i, m in enumerate(history) if isinstance(m, HumanMessage))
        user_text = history[last_human_index].content
        earlier = history[:last_human_index]
        tool_results = [m for m in history[last_human_index + 1:] if isinstance(m, ToolMessage)]

        if tool_results:
            return self.answer_after_tools(user_text, tool_results, system)
        return self.first_step(user_text, earlier, system)

    # ---------- first reaction to a new user message ----------
    def first_step(self, text, earlier, system):
        lower = text.lower()
        earlier_users = [m.content for m in earlier if isinstance(m, HumanMessage)]
        topic = find_topic(text) or next((find_topic(t) for t in reversed(earlier_users) if find_topic(t)), None)

        if "language" in lower and ("which" in lower or "told" in lower or "preferred" in lower):
            return self.say(self.recall_answer(text, earlier_users, system))
        if "tried that" in lower:
            # "that" can only be understood from the earlier messages of this thread
            given = [m.content for m in earlier if isinstance(m, AIMessage) and "instructions from" in m.content]
            if given:
                guide = re.search(r"instructions from (\S+?):", given[-1]).group(1)
                return self.say(f"I understand: 'that' means the {guide} instructions I gave earlier in this "
                                "conversation, and they did not solve the problem. I can check the service "
                                "status or prepare a local ticket draft - which would you like?")
            return self.say("I do not see earlier instructions in this conversation. What did you try?")
        if any(phrase in lower for phrase in ["did i already try", "did we discuss", "which steps did i"]):
            return self.say(self.recall_answer(text, earlier_users, system))

        asked_ticket_before = any("ticket" in t.lower() for t in earlier_users)
        steps = extract_steps(text)

        if any(word in lower for word in ["status", "is it down", "is down", "known problem", "whether wi-fi is down"]):
            service = "account_recovery" if topic == "password" else "wifi"
            return AIMessage(content="", tool_calls=[tool_call("check_service_status", {"service": service})])

        if "ticket" in lower or asked_ticket_before:
            if topic and steps:
                summary = extract_problem(text, topic) or f"{topic} problem reported by the user"
                return AIMessage(content="", tool_calls=[tool_call(
                    "prepare_ticket", {"category": topic, "summary": summary, "steps_tried": steps})])
            return self.say("What problem are you having, and what have you already tried?")

        if topic:
            return AIMessage(content="", tool_calls=[tool_call("find_instructions", {"topic": topic})])
        return self.say("I can help with Wi-Fi problems, password recovery, service status and local ticket drafts. "
                        "What is your problem?")

    # ---------- reaction after our program returned tool results ----------
    def answer_after_tools(self, text, tool_results, system):
        lower = text.lower()
        last = json.loads(tool_results[-1].content)
        name = tool_results[-1].name

        if name == "find_instructions":
            steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(last["steps"], 1))
            return self.say(f"Here are the instructions from {last['instruction_id']}:\n{steps}")

        if name == "check_service_status":
            status, service = last["status"], last["service"]
            wants_ticket_if_ok = "ticket" in lower and ("if it is operating" in lower or "if it is working" in lower)
            if status == "operational" and wants_ticket_if_ok:
                topic = "password" if service == "account_recovery" else "wifi"
                steps = extract_steps(text)
                summary = extract_problem(text, topic) or f"{topic} problem reported by the user"
                if not steps:
                    return self.say(f"The demo status source reports {service} is operating. "
                                    "What have you already tried? I need that for the ticket.")
                return AIMessage(content="", tool_calls=[tool_call(
                    "prepare_ticket", {"category": topic, "summary": summary, "steps_tried": steps})])
            if status == "operational":
                return self.say(f"The demo status source reports {service} is operating: {last['message']}")
            if status == "outage":
                extra = (" Your request said to prepare a ticket only if the service was operating, "
                         "so I have not prepared that ticket." if "ticket" in lower else "")
                return self.say(f"The demo status source reports an outage for {service}: {last['message']}"
                                f"{extra} The data does not give a repair time.")
            return self.say(f"I could not determine the {service} status ({last['message']}). "
                            "I cannot confirm that it is operating, so I have not prepared a ticket. "
                            "Would you like a ticket draft explaining that the status could not be checked?")

        if name == "prepare_ticket":
            if last.get("created"):
                return self.say(f"I prepared a local ticket draft: {last['file']} "
                                f"(service status: {last['ticket']['service_status']}). Nothing has been sent.")
            return self.say("I could not prepare the ticket: " + last["message"])

        return self.say("Something went wrong with the tool: " + json.dumps(last))

    # ---------- answers that use memory ----------
    def recall_answer(self, text, earlier_users, system):
        parts = []
        lower = text.lower()
        language = re.search(r"Saved preferred reply language for this user: (\w+)", system)
        if "language" in lower:
            parts.append(f"You asked me to reply in {language.group(1)} (from user memory)." if language
                         else "No preferred language has been recorded for you.")
        problem_messages = [t for t in earlier_users if find_topic(t)]
        if problem_messages:
            topic = find_topic(problem_messages[-1])
            steps = [s for t in earlier_users for s in extract_steps(t)]
            sentence = f"In this conversation you reported a {'Wi-Fi' if topic == 'wifi' else 'password'} problem"
            sentence += f" and said you already tried: {'; '.join(steps)}." if steps else "."
            parts.append(sentence)
        elif "problem" in lower or "steps" in lower:
            parts.append("No previous problem has been recorded for you in this conversation.")
        if "help with" in lower:
            parts.append("I can help with Wi-Fi troubleshooting, password recovery guidance, "
                         "service status lookup and preparing local ticket drafts.")
        return " ".join(parts)

    def say(self, text):
        note = ""
        if getattr(self, "language", None) and self.language != "English":
            note = f"(user memory says reply in {self.language}; the fake model can only write English) "
        return AIMessage(content=LABEL + note + text)
