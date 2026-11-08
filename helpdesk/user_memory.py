"""Part 3: user memory (long-term, per user_id).

Saved in a small JSON file, for example:
    {"demo-user-a": {"language": "Kazakh"}}

This is separate from conversation memory, which LangGraph's SQLite
checkpointer saves per thread.
"""

import json
import re
from pathlib import Path

KNOWN_LANGUAGES = ["Kazakh", "Russian", "English", "Turkish", "Chinese", "German", "French", "Spanish"]

# e.g. "I prefer replies in Kazakh", "please reply in Russian", "answer in English"
PREFERENCE_PATTERNS = [
    r"prefer\s+(?:replies|answers|responses)?\s*in\s+([a-z]+)",
    r"(?:reply|answer|respond|write)\s+(?:to me\s+)?in\s+([a-z]+)",
]


def detect_language_preference(message):
    """Return a language name if the message asks for a reply language, else None."""
    text = message.lower()
    for pattern in PREFERENCE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            word = match.group(1).capitalize()
            if word in KNOWN_LANGUAGES:
                return word
    return None


class UserMemory:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def _load_all(self):
        if not self.file_path.exists():
            return {}
        with open(self.file_path, encoding="utf-8") as f:
            return json.load(f)

    def get(self, user_id):
        """Return the saved preferences of ONE user (empty dict if none)."""
        return self._load_all().get(user_id, {})

    def save(self, user_id, key, value):
        data = self._load_all()
        data.setdefault(user_id, {})[key] = value
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
