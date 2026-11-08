"""User-memory unit tests. They catch a broken preference detector (for example
saving a preference from a question) and a store that mixes users."""

from helpdesk.user_memory import UserMemory, detect_language_preference


def test_detects_language_preference():
    assert detect_language_preference("Please remember that I prefer replies in Kazakh.") == "Kazakh"
    assert detect_language_preference("Reply in Russian please") == "Russian"


def test_question_about_language_is_not_a_new_preference():
    assert detect_language_preference("Which language did I ask you to use?") is None


def test_store_keeps_users_separate(tmp_path):
    memory = UserMemory(tmp_path / "user_memory.json")
    memory.save("demo-user-a", "language", "Kazakh")
    assert memory.get("demo-user-a") == {"language": "Kazakh"}
    assert memory.get("demo-user-b") == {}
    assert UserMemory(tmp_path / "user_memory.json").get("demo-user-a")["language"] == "Kazakh"  # re-read from disk
