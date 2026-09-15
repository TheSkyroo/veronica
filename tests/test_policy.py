import pytest

from veronica.brain.policy import classify

ALLOW = "allow"
CONFIRM = "confirm"

CASES = [
    # built-ins
    ("Read", {"file_path": "/a"}, ALLOW),
    ("Glob", {"pattern": "*.py"}, ALLOW),
    ("Grep", {"pattern": "x"}, ALLOW),
    ("WebSearch", {"query": "weather"}, ALLOW),
    ("WebFetch", {"url": "https://x"}, ALLOW),
    ("Write", {"file_path": "/a"}, CONFIRM),
    ("Edit", {"file_path": "/a"}, CONFIRM),
    ("NotebookEdit", {}, CONFIRM),
    ("SomethingNew", {}, CONFIRM),
    # bash safe
    ("Bash", {"command": "ls -la ~/Desktop"}, ALLOW),
    ("Bash", {"command": "pbpaste"}, ALLOW),
    ("Bash", {"command": "date"}, ALLOW),
    ("Bash", {"command": "open -a Safari"}, ALLOW),
    ("Bash", {"command": "open -a 'Google Chrome'"}, ALLOW),
    ("Bash", {"command": "open https://example.com"}, ALLOW),
    ("Bash", {"command": "cat /etc/hosts"}, ALLOW),
    ("Bash", {"command": "open -a /tmp/evil.app"}, CONFIRM),
    ("Bash", {"command": "open -a ../x"}, CONFIRM),
    ("Bash", {"command": "open -a -x"}, CONFIRM),
    ("Bash", {"command": "open -a 'Visual Studio Code'"}, ALLOW),
    # bash confirm
    ("Bash", {"command": "rm -rf ~/x"}, CONFIRM),
    ("Bash", {"command": "sudo ls"}, CONFIRM),
    ("Bash", {"command": "ls; rm -rf ~"}, CONFIRM),
    ("Bash", {"command": "ls && rm x"}, CONFIRM),
    ("Bash", {"command": "cat a | grep b"}, CONFIRM),
    ("Bash", {"command": "echo hi > f"}, CONFIRM),
    ("Bash", {"command": "ls $(rm x)"}, CONFIRM),
    ("Bash", {"command": "ls `rm x`"}, CONFIRM),
    ("Bash", {"command": "ls\nrm x"}, CONFIRM),
    ("Bash", {"command": "open file:///etc/passwd"}, CONFIRM),
    ("Bash", {"command": "open -a Safari --args x"}, CONFIRM),
    ("Bash", {"command": "open /Applications"}, CONFIRM),
    ("Bash", {"command": "curl https://x"}, CONFIRM),
    ("Bash", {"command": "git push"}, CONFIRM),
    ("Bash", {"command": ""}, CONFIRM),
    ("Bash", {"command": "ls 'unterminated"}, CONFIRM),
    ("Bash", {}, CONFIRM),
    # mac tools
    ("mcp__mac__open_app", {"name": "Safari"}, ALLOW),
    ("mcp__mac__open_url", {"url": "https://x"}, ALLOW),
    ("mcp__mac__clipboard_read", {}, ALLOW),
    ("mcp__mac__clipboard_write", {"text": "x"}, CONFIRM),
    ("mcp__mac__notify", {"title": "a", "message": "b"}, ALLOW),
    ("mcp__mac__volume_get", {}, ALLOW),
    ("mcp__mac__volume_set", {"level": 30}, ALLOW),
    ("mcp__mac__applescript", {"script": "beep"}, CONFIRM),
    ("mcp__mac__unknown", {}, CONFIRM),
]


@pytest.mark.parametrize("tool,inp,expected", CASES)
def test_classify(tool, inp, expected):
    assert classify(tool, inp) == expected
