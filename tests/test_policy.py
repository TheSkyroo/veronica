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
    ("Bash", {"command": "curl https://x"}, ALLOW),
    ("Bash", {"command": 'curl -s --max-time 5 "https://wttr.in/?format=3"'}, ALLOW),
    ("Bash", {"command": "curl -X POST https://x"}, CONFIRM),
    ("Bash", {"command": "curl -d a=b https://x"}, CONFIRM),
    ("Bash", {"command": "curl -o f https://x"}, CONFIRM),
    ("Bash", {"command": "curl https://x https://y"}, CONFIRM),
    ("Bash", {"command": "curl -s file:///etc/passwd"}, CONFIRM),
    ("Bash", {"command": 'curl -H "Accept: text/plain" https://x'}, ALLOW),
    ("Bash", {"command": "curl -F a=b https://x"}, CONFIRM),
    ("Bash", {"command": "curl -O https://x"}, CONFIRM),
    ("Bash", {"command": "curl -T f https://x"}, CONFIRM),
    ("Bash", {"command": "curl --upload-file f https://x"}, CONFIRM),
    ("Bash", {"command": "curl -u user:pass https://x"}, CONFIRM),
    ("Bash", {"command": "curl --output f https://x"}, CONFIRM),
    ("Bash", {"command": "curl -K f https://x"}, CONFIRM),
    ("Bash", {"command": "curl -c f https://x"}, CONFIRM),
    ("Bash", {"command": "curl -b f https://x"}, CONFIRM),
    ("Bash", {"command": "curl --max-time=5 https://x"}, CONFIRM),  # curl rejects --opt=value
    ("Bash", {"command": "curl -A 'my-agent' https://x"}, ALLOW),
    ("Bash", {"command": "curl --compressed https://x"}, ALLOW),
    # -H/-A reading a local file into a header (exfil) — must always confirm
    ("Bash", {"command": "curl -H @headers.txt https://x"}, CONFIRM),
    ("Bash", {"command": "curl --header @headers.txt https://x"}, CONFIRM),
    ("Bash", {"command": "curl -A @agent.txt https://x"}, CONFIRM),
    ("Bash", {"command": "curl --user-agent @agent.txt https://x"}, CONFIRM),
    # header name allowlist (case-insensitive)
    ("Bash", {"command": 'curl -H "Accept-Language: en" https://x'}, ALLOW),
    ("Bash", {"command": 'curl -H "ACCEPT-ENCODING: gzip" https://x'}, ALLOW),
    ("Bash", {"command": 'curl -H "Cache-Control: no-cache" https://x'}, ALLOW),
    ("Bash", {"command": 'curl -H "Authorization: Bearer x" https://x'}, CONFIRM),
    ("Bash", {"command": 'curl -H "Cookie: a=b" https://x'}, CONFIRM),
    ("Bash", {"command": 'curl -H "X-HTTP-Method-Override: DELETE" https://x'}, CONFIRM),
    # credentials / private / loopback / link-local / metadata hosts
    ("Bash", {"command": "curl https://user:pass@example.com"}, CONFIRM),
    ("Bash", {"command": "curl https://localhost/"}, CONFIRM),
    ("Bash", {"command": "curl https://127.0.0.1/"}, CONFIRM),
    ("Bash", {"command": "curl https://0.0.0.0/"}, CONFIRM),
    ("Bash", {"command": "curl https://10.1.2.3/"}, CONFIRM),
    ("Bash", {"command": "curl https://172.16.0.1/"}, CONFIRM),
    ("Bash", {"command": "curl https://172.31.255.255/"}, CONFIRM),
    ("Bash", {"command": "curl https://172.32.0.1/"}, ALLOW),  # just outside 172.16-31
    ("Bash", {"command": "curl https://192.168.1.1/"}, CONFIRM),
    ("Bash", {"command": "curl https://169.254.169.254/"}, CONFIRM),  # cloud metadata
    ("Bash", {"command": "curl https://[::1]/"}, CONFIRM),
    ("Bash", {"command": "curl \"https://[::1\""}, CONFIRM),  # malformed URL -> confirm
    # combined short flags: only s/S/L/f chars allowed combined
    ("Bash", {"command": "curl -sS https://x"}, ALLOW),
    ("Bash", {"command": "curl -sL https://x"}, ALLOW),
    ("Bash", {"command": "curl -fsSL https://x"}, ALLOW),
    ("Bash", {"command": "curl -f https://x"}, ALLOW),
    ("Bash", {"command": "curl --fail https://x"}, ALLOW),
    ("Bash", {"command": "curl -sX https://x"}, CONFIRM),  # X not in the safe combo set
    # attached (no separating space) flag+value forms must confirm
    ("Bash", {"command": "curl -m5 https://x"}, CONFIRM),
    ("Bash", {"command": 'curl -H"Accept: text/plain" https://x'}, CONFIRM),
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
    # pim tools
    ("mcp__pim__calendar_events", {"day": "today"}, ALLOW),
    ("mcp__pim__calendar_create", {"title": "x", "start": "2026-09-20 10:00"}, CONFIRM),
    ("mcp__pim__mail_unread", {}, ALLOW),
    ("mcp__pim__mail_search", {"query": "x"}, ALLOW),
    ("mcp__pim__mail_send", {"to": "a@b.com", "subject": "s", "body": "b"}, CONFIRM),
    ("mcp__pim__reminder_create", {"title": "x"}, CONFIRM),
    ("mcp__pim__reminders_due", {}, ALLOW),
    ("mcp__pim__timer_set", {"minutes": 1}, ALLOW),
    ("mcp__pim__timer_list", {}, ALLOW),
    ("mcp__pim__timer_cancel", {"label": "x"}, ALLOW),
    ("mcp__pim__notes_create", {"title": "t", "body": "b"}, ALLOW),
    ("mcp__pim__unknown", {}, CONFIRM),
    # screen / music tools (batch A)
    ("mcp__screen__screenshot", {"region": "screen"}, ALLOW),
    ("mcp__screen__screenshot", {}, ALLOW),
    ("mcp__screen__bogus", {}, CONFIRM),
    ("mcp__music__music_play", {"query": "adele"}, ALLOW),
    ("mcp__music__music_pause", {}, ALLOW),
    ("mcp__music__music_next", {}, ALLOW),
    ("mcp__music__music_prev", {}, ALLOW),
    ("mcp__music__music_now_playing", {}, ALLOW),
    ("mcp__music__music_volume", {"level": 30}, ALLOW),
    ("mcp__music__bogus", {}, CONFIRM),
    # memory tools
    ("mcp__memory__recall", {"query": "weather"}, ALLOW),
    ("mcp__memory__facts_list", {}, ALLOW),
    ("mcp__memory__fact_add", {"text": "likes tea"}, CONFIRM),
    ("mcp__memory__fact_delete", {"text": "likes tea"}, CONFIRM),
    ("mcp__memory__unknown", {}, CONFIRM),
    ("mcp__unknownserver__anything", {}, CONFIRM),
]


@pytest.mark.parametrize("tool,inp,expected", CASES)
def test_classify(tool, inp, expected):
    assert classify(tool, inp) == expected


@pytest.mark.parametrize("short,expected", [
    ("browser_tabs", "allow"), ("browser_open", "allow"), ("browser_read", "allow"),
    ("browser_find", "allow"), ("browser_scroll", "allow"), ("browser_back", "allow"),
    ("browser_click", "confirm"), ("browser_type", "confirm"), ("browser_unknown", "confirm"),
])
def test_browser_tool_risk(short, expected):
    assert classify(f"mcp__browser__{short}", {}) == expected
