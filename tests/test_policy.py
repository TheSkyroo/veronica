import pytest

from veronica.brain.policy import AUTO_ALLOWABLE, classify
from veronica.config import Settings

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
    # no allowlist passed: every shortcut is asked about
    ("mcp__mac__run_shortcut", {"name": "Morning"}, CONFIRM),
    # pim tools
    ("mcp__pim__calendar_events", {"day": "today"}, ALLOW),
    ("mcp__pim__calendar_create", {"title": "x", "start": "2026-09-20 10:00"}, CONFIRM),
    ("mcp__pim__mail_unread", {}, ALLOW),
    ("mcp__pim__mail_search", {"query": "x"}, ALLOW),
    ("mcp__pim__mail_send", {"to": "a@b.com", "subject": "s", "body": "b"}, CONFIRM),
    ("mcp__pim__message_send", {"to": "+1555", "body": "b"}, CONFIRM),
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


@pytest.mark.parametrize("short,expected", [
    ("computer_move", "allow"), ("computer_scroll", "allow"), ("computer_find", "allow"),
    ("computer_click", "confirm"), ("computer_click_text", "confirm"), ("computer_drag", "confirm"),
    ("computer_type", "confirm"), ("computer_key", "confirm"), ("computer_unknown", "confirm"),
])
def test_computer_tool_risk(short, expected):
    assert classify(f"mcp__computer__{short}", {}) == expected


# --- always_confirm: never pre-approvable, never trusted ---------------------

from veronica.brain.policy import TRUST_EXCLUDED_BUNDLES, always_confirm
from veronica.tools.computer_events import Front

_FINDER = Front(app="Finder", bundle_id="com.apple.finder", window_title="Desktop", pid=1)
_TERMINAL = Front(app="Terminal", bundle_id="com.apple.Terminal", window_title="zsh", pid=2)
_SECAGENT = Front(app="SecurityAgent", bundle_id="com.apple.SecurityAgent", window_title="", pid=3)

ALWAYS_CASES = [
    ("mcp__pim__mail_send", {"to": "a@b.c"}, True),
    ("mcp__pim__message_send", {"to": "+1555", "body": "on my way"}, True),
    ("mcp__pim__calendar_create", {"title": "x"}, False),
    ("mcp__mac__clipboard_write", {"text": "x"}, False),
    ("mcp__memory__fact_add", {"text": "x"}, False),
    ("Write", {"file_path": "/a"}, False),
    # bash: destructive / power / privilege
    ("Bash", {"command": "rm -rf build"}, True),
    ("Bash", {"command": "rm -r build"}, True),
    ("Bash", {"command": "rm -fr build"}, True),
    ("Bash", {"command": "rm -Rf build"}, True),
    ("Bash", {"command": "rm notes.txt"}, False),
    ("Bash", {"command": "cd x && rm -rf y"}, True),
    # wrappers that hide the real command
    ("Bash", {"command": "sh -c 'rm -rf ~/Downloads/*'"}, True),
    ("Bash", {"command": "bash -c 'echo hi'"}, True),
    ("Bash", {"command": "python3 -c 'import shutil'"}, True),
    ("Bash", {"command": "ls | xargs rm"}, True),
    ("Bash", {"command": "find ~/Downloads -name '*.tmp' -delete"}, True),
    ("Bash", {"command": "find . -exec rm {} \\;"}, True),
    ("Bash", {"command": "find . -name '*.py'"}, False),
    ("Bash", {"command": "eval \"$cmd\""}, True),
    ("Bash", {"command": "python3 script.py"}, False),
    ("Bash", {"command": "git push --force origin main"}, True),
    ("Bash", {"command": "git push -f"}, True),
    ("Bash", {"command": "git push --force-with-lease"}, True),
    ("Bash", {"command": "git push origin main"}, False),
    ("Bash", {"command": "git commit -m 'rm -rf'"}, False),
    ("Bash", {"command": "shutdown -h now"}, True),
    ("Bash", {"command": "sudo shutdown -r now"}, True),
    ("Bash", {"command": "reboot"}, True),
    ("Bash", {"command": "halt"}, True),
    ("Bash", {"command": "/sbin/reboot"}, True),
    ("Bash", {"command": "pmset sleepnow"}, True),
    ("Bash", {"command": "pmset restart"}, True),
    ("Bash", {"command": "pmset -g"}, False),
    ("Bash", {"command": "osascript -e 'tell app \"System Events\" to shut down'"}, True),
    ("Bash", {"command": "osascript -e 'tell app \"System Events\" to restart'"}, True),
    ("Bash", {"command": "osascript -e 'tell app \"System Events\" to log out'"}, True),
    ("Bash", {"command": "osascript -e 'tell app \"System Events\" to sleep'"}, True),
    ("Bash", {"command": "osascript -e 'display notification \"hi\"'"}, False),
    ("Bash", {"command": "sudo ls"}, True),
    ("Bash", {"command": "killall Finder"}, True),
    ("Bash", {"command": "diskutil list"}, True),
    ("Bash", {"command": "launchctl unload ~/Library/LaunchAgents/x.plist"}, True),
    ("Bash", {"command": "launchctl bootout gui/501/x"}, True),
    ("Bash", {"command": "launchctl list"}, False),
    ("Bash", {"command": "defaults write com.apple.finder AppleShowAllFiles YES"}, True),
    ("Bash", {"command": "defaults write com.example.app key 1"}, False),
    ("Bash", {"command": "defaults read com.apple.finder"}, False),
    ("Bash", {"command": "tccutil reset All"}, True),
    ("Bash", {"command": "echo 'rm -rf'"}, False),
    ("Bash", {"command": "rm -rf 'unterminated"}, True),   # unparsable: assume the worst
    # computer: Enter, terminals, system dialogs
    ("mcp__computer__computer_key", {"combo": "enter"}, True),
    ("mcp__computer__computer_key", {"combo": "Return"}, True),
    ("mcp__computer__computer_key", {"combo": "cmd+enter"}, False),
    ("mcp__computer__computer_key", {"combo": "cmd+s"}, False),
    ("mcp__computer__computer_key", {"combo": ""}, False),
    ("mcp__computer__computer_type", {"text": "hi", "submit": True}, True),
    ("mcp__computer__computer_type", {"text": "hi"}, False),
    ("mcp__computer__computer_click", {"x": 1, "y": 2}, False),
    # applescript
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to shut down'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to restart'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to log out'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "Finder" to delete file "x"'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "Finder" to empty trash'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to keystroke return'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to key code 36'}, True),
    ("mcp__mac__applescript", {"script": 'tell application "System Events" to keystroke "a"'}, False),
    ("mcp__mac__applescript", {"script": 'tell application "Finder" to activate'}, False),
]


@pytest.mark.parametrize("tool,inp,expected", ALWAYS_CASES)
def test_always_confirm(tool, inp, expected):
    assert always_confirm(tool, inp) is expected
    assert always_confirm(tool, inp, _FINDER) is expected


def test_always_confirm_computer_type_in_terminal():
    assert always_confirm("mcp__computer__computer_type", {"text": "ls"}, _TERMINAL) is True
    assert always_confirm("mcp__computer__computer_type", {"text": "ls"}, _FINDER) is False
    for bundle in TRUST_EXCLUDED_BUNDLES:
        front = Front(app="t", bundle_id=bundle, window_title="", pid=9)
        assert always_confirm("mcp__computer__computer_type", {"text": "ls"}, front) is True
    # a click in a terminal is not an always-confirm (the trust window
    # already never covers terminals)
    assert always_confirm("mcp__computer__computer_click", {"x": 1, "y": 1}, _TERMINAL) is False


@pytest.mark.parametrize("short,inp", [
    ("computer_click", {"x": 1, "y": 1}), ("computer_click_text", {"text": "Allow"}),
    ("computer_drag", {}), ("computer_type", {"text": "x"}), ("computer_key", {"combo": "cmd+s"}),
    ("computer_move", {"x": 1, "y": 1}),
])
def test_always_confirm_every_computer_tool_on_system_dialog(short, inp):
    assert always_confirm(f"mcp__computer__{short}", inp, _SECAGENT) is True
    assert always_confirm(f"mcp__computer__{short}", inp, None) is False


# --- run_shortcut: allow only what the user put on the allowlist -------------

def test_run_shortcut_allowlisted_name_is_allowed():
    assert classify("mcp__mac__run_shortcut", {"name": "Morning"}, ["Morning"]) == ALLOW


def test_run_shortcut_allowlist_ignores_case_and_padding():
    assert classify("mcp__mac__run_shortcut", {"name": " morning "}, [" Morning"]) == ALLOW


def test_run_shortcut_off_the_allowlist_confirms():
    assert classify("mcp__mac__run_shortcut", {"name": "Wipe Disk"}, ["Morning"]) == CONFIRM
    assert classify("mcp__mac__run_shortcut", {"name": ""}, ["Morning", ""]) == CONFIRM
    assert classify("mcp__mac__run_shortcut", {}, ["Morning"]) == CONFIRM


def test_run_shortcut_empty_allowlist_allows_nothing():
    assert classify("mcp__mac__run_shortcut", {"name": "Morning"}, []) == CONFIRM


def test_the_allowlist_does_not_leak_to_other_tools():
    # the extra argument must not turn anything else into an auto-allow
    assert classify("mcp__mac__applescript", {"script": "Morning"}, ["Morning"]) == CONFIRM
    assert classify("mcp__pim__message_send", {"to": "Morning"}, ["Morning"]) == CONFIRM


def test_run_shortcut_is_not_always_confirm():
    # allowlisted shortcuts are meant to be auto-allowed; only the classify
    # gate decides, so this must stay False
    assert always_confirm("mcp__mac__run_shortcut", {"name": "Morning"}) is False


def test_always_confirm_future_send_tools():
    # any messages/mail "send" tool on any server is an always-ask
    assert always_confirm("mcp__pim__messages_send", {"to": "x"}) is True
    assert always_confirm("mcp__messages__send", {"to": "x"}) is True
    assert always_confirm("mcp__pim__mail_search", {"query": "x"}) is False


# --- auto-allow: confirm-class tools the user approved for good --------------

def test_auto_allowable_is_exactly_the_eligible_set():
    assert AUTO_ALLOWABLE == {
        "mcp__mac__clipboard_write",
        "mcp__pim__calendar_create",
        "mcp__pim__reminder_create",
        "mcp__memory__fact_add",
        "mcp__memory__fact_delete",
        "mcp__browser__browser_click",
        "mcp__browser__browser_type",
    }


@pytest.mark.parametrize("tool", sorted(AUTO_ALLOWABLE))
def test_eligible_tool_confirms_until_it_is_on_the_list(tool):
    assert classify(tool, {}) == CONFIRM
    assert classify(tool, {}, (), [tool]) == ALLOW


def test_auto_allow_only_covers_the_tool_that_is_listed():
    assert classify("mcp__pim__reminder_create", {}, (), ["mcp__mac__clipboard_write"]) == CONFIRM


@pytest.mark.parametrize("tool, inp", [
    ("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"}),
    ("mcp__pim__message_send", {"to": "a", "body": "y"}),
    ("mcp__mac__applescript", {"script": "delete everything"}),
    ("mcp__computer__computer_click", {"x": 1, "y": 2}),
    ("mcp__computer__computer_type", {"text": "rm -rf /"}),
    ("mcp__computer__computer_key", {"combo": "return"}),
    ("mcp__mac__run_shortcut", {"name": "Wipe Disk"}),
    ("Bash", {"command": "rm -rf /tmp/x"}),
])
def test_ineligible_tool_still_confirms_even_if_hand_typed_into_the_setting(tool, inp):
    assert classify(tool, inp, (), [tool]) == CONFIRM


def test_always_confirm_wins_over_the_auto_allow_list():
    # mail_send isn't eligible anyway; this pins the belt-and-braces check.
    assert always_confirm("mcp__pim__mail_send", {}) is True
    assert classify("mcp__pim__mail_send", {}, (), ["mcp__pim__mail_send"]) == CONFIRM


def test_auto_allow_entries_are_trimmed_and_blanks_ignored():
    assert classify("mcp__mac__clipboard_write", {}, (), [" mcp__mac__clipboard_write "]) == ALLOW
    assert classify("mcp__mac__clipboard_write", {}, (), ["", None]) == CONFIRM


def test_clipboard_write_is_auto_allowed_by_default():
    assert Settings().auto_allow_tools == ["mcp__mac__clipboard_write"]
    assert classify("mcp__mac__clipboard_write", {"text": "hi"}, (), Settings().auto_allow_tools) == ALLOW
