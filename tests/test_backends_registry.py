import subprocess

from veronica.brain.backends import BACKENDS, Availability, check_backend, make_brain
from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.backends.claude import ClaudeBrain
from veronica.brain.backends.codex import CodexBrain
from veronica.brain.backends.copilot import CopilotBrain
from veronica.brain.gate import ToolGate
from veronica.config import BRAIN_BACKENDS, Settings


def test_registry_order_and_commands():
    assert tuple(BACKENDS) == ("codex", "antigravity", "claude", "copilot") == BRAIN_BACKENDS
    assert [b.label for b in BACKENDS.values()] == ["Codex", "Antigravity", "Claude", "Copilot"]
    assert [b.binary for b in BACKENDS.values()] == ["codex", "agy", "claude", "copilot"]
    assert BACKENDS["codex"].install_cmd == "npm i -g @openai/codex" and BACKENDS["codex"].login_cmd == "codex login"
    assert BACKENDS["copilot"].install_cmd == "npm i -g @github/copilot"
    assert BACKENDS["claude"].install_cmd == "npm i -g @anthropic-ai/claude-code"
    assert "antigravity.google/cli/install.sh" in BACKENDS["antigravity"].install_cmd
    assert BACKENDS["codex"].login_markers == (".codex/auth.json",)
    assert BACKENDS["copilot"].login_markers == (".copilot/config.json",)
    assert BACKENDS["claude"].login_markers == ()
    assert BACKENDS["codex"].cls is CodexBrain and BACKENDS["claude"].cls is ClaudeBrain
    assert BACKENDS["antigravity"].cls is AntigravityBrain and BACKENDS["copilot"].cls is CopilotBrain
    for name, info in BACKENDS.items():
        assert info.name == name == info.cls.name


def test_check_backend_matrix(tmp_path):
    a = check_backend("codex", which=lambda b: None)
    assert a == Availability(False, "not installed",
                             "Codex isn't installed — run npm i -g @openai/codex, then codex login.")
    a = check_backend("codex", which=lambda b: "/x/codex", home=tmp_path)
    assert a == Availability(False, "not logged in", "Codex isn't logged in — run codex login in a terminal.")
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "auth.json").write_text("{}")
    a = check_backend("codex", which=lambda b: "/x/codex", home=tmp_path)
    assert a.ok and a.reason == "ok" and a.hint == ""
    hint = check_backend("copilot", which=lambda b: None).hint
    assert "npm i -g @github/copilot" in hint and "copilot login" in hint
    # claude: the SDK finds the CLI; only the binary is checked, login is assumed
    assert check_backend("claude", which=lambda b: "/x/claude", home=tmp_path).ok
    assert check_backend("claude", which=lambda b: None).reason == "not installed"


def test_check_backend_asks_which_for_the_binary(tmp_path):
    asked = []
    check_backend("antigravity", which=lambda b: asked.append(b), home=tmp_path)
    assert asked == ["agy"]


def test_check_backend_copilot_marker(tmp_path):
    assert check_backend("copilot", which=lambda b: "/x/copilot", home=tmp_path).reason == "not logged in"
    (tmp_path / ".copilot").mkdir()
    (tmp_path / ".copilot" / "config.json").write_text("{}")
    assert check_backend("copilot", which=lambda b: "/x/copilot", home=tmp_path).ok


def test_check_backend_antigravity_dir_marker_first(tmp_path):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 44)

    a = check_backend("antigravity", which=lambda b: "/x/agy", home=tmp_path, run=run)
    assert a.reason == "not logged in" and a.hint == "Antigravity isn't logged in — run agy in a terminal."
    assert calls == [["security", "find-generic-password", "-a", "antigravity"]]
    (tmp_path / ".gemini" / "antigravity-cli" / "conversations").mkdir(parents=True)
    calls.clear()
    assert check_backend("antigravity", which=lambda b: "/x/agy", home=tmp_path, run=run).ok
    assert calls == []   # the directory marker settles it; the Keychain is not consulted


def test_check_backend_antigravity_keychain_fallback(tmp_path):
    run = lambda argv, **kw: subprocess.CompletedProcess(argv, 0)
    assert check_backend("antigravity", which=lambda b: "/x/agy", home=tmp_path, run=run).ok

    def boom(argv, **kw):
        raise OSError("no security binary")

    assert check_backend("antigravity", which=lambda b: "/x/agy", home=tmp_path, run=boom).reason == "not logged in"


def test_check_backend_exists_override(tmp_path):
    seen = []

    def exists(p):
        seen.append(p)
        return True

    assert check_backend("codex", which=lambda b: "/x", exists=exists, home=tmp_path).ok
    assert seen == [tmp_path / ".codex" / "auth.json"]


def test_check_backend_unknown_name():
    a = check_backend("qwen", which=lambda b: "/x")
    assert not a.ok and a.reason == "not installed" and "qwen" in a.hint


def test_make_brain_builds_each(tmp_path):
    s = Settings(home=tmp_path)
    g = ToolGate(s, None)
    cards = []
    on_tool = lambda su, d: cards.append((su, d))
    for name in BACKENDS:
        b = make_brain(name, s, gate=g, on_tool=on_tool, memory="mem")
        assert b.name == name and b.gate is g and b._memory == "mem"
        assert isinstance(b, BACKENDS[name].cls)
        if name != "claude":
            assert b._on_tool is on_tool


def test_make_brain_unknown_raises(tmp_path):
    s = Settings(home=tmp_path)
    try:
        make_brain("qwen", s, gate=ToolGate(s, None))
    except KeyError as e:
        assert "qwen" in str(e)
    else:
        raise AssertionError("expected KeyError")
