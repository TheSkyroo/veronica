import pytest

from veronica.tools import music


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def text(res):
    return res["content"][0]["text"]


def _scripted(monkeypatch, responses):
    """responses: list of Done, one per subprocess.run call, in order."""
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        idx = len(calls) - 1
        return responses[idx] if idx < len(responses) else Done(out="")

    monkeypatch.setattr(music.subprocess, "run", run)
    return calls


def spotify_running(running: bool):
    return Done(out="true" if running else "false")


async def test_music_play_no_query_spotify(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(True), Done(out="")])
    res = await music.music_play.handler({})
    assert not res.get("is_error")
    assert text(res) == "Playing."
    assert 'tell application "Spotify" to play' in calls[1][0][2]


async def test_music_play_no_query_music_app(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="")])
    res = await music.music_play.handler({})
    assert 'tell application "Music" to play' in calls[1][0][2]


async def test_music_play_with_query_spotify_uses_search_url(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(True), Done(out="")])
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert not res.get("is_error")
    assert text(res) == "Playing bohemian rhapsody."
    script = calls[1][0][2]
    # T3: the query is percent-encoded into the URL
    assert "spotify:search:bohemian%20rhapsody" in script
    assert "spotify:search:bohemian rhapsody" not in script
    assert "delay 1" in script
    assert 'tell application "Spotify" to play' in script


async def test_music_play_spotify_query_percent_encodes_specials(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(True), Done(out="")])
    await music.music_play.handler({"query": 'AC/DC & "friends" #1 café'})
    script = calls[1][0][2]
    assert "spotify:search:AC%2FDC%20%26%20%22friends%22%20%231%20caf%C3%A9" in script
    assert '"friends"' not in script.split("open location")[1].split("\n")[0].strip('"')


def test_music_q_escapes_backslashes_and_quotes():
    assert music._q('a\\b "c"') == 'a\\\\b \\"c\\"'


async def test_music_play_with_query_music_app(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="")])
    await music.music_play.handler({"query": "adele"})
    script = calls[1][0][2]
    assert 'tell application "Music"' in script
    assert 'name contains "adele"' in script
    assert 'artist contains "adele"' in script


async def test_music_play_query_escapes_quotes(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="")])
    await music.music_play.handler({"query": 'she said "hi"'})
    script = calls[1][0][2]
    assert '\\"hi\\"' in script


async def test_music_pause(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(True), Done(out="")])
    res = await music.music_pause.handler({})
    assert text(res) == "Paused."
    assert 'tell application "Spotify" to pause' in calls[1][0][2]


async def test_music_next(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="")])
    res = await music.music_next.handler({})
    assert text(res) == "Skipped."
    assert 'tell application "Music" to next track' in calls[1][0][2]


async def test_music_prev(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(True), Done(out="")])
    res = await music.music_prev.handler({})
    assert text(res) == "Playing previous track."
    assert 'tell application "Spotify" to previous track' in calls[1][0][2]


async def test_music_now_playing_nothing_running(monkeypatch):
    # detect_backend(): spotify not running; then _app_running("Music"): not running
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="false")])
    res = await music.music_now_playing.handler({})
    assert not res.get("is_error")
    assert text(res) == "Nothing is playing."


async def test_music_now_playing_spotify_playing(monkeypatch):
    calls = _scripted(monkeypatch, [
        spotify_running(True),   # detect_backend
        Done(out="true"),        # _app_running("Spotify")
        Done(out="Song Name\tThe Artist\tplaying"),
    ])
    res = await music.music_now_playing.handler({})
    assert text(res) == "Now playing Song Name by The Artist."


async def test_music_now_playing_paused(monkeypatch):
    calls = _scripted(monkeypatch, [
        spotify_running(False),
        Done(out="true"),
        Done(out="Song Name\tThe Artist\tpaused"),
    ])
    res = await music.music_now_playing.handler({})
    assert text(res) == "Paused on Song Name by The Artist."


async def test_music_now_playing_stopped_state(monkeypatch):
    calls = _scripted(monkeypatch, [
        spotify_running(True),
        Done(out="true"),
        Done(out=""),
    ])
    res = await music.music_now_playing.handler({})
    assert text(res) == "Nothing is playing."


async def test_music_volume_clamps(monkeypatch):
    calls = _scripted(monkeypatch, [spotify_running(False), Done(out="")])
    res = await music.music_volume.handler({"level": 250})
    assert text(res) == "Volume set to 100."
    assert "set sound volume to 100" in calls[1][0][2]


async def test_music_volume_bad_level_is_error(monkeypatch):
    res = await music.music_volume.handler({"level": "abc"})
    assert res["is_error"]


def test_server_and_names():
    assert music.music_server["name"] == "music"
    assert set(music.MUSIC_TOOL_NAMES) == {
        "music_play", "music_pause", "music_next", "music_prev",
        "music_now_playing", "music_volume",
    }


@pytest.mark.live
async def test_live_now_playing_no_error():
    res = await music.music_now_playing.handler({})
    assert not res.get("is_error")
