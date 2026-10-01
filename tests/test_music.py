from types import SimpleNamespace

import pytest

from veronica.tools import music


def text(res):
    return res["content"][0]["text"]


class FakeSession:
    """A GSMTC session: async try_* controls, playback info, media props."""

    def __init__(self, status=music.STATUS_PLAYING, title="Song Name", artist="The Artist",
                 app="Spotify.exe", accept=True):
        self.calls = []
        self.status, self.title, self.artist = status, title, artist
        self.source_app_user_model_id = app
        self.accept = accept

    def __getattr__(self, name):
        if name.startswith("try_") and name != "try_get_media_properties_async":
            async def control():
                self.calls.append(name)
                return self.accept
            return control
        raise AttributeError(name)

    def get_playback_info(self):
        return SimpleNamespace(playback_status=self.status)

    async def try_get_media_properties_async(self):
        return SimpleNamespace(title=self.title, artist=self.artist)


@pytest.fixture
def player(monkeypatch):
    """The OS seams replaced: one current media session (None to simulate
    no player), a Spotify-installed switch, and a log of opened URIs and
    per-app volume calls."""
    state = SimpleNamespace(session=FakeSession(), spotify=True, opened=[], volumes=[],
                            mixer_hit="Spotify.exe")

    async def session():
        return state.session

    def set_volume(app_id, level):
        state.volumes.append((app_id, level))
        return state.mixer_hit

    monkeypatch.setattr(music, "_media_session", session)
    monkeypatch.setattr(music, "_spotify_installed", lambda: state.spotify)
    monkeypatch.setattr(music, "_open", state.opened.append)
    monkeypatch.setattr(music, "_set_app_volume", set_volume)
    return state


async def test_music_play_no_query_resumes(player):
    res = await music.music_play.handler({})
    assert text(res) == "Playing." and player.session.calls == ["try_play_async"]
    assert player.opened == []


async def test_music_play_no_query_without_a_player_is_error(player):
    player.session = None
    res = await music.music_play.handler({})
    assert res["is_error"] and "no music app" in text(res)


async def test_music_session_manager_failure_counts_as_no_player(player, monkeypatch):
    async def broken():
        raise ImportError("no winrt")
    monkeypatch.setattr(music, "_media_session", broken)
    assert (await music.music_pause.handler({}))["is_error"]
    assert text(await music.music_now_playing.handler({})) == "Nothing is playing."


async def test_music_play_with_query_opens_spotify_search(player):
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert not res.get("is_error")
    assert player.opened == ["spotify:search:bohemian%20rhapsody"]
    assert text(res) == "Searching Spotify for bohemian rhapsody."


async def test_music_play_spotify_query_percent_encodes_specials(player):
    await music.music_play.handler({"query": 'AC/DC & "friends" #1 café'})
    assert player.opened == ["spotify:search:AC%2FDC%20%26%20%22friends%22%20%231%20caf%C3%A9"]


async def test_music_play_without_spotify_uses_youtube_music(player):
    player.spotify = False
    res = await music.music_play.handler({"query": "adele & co"})
    assert player.opened == ["https://music.youtube.com/search?q=adele+%26+co"]
    assert text(res) == "Searching YouTube Music for adele & co."


async def test_music_play_query_whitespace_collapsed(player):
    await music.music_play.handler({"query": "  a\n\tb  "})
    assert player.opened == ["spotify:search:a%20b"]


async def test_music_pause(player):
    res = await music.music_pause.handler({})
    assert text(res) == "Paused." and player.session.calls == ["try_pause_async"]


async def test_music_next(player):
    res = await music.music_next.handler({})
    assert text(res) == "Skipped." and player.session.calls == ["try_skip_next_async"]


async def test_music_prev(player):
    res = await music.music_prev.handler({})
    assert text(res) == "Playing previous track."
    assert player.session.calls == ["try_skip_previous_async"]


async def test_music_control_refused_is_error(player):
    player.session.accept = False
    assert (await music.music_next.handler({}))["is_error"]


async def test_music_now_playing_nothing_running(player):
    player.session = None
    res = await music.music_now_playing.handler({})
    assert not res.get("is_error") and text(res) == "Nothing is playing."


async def test_music_now_playing_playing(player):
    assert text(await music.music_now_playing.handler({})) == "Now playing Song Name by The Artist."


async def test_music_now_playing_paused(player):
    player.session.status = music.STATUS_PAUSED
    assert text(await music.music_now_playing.handler({})) == "Paused on Song Name by The Artist."


async def test_music_now_playing_no_artist(player):
    player.session.artist = ""
    assert text(await music.music_now_playing.handler({})) == "Now playing Song Name."


@pytest.mark.parametrize("status", [0, 1, 2, 3])
async def test_music_now_playing_stopped_state(player, status):
    player.session.status = status
    assert text(await music.music_now_playing.handler({})) == "Nothing is playing."


async def test_music_volume_clamps_and_targets_the_player(player):
    res = await music.music_volume.handler({"level": 250})
    assert text(res) == "Volume set to 100."
    assert player.volumes == [("Spotify.exe", 100)]


async def test_music_volume_no_audio_session_is_error(player):
    player.mixer_hit = None
    res = await music.music_volume.handler({"level": 30})
    assert res["is_error"] and "Spotify.exe" in text(res)


async def test_music_volume_bad_level_is_error(player):
    res = await music.music_volume.handler({"level": "abc"})
    assert res["is_error"] and player.volumes == []


@pytest.mark.parametrize("app_id, proc, hit", [
    ("Spotify.exe", "Spotify.exe", True),
    ("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify", "Spotify.exe", True),
    ("Chrome", "chrome.exe", True),
    ("MSEdge", "msedge.exe", True),
    ("Spotify.exe", "chrome.exe", False),
    ("Spotify.exe", "", False),
])
def test_app_matches(app_id, proc, hit):
    assert music._app_matches(app_id, proc) is hit


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
