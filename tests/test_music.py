import json
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
    per-app volume calls. Spotify isn't connected."""
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
    monkeypatch.setattr(music, "_spotify_connected", lambda: False)
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


# -- play <query>: Spotify ------------------------------------------------------------
class FakeSpotify:
    """The Spotify Web API: canned search results, a device list that can
    change after N polls, and a play endpoint that records or refuses."""

    def __init__(self):
        self.calls = []
        self.tracks = [{"uri": "spotify:track:t1", "name": "Bohemian Rhapsody",
                        "artists": [{"name": "Queen"}]}]
        self.artists = [{"uri": "spotify:artist:a1", "name": "Queen"}]
        self.playlists = [{"uri": "spotify:playlist:p1", "name": "Chill Mix"}]
        self.devices = [[{"id": "pc", "name": "MYPC", "type": "Computer", "is_active": False}]]
        self.play_error = None

    def __call__(self, method, path, params=None, body=None):
        self.calls.append((method, path, params, body))
        if path == "/search":
            return {"tracks": {"items": self.tracks}, "artists": {"items": self.artists},
                    "playlists": {"items": self.playlists}}
        if path == "/me/player/devices":
            devs = self.devices[0] if len(self.devices) == 1 else self.devices.pop(0)
            return {"devices": devs}
        if path == "/me/player/play":
            if self.play_error:
                raise self.play_error
            return {}
        raise AssertionError(path)

    def played(self):
        return [(p, b) for m, path, p, b in self.calls if path == "/me/player/play"]


@pytest.fixture
def spotify(player, monkeypatch):
    api = FakeSpotify()
    monkeypatch.setattr(music, "_spotify_connected", lambda: True)
    monkeypatch.setattr(music, "_spotify_api", api)
    monkeypatch.setattr(music, "_sleep", lambda s: None)
    monkeypatch.setenv("COMPUTERNAME", "MyPC")
    return api


@pytest.fixture
def youtube(player, monkeypatch):
    """The YouTube results page: canned HTML (or an exception), and the
    URLs fetched."""
    state = SimpleNamespace(html=YT_PAGE, fetched=[])

    def get(url):
        state.fetched.append(url)
        if isinstance(state.html, Exception):
            raise state.html
        return state.html

    monkeypatch.setattr(music, "_http_get", get)
    return state


def _vr(vid, title, live=False):
    vr = {"videoId": vid, "title": {"runs": [{"text": title}]}}
    if live:
        vr["badges"] = [{"metadataBadgeRenderer": {"style": "BADGE_STYLE_TYPE_LIVE_NOW", "label": "LIVE"}}]
    else:
        vr["lengthText"] = {"simpleText": "5:55"}
    return {"videoRenderer": vr}


YT_DATA = {"contents": {"twoColumnSearchResultsRenderer": {"primaryContents": {
    "sectionListRenderer": {"contents": [{"itemSectionRenderer": {"contents": [
        {"adSlotRenderer": {"videoId": "AAAAAAAAAAA"}},
        {"reelShelfRenderer": {"items": [{"reelItemRenderer": {"videoId": "SSSSSSSSSSS"}}]}},
        _vr("LLLLLLLLLLL", "Live radio", live=True),
        _vr("fJ9rUzIMcZQ", "Queen – Bohemian Rhapsody (Official Video)"),
        _vr("zzzzzzzzzzz", "Second"),
    ]}}]}}}}}
YT_PAGE = ('<html><script>var foo = 1;</script><script nonce="x">var ytInitialData = '
           + json.dumps(YT_DATA) + ';</script><script>var x = {"a": "};"};</script></html>')


async def test_spotify_search_picks_device_and_plays_track(spotify, youtube, player):
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert text(res) == "Playing Bohemian Rhapsody by Queen on Spotify."
    assert spotify.played() == [({"device_id": "pc"}, {"uris": ["spotify:track:t1"]})]
    assert player.opened == [] and youtube.fetched == []


async def test_spotify_prefers_the_active_device(spotify, youtube, player):
    spotify.devices = [[{"id": "pc", "name": "MYPC", "type": "Computer"},
                        {"id": "phone", "name": "Pixel", "type": "Smartphone", "is_active": True}]]
    await music.music_play.handler({"query": "bohemian rhapsody"})
    assert spotify.played()[0][0] == {"device_id": "phone"}


async def test_spotify_prefers_this_pc_over_other_computers(spotify, youtube, player):
    spotify.devices = [[{"id": "lap", "name": "LAPTOP", "type": "Computer"},
                        {"id": "pc", "name": "MyPC", "type": "Computer"}]]
    await music.music_play.handler({"query": "bohemian rhapsody"})
    assert spotify.played()[0][0] == {"device_id": "pc"}


async def test_spotify_artist_query_plays_the_artist(spotify, youtube, player):
    res = await music.music_play.handler({"query": "queen"})
    assert text(res) == "Playing Queen on Spotify."
    assert spotify.played()[0][1] == {"context_uri": "spotify:artist:a1"}


async def test_spotify_playlist_query(spotify, youtube, player):
    res = await music.music_play.handler({"query": "playlist chill"})
    assert text(res) == "Playing the playlist Chill Mix on Spotify."
    assert spotify.calls[0][2]["type"] == "playlist" and spotify.calls[0][2]["q"] == "chill"
    assert spotify.played()[0][1] == {"context_uri": "spotify:playlist:p1"}


async def test_spotify_no_device_launches_app_and_polls(spotify, youtube, player):
    spotify.devices = [[], [], [{"id": "pc", "name": "MyPC", "type": "Computer"}]]
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert player.opened == ["spotify:"]
    assert text(res) == "Playing Bohemian Rhapsody by Queen on Spotify."
    assert spotify.played()[0][0] == {"device_id": "pc"}


async def test_spotify_device_never_appears_falls_back_to_youtube(spotify, youtube, player):
    spotify.devices = [[]]
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    polls = [c for c in spotify.calls if c[1] == "/me/player/devices"]
    assert 2 < len(polls) <= 12
    assert player.opened == ["spotify:", "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"]
    assert spotify.played() == []
    assert "on YouTube" in text(res)


async def test_spotify_not_installed_does_not_launch(spotify, youtube, player):
    spotify.devices = [[]]
    player.spotify = False
    await music.music_play.handler({"query": "bohemian rhapsody"})
    assert player.opened == ["https://www.youtube.com/watch?v=fJ9rUzIMcZQ"]


async def test_spotify_premium_required_falls_back_to_youtube(spotify, youtube, player):
    spotify.play_error = music.SpotifyError(403, "Premium required", "PREMIUM_REQUIRED")
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert not res.get("is_error")
    assert player.opened == ["https://www.youtube.com/watch?v=fJ9rUzIMcZQ"]
    assert text(res) == ("Spotify Premium is needed for playback control, so playing "
                         "Queen – Bohemian Rhapsody (Official Video) on YouTube.")


async def test_spotify_api_failure_falls_back_quietly(spotify, youtube, player):
    spotify.play_error = music.SpotifyError(0, "couldn't reach Spotify")
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert text(res) == "Playing Queen – Bohemian Rhapsody (Official Video) on YouTube."


# -- play <query>: YouTube --------------------------------------------------------------
async def test_not_connected_plays_top_youtube_video(youtube, player):
    res = await music.music_play.handler({"query": "bohemian rhapsody"})
    assert youtube.fetched == ["https://www.youtube.com/results?search_query=bohemian+rhapsody"]
    assert player.opened == ["https://www.youtube.com/watch?v=fJ9rUzIMcZQ"]
    assert text(res) == "Playing Queen – Bohemian Rhapsody (Official Video) on YouTube."


async def test_youtube_query_encoded_and_whitespace_collapsed(youtube, player):
    await music.music_play.handler({"query": "  AC/DC &\n\t\"friends\" #1 café "})
    assert youtube.fetched == [
        "https://www.youtube.com/results?search_query=AC%2FDC+%26+%22friends%22+%231+caf%C3%A9"]


def test_youtube_parse_skips_ads_shorts_and_live():
    assert music._youtube_top(YT_PAGE) == ("fJ9rUzIMcZQ", "Queen – Bohemian Rhapsody (Official Video)")


def test_youtube_parse_window_assignment_and_simple_text():
    data = {"x": [{"videoRenderer": {"videoId": "abcdefghij_", "lengthText": {},
                                     "title": {"simpleText": "Simple"}}}]}
    html = 'window["ytInitialData"] = ' + json.dumps(data) + ";"
    assert music._youtube_top(html) == ("abcdefghij_", "Simple")


def test_youtube_parse_only_live_still_plays_it():
    html = "var ytInitialData = " + json.dumps({"c": [_vr("LLLLLLLLLLL", "Live", live=True)]}) + ";"
    assert music._youtube_top(html) == ("LLLLLLLLLLL", "Live")


def test_youtube_parse_regex_fallback():
    html = 'var ytInitialData = {broken json; ... "videoId":"dQw4w9WgXcQ" ...'
    assert music._youtube_top(html) == ("dQw4w9WgXcQ", "")
    assert music._youtube_top('<a "videoId":"dQw4w9WgXcQ">') == ("dQw4w9WgXcQ", "")
    assert music._youtube_top("<html>consent</html>") is None


async def test_youtube_regex_fallback_says_the_query(youtube, player):
    youtube.html = '"videoId":"dQw4w9WgXcQ"'
    res = await music.music_play.handler({"query": "never gonna"})
    assert player.opened == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
    assert text(res) == "Playing never gonna on YouTube."


@pytest.mark.parametrize("html", [RuntimeError("offline"), "<html>no results</html>"])
async def test_youtube_failure_opens_results_page(youtube, player, html):
    youtube.html = html
    res = await music.music_play.handler({"query": "adele & co"})
    assert not res.get("is_error")
    assert player.opened == ["https://www.youtube.com/results?search_query=adele+%26+co"]
    assert text(res) == "I couldn't start a video, so I opened YouTube results for adele & co."


async def test_spotify_connected_check_failure_uses_youtube(youtube, player, monkeypatch):
    def broken():
        raise OSError("disk")
    monkeypatch.setattr(music, "_spotify_connected", broken)
    res = await music.music_play.handler({"query": "x"})
    assert "on YouTube" in text(res)


def test_open_only_web_and_spotify_uris(monkeypatch):
    opened = []
    monkeypatch.setattr(music.os, "startfile", opened.append, raising=False)
    music._open("https://www.youtube.com/watch?v=x")
    music._open("spotify:")
    for bad in ("C:\\Windows\\notepad.exe", "file:///c:/x", "ms-settings:"):
        with pytest.raises(ValueError):
            music._open(bad)
    assert opened == ["https://www.youtube.com/watch?v=x", "spotify:"]


def test_choose_device_skips_restricted():
    devs = [{"id": "a", "type": "Computer", "is_active": True, "is_restricted": True},
            {"id": "b", "type": "Speaker"}]
    assert music._choose_device(devs) is None


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
