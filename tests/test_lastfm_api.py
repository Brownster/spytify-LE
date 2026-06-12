from spotify_splitter import lastfm_api


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_get_lastfm_client_reuses_same_api_key(monkeypatch):
    monkeypatch.setattr(lastfm_api, "_global_lastfm_client", None)
    monkeypatch.setattr(lastfm_api, "_global_lastfm_api_key", None)

    first = lastfm_api.get_lastfm_client(api_key="key-1")
    second = lastfm_api.get_lastfm_client(api_key="key-1")

    assert second is first


def test_get_lastfm_client_recreates_when_api_key_changes(monkeypatch):
    monkeypatch.setattr(lastfm_api, "_global_lastfm_client", None)
    monkeypatch.setattr(lastfm_api, "_global_lastfm_api_key", None)

    first = lastfm_api.get_lastfm_client(api_key="key-1")
    second = lastfm_api.get_lastfm_client(api_key="key-2")

    assert second is not first
    assert second.api_key == "key-2"


def test_track_top_tags_uses_track_method_and_filters_noise(monkeypatch):
    client = lastfm_api.LastFMAPI(api_key="key")
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse({
            "toptags": {
                "tag": [
                    {"name": "seen live"},
                    {"name": "shoegaze"},
                    {"name": "dream pop"},
                ]
            }
        })

    monkeypatch.setattr(client.session, "get", fake_get)

    assert client._get_track_top_tags("Artist", "Title") == ["shoegaze", "dream pop"]
    assert calls[0][1]["method"] == "track.getTopTags"
    assert calls[0][1]["track"] == "Title"
    assert calls[0][2] == 10


def test_artist_top_tags_uses_artist_method_without_track(monkeypatch):
    client = lastfm_api.LastFMAPI(api_key="key")
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return FakeResponse({"toptags": {"tag": {"name": "soul"}}})

    monkeypatch.setattr(client.session, "get", fake_get)

    assert client._get_artist_top_tags("Artist") == ["soul"]
    assert calls[0][1]["method"] == "artist.getTopTags"
    assert "track" not in calls[0][1]
