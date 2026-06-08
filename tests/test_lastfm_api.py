from spotify_splitter import lastfm_api


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
