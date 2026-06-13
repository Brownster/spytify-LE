from queue import Queue
from unittest.mock import Mock, patch

import numpy as np

from spotify_splitter.audio import AudioStream


def test_callback_counts_dropped_frames_on_full_queue():
    ui_callback = Mock()
    q = Queue(maxsize=1)
    q.put_nowait(np.zeros((16, 2), dtype=np.float32))

    with patch("spotify_splitter.audio.sd") as mock_sd:
        mock_sd.InputStream.return_value = Mock()
        stream = AudioStream("monitor", q=q, ui_callback=ui_callback)

    stream._callback(np.ones((32, 2), dtype=np.float32), 32, Mock(), None)

    assert stream.dropped_frames == 32
    ui_callback.assert_called_once_with("buffer_warning", {"dropped_frames": 32})


def test_callback_status_warning_does_not_count_as_dropped_frames():
    ui_callback = Mock()

    with patch("spotify_splitter.audio.sd") as mock_sd:
        mock_sd.InputStream.return_value = Mock()
        stream = AudioStream("monitor", q=Queue(maxsize=2), ui_callback=ui_callback)

    stream._callback(np.ones((32, 2), dtype=np.float32), 32, Mock(), "overflow")

    assert stream.dropped_frames == 0
    ui_callback.assert_called_once_with("buffer_warning", None)
