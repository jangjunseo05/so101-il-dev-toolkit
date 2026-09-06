"""로컬 전용 MJPEG 스트리밍 서버 (CLAUDE.md 44절).

`run_teleop_real.py`/`run_inference_mujoco.py`가 오프스크린으로 렌더링한
"씬"(3인칭) 카메라 프레임을, 브라우저(`dashboard/pages/1_데이터_수집.py`,
`4_추론.py`)에서 MuJoCo 네이티브 창을 보지 않고도 볼 수 있도록 로컬 HTTP로
스트리밍한다. 네이티브 창을 대체하는 게 아니라 추가하는 것 -- S/X 키 녹화
조작은 여전히 네이티브 창의 `key_callback`에 묶여 있고 이 모듈은 관여하지
않는다.

설계 전제(이 프로젝트 CLAUDE.md 29-6절과 동일): 서버와 브라우저가 항상 같은
물리 머신이라는 것. 인증이나 원격 접근을 고려하지 않고 127.0.0.1에만
바인딩한다. 이 전제가 바뀌면(원격/멀티유저 구조로 전환) 이 모듈은 재설계가
필요하다.

JPEG 인코딩은 메인 제어 루프가 아니라 각 브라우저 연결을 처리하는 HTTP
핸들러 스레드에서 수행한다 -- `publish_frame()`은 raw RGB numpy 배열을
저장만 하고 즉시 반환하므로, 제어 루프(50Hz)/녹화 캐던스(30Hz)의 타이밍에
영향을 주지 않는다. 아무도 스트림을 보고 있지 않으면 인코딩 자체가
일어나지 않는다(요청이 와야 인코딩).
"""

from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


class _LatestFrameStore:
    """스레드 세이프하게 "가장 최신 프레임 1장"만 들고 있는다(오래된 프레임을
    큐잉하지 않음 -- 스트리밍이 밀려도 항상 최신 상태만 보여주면 되고, 밀린
    프레임을 뒤늦게 다 보여줄 필요가 없는 모니터링 용도이기 때문)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


def _build_handler(store: _LatestFrameStore, target_fps: float):
    from PIL import Image

    min_interval = 1.0 / target_fps

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            # 각 스크립트가 stdout을 세션 로그 파일로 tee하고 있어(_setup_file_logging()),
            # HTTP 액세스 로그(요청마다 한 줄)로 그 로그가 도배되는 걸 막는다.
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    frame = store.get()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    buf = io.BytesIO()
                    Image.fromarray(frame).save(buf, format="JPEG", quality=70)
                    jpeg_bytes = buf.getvalue()
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode())
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(min_interval)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # 브라우저 탭을 닫거나 다른 페이지로 이동하면 정상적으로 발생함
                # (연결이 끊긴 소켓에 쓰기를 시도) -- 에러가 아니라 예상된 종료.
                pass

    return _Handler


class SceneStreamServer:
    """씬 카메라 프레임을 로컬 HTTP(MJPEG, `GET /stream`)로 스트리밍하는 서버.

    사용법::

        server = SceneStreamServer(port=8531)
        server.start()
        ...
        server.publish_frame(rgb_uint8_array)  # 매 프레임 호출, 인코딩 없이 즉시 반환
        ...
        server.stop()
    """

    def __init__(self, port: int, target_fps: float = 15.0) -> None:
        self._store = _LatestFrameStore()
        handler_cls = _build_handler(self._store, target_fps)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        self._thread: threading.Thread | None = None
        self.port = port

    def start(self) -> None:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def publish_frame(self, rgb_uint8_array: np.ndarray) -> None:
        self._store.set(rgb_uint8_array)

    def stop(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
