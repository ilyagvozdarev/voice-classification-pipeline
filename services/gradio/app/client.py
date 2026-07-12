import requests

from .config import settings


class MasterClient:
    """Blocking HTTP client for the master orchestrator.

    Gradio runs event handlers in a threadpool, so a synchronous client keeps
    the front-end code simple without blocking the UI.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.master_url).rstrip("/")

    def process(self, audio_path: str) -> dict:
        with open(audio_path, "rb") as fh:
            files = {"audio": (audio_path.rsplit("/", 1)[-1], fh, "application/octet-stream")}
            resp = requests.post(
                f"{self.base_url}/process",
                files=files,
                timeout=settings.request_timeout_s,
            )
        resp.raise_for_status()
        return resp.json()


master_client = MasterClient()



# files = {"audio": (audio_path.rsplit("/", 1)[-1], fh, "application/octet-stream")}:
#     "audio" — имя поля формы (именно его master ждёт: audio: UploadFile = File(...) в master/app/main.py);
#     fh — открытый файловый объект (читается как тело);
#     "application/octet-stream" — MIME-тип.
