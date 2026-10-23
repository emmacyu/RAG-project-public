"""The /upload endpoint must not block the event loop while a PDF is parsed and embedded."""
import asyncio
import io
import os
import sys
import time
from pathlib import Path

backend_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(backend_dir))


def test_upload_does_not_block_other_requests(monkeypatch, tmp_path):
    for key, value in {"OPENROUTER_API_KEY": "test", "DATABASE_URL": "postgresql://x", "REDIS_URL": "redis://x"}.items():
        monkeypatch.setenv(key, os.environ.get(key, value))
    import app.main as main
    from fastapi import UploadFile

    monkeypatch.chdir(tmp_path)                                    # the endpoint writes into ./data
    monkeypatch.setattr(main, "process_pdf", lambda path: time.sleep(0.5))  # stand-in for slow embedding

    async def scenario():
        finished = {}

        async def upload():
            upload_file = UploadFile(file=io.BytesIO(b"%PDF-1.7 test"), filename="doc.pdf")
            result = await main.upload_document(upload_file)
            finished["upload"] = time.perf_counter()
            return result

        async def other_request():                                  # 0.2 s of other work, e.g. a /chat call
            for _ in range(10):
                await asyncio.sleep(0.02)
            finished["other"] = time.perf_counter()

        result, _ = await asyncio.gather(upload(), other_request())
        return result, finished

    result, finished = asyncio.run(scenario())
    assert result["status"] == "success"
    # If the upload blocked the event loop, the other request could only run after it had finished.
    assert finished["other"] < finished["upload"]
