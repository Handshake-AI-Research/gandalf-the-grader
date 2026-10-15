import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from openhands.sdk.llm import Message, TextContent

from gandalf.judge import create_llm


def test_gpt61_requests_responses_with_high_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body})
            response = {
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("USE_LITELLM_PROXY", "false")
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        llm = create_llm("openai/gpt-6.1-sol")
        llm.num_retries = 0
        assert llm.uses_responses_api()
        llm.responses(messages=[Message(role="user", content=[TextContent(text="hello")])])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert requests[0]["path"] == "/v1/responses"
    assert requests[0]["body"]["model"] == "gpt-6.1-sol"
    assert requests[0]["body"]["reasoning"] == {"effort": "high"}
    assert "temperature" not in requests[0]["body"]
    assert "top_p" not in requests[0]["body"]
