import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from gandalf.judge import create_llm, run_judge_batch


def test_gpt61_writes_verdict_through_responses_with_high_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[dict[str, Any]] = []
    verdict_path = tmp_path / "verdict.json"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": body})
            if len(requests) == 1:
                name = "file_editor"
                arguments = {
                    "command": "create",
                    "path": str(verdict_path),
                    "file_text": json.dumps(
                        [{"index": 0, "met": True, "reasoning": "The file exists.", "evidence": []}]
                    ),
                }
            else:
                name = "finish"
                arguments = {"message": "Verdict written."}
            response = {
                "id": f"resp-{len(requests)}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "output": [
                    {
                        "id": f"fc-{len(requests)}",
                        "type": "function_call",
                        "call_id": f"call-{len(requests)}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
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
        monkeypatch.setenv("HOME", str(tmp_path))
        assert create_llm("openai/gpt-6.1-sol").uses_responses_api()
        monkeypatch.setattr("gandalf.judge.make_verdict_path", lambda **_kwargs: str(verdict_path))
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "model": "openai/gpt-6.1-sol",
                    "instructions": "Create the required file.",
                    "final_output": "Done.",
                    "criteria": ["The file exists."],
                    "workdir": str(tmp_path),
                }
            )
        )
        run_judge_batch(str(input_path), str(output_path))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    result = json.loads(output_path.read_text())
    assert result["verdicts"] == [{"met": True, "reasoning": "The file exists.", "evidence": []}]
    assert result["llm_usage"]["prompt_tokens"] == 20
    assert result["llm_usage"]["completion_tokens"] == 20
    assert len(requests) == 2
    for request in requests:
        assert request["path"] == "/v1/responses"
        assert request["body"]["model"] == "gpt-6.1-sol"
        assert request["body"]["reasoning"] == {"effort": "high"}
        assert not {"temperature", "top_p", "presence_penalty", "frequency_penalty"} & request["body"].keys()
    assert any(item.get("type") == "function_call_output" for item in requests[1]["body"]["input"])
