"""Tests for gandalf_grader.trajectory."""

import json
import os

import pytest

from gandalf_grader.trajectory import load_trajectory_final_output

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadTrajectoryFinalOutput:
    def test_extracts_final_output(self):
        result = load_trajectory_final_output(os.path.join(FIXTURES, "sample_trajectory.json"))
        assert result == "Done! I created index.html with a Hello World page."

    def test_skips_tool_call_messages(self):
        result = load_trajectory_final_output(os.path.join(FIXTURES, "sample_trajectory.json"))
        # The second agent message has tool_calls, so final_output should be the third
        assert "I'll create the file now" not in result

    def test_empty_steps(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"steps": []}))
        assert load_trajectory_final_output(str(p)) == ""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_trajectory_final_output("/nonexistent/trajectory.json")

    def test_skips_trailing_empty_message(self, tmp_path):
        """When the last agent message has empty content (e.g. reasoning-only
        turn), the function should return the preceding non-empty message."""
        p = tmp_path / "trailing_empty.json"
        p.write_text(json.dumps({
            "steps": [
                {"source": "user", "message": "Do the task"},
                {"source": "agent", "message": "Here is the result."},
                {"source": "agent", "message": ""},
            ]
        }))
        assert load_trajectory_final_output(str(p)) == "Here is the result."

    def test_skips_trailing_whitespace_only_message(self, tmp_path):
        p = tmp_path / "trailing_ws.json"
        p.write_text(json.dumps({
            "steps": [
                {"source": "user", "message": "Do the task"},
                {"source": "agent", "message": "Here is the result."},
                {"source": "agent", "message": "   \n  "},
            ]
        }))
        assert load_trajectory_final_output(str(p)) == "Here is the result."

    def test_all_agent_messages_empty(self, tmp_path):
        p = tmp_path / "all_empty.json"
        p.write_text(json.dumps({
            "steps": [
                {"source": "user", "message": "Do the task"},
                {"source": "agent", "message": ""},
                {"source": "agent", "message": ""},
            ]
        }))
        assert load_trajectory_final_output(str(p)) == ""

    def test_no_agent_messages(self, tmp_path):
        p = tmp_path / "user_only.json"
        p.write_text(json.dumps({
            "steps": [{"source": "user", "message": "hello"}]
        }))
        assert load_trajectory_final_output(str(p)) == ""
