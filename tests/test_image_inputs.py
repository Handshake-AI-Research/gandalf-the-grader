"""Image observations must fit the judge provider without changing evidence files."""

import base64
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import resolve_tool
from openhands.sdk.workspace import LocalWorkspace
from openhands.tools.file_editor import FileEditorAction, FileEditorObservation
from PIL import Image

from gandalf.file_editor import GANDALF_FILE_EDITOR_TOOL, _uses_openai_image_limit
from gandalf.image_inputs import _fit_size, prepare_openai_image_url
from gandalf.judge import run_agent_session


@pytest.mark.parametrize("model", ["openai/gpt-4.1", "gpt-4.1"])
def test_judge_file_view_fits_openai_patch_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    source = tmp_path / "large-chart.png"
    Image.new("RGB", (5100, 6000), "white").save(source)
    original = source.read_bytes()
    returned_urls: list[str] = []

    class LocalToolSession:
        """Execute the real configured file tool without making a model request."""

        def __init__(self, agent: Agent, workspace: str) -> None:
            self.state = ConversationState(
                id=uuid4(), agent=agent, workspace=LocalWorkspace(working_dir=workspace), persistence_dir=None
            )

        def send_message(self, _prompt: str) -> None:
            pass

        def run(self) -> None:
            spec = next(tool for tool in self.state.agent.tools if "file_editor" in tool.name)
            tool = resolve_tool(spec, self.state)[0]
            observation = tool(FileEditorAction(command="view", path=str(source)))
            assert not observation.is_error
            returned_urls.extend(
                url for part in observation.content if isinstance(part, ImageContent) for url in part.image_urls
            )

    monkeypatch.setenv("LLM_API_KEY", "unused-test-key")
    # run_agent_session sets HOME; restore it after the test.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("gandalf.judge.Conversation", LocalToolSession)
    run_agent_session(model, [], str(tmp_path), "Inspect the chart")

    assert len(returned_urls) == 1
    header, encoded = returned_urls[0].split(",", 1)
    assert header == "data:image/png;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as prepared:
        assert prepared.size == (5088, 5987)
    assert source.read_bytes() == original


def view_image(path: Path, model: str = "openai/gpt-4.1") -> FileEditorObservation:
    llm = LLM(model=model, api_key="unused-test-key")
    state = ConversationState(
        id=uuid4(),
        agent=Agent(llm=llm, tools=[Tool(name=GANDALF_FILE_EDITOR_TOOL)]),
        workspace=LocalWorkspace(working_dir=str(path.parent)),
        persistence_dir=None,
    )
    tool = resolve_tool(Tool(name=GANDALF_FILE_EDITOR_TOOL), state)[0]
    assert tool.name == "file_editor"
    result = tool(FileEditorAction(command="view", path=str(path)))
    assert isinstance(result, FileEditorObservation)
    return result


def image_payload(observation: FileEditorObservation) -> tuple[str, bytes]:
    assert not observation.is_error
    part = next(part for part in observation.content if isinstance(part, ImageContent))
    assert len(part.image_urls) == 1
    header, encoded = part.image_urls[0].split(",", 1)
    return header, base64.b64decode(encoded)


@pytest.mark.parametrize("model", ["openai/gpt-4.1", "openrouter/openai/gpt-4.1"])
def test_oversized_alpha_image_keeps_transparency_and_observation_metadata(tmp_path: Path, model: str) -> None:
    source = tmp_path / "chart.png"
    Image.new("RGBA", (5100, 6000), (20, 40, 60, 78)).save(source)
    original = source.read_bytes()
    observation = view_image(source, model)
    header, prepared = image_payload(observation)
    assert header == "data:image/png;base64"
    assert observation.path == str(source)
    assert observation.prev_exist
    assert any(isinstance(part, TextContent) and "read successfully" in part.text for part in observation.content)
    with Image.open(BytesIO(prepared)) as result:
        assert result.size == (5088, 5987)
        assert result.mode == "RGBA"
        pixel = result.getpixel((100, 100))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 78
    assert source.read_bytes() == original


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-flash"])
def test_non_openai_observation_keeps_oversized_bytes(tmp_path: Path, model: str) -> None:
    source = tmp_path / "chart.png"
    Image.new("RGB", (5100, 6000)).save(source)
    original = source.read_bytes()
    _, prepared = image_payload(view_image(source, model))
    assert prepared == original
    assert source.read_bytes() == original


@pytest.mark.parametrize("size", [(32, 32), (4800, 6400)])
def test_compliant_image_including_exact_patch_limit_is_byte_identical(tmp_path: Path, size: tuple[int, int]) -> None:
    source = tmp_path / "chart.png"
    Image.new("RGB", size).save(source)
    original = source.read_bytes()
    _, prepared = image_payload(view_image(source))
    assert prepared == original


def test_oversized_jpeg_stays_jpeg_and_applies_exif_orientation(tmp_path: Path) -> None:
    source = tmp_path / "chart.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (6000, 5100), "red").save(source, exif=exif)
    original = source.read_bytes()
    header, prepared = image_payload(view_image(source))
    assert header == "data:image/jpeg;base64"
    with Image.open(BytesIO(prepared)) as result:
        assert result.format == "JPEG"
        assert result.size == (5088, 5987)
        assert result.getexif().get(274, 1) == 1
    assert source.read_bytes() == original


def test_oversized_palette_transparency_survives_resize(tmp_path: Path) -> None:
    source = tmp_path / "chart.png"
    image = Image.new("P", (5100, 6000))
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.save(source, transparency=0)
    _, prepared = image_payload(view_image(source))
    with Image.open(BytesIO(prepared)) as result:
        assert result.size == (5088, 5987)
        pixel = result.convert("RGBA").getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 0


def test_oversized_animation_returns_tool_error_without_flattening(tmp_path: Path) -> None:
    source = tmp_path / "animation.gif"
    first = Image.new("P", (5100, 6000), 0)
    second = Image.new("P", (5100, 6000), 1)
    first.save(source, save_all=True, append_images=[second], duration=100, loop=0)
    original = source.read_bytes()
    observation = view_image(source)
    assert observation.is_error
    assert not any(isinstance(part, ImageContent) for part in observation.content)
    assert "animated image" in observation.text
    assert source.read_bytes() == original


def test_corrupt_image_returns_tool_error(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.png"
    source.write_bytes(b"not an image")
    observation = view_image(source)
    assert observation.is_error
    assert not any(isinstance(part, ImageContent) for part in observation.content)
    assert source.read_bytes() == b"not an image"


def test_decoder_resource_limit_becomes_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "chart.png"
    Image.new("RGB", (32, 32)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    observation = view_image(source)
    assert observation.is_error
    assert not any(isinstance(part, ImageContent) for part in observation.content)


def test_text_edits_and_undo_keep_upstream_behavior(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    state = ConversationState(
        id=uuid4(),
        agent=Agent(llm=LLM(model="openai/gpt-4.1", api_key="unused-test-key")),
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
        persistence_dir=None,
    )
    tool = resolve_tool(Tool(name=GANDALF_FILE_EDITOR_TOOL), state)[0]
    tool(FileEditorAction(command="create", path=str(source), file_text="before"))
    tool(FileEditorAction(command="str_replace", path=str(source), old_str="before", new_str="after"))
    assert source.read_text() == "after"
    tool(FileEditorAction(command="undo_edit", path=str(source)))
    assert source.read_text() == "before"
    assert not tool(FileEditorAction(command="view", path=str(source))).is_error


def test_remote_image_url_is_not_fetched() -> None:
    url = "https://example.invalid/chart.png"
    assert prepare_openai_image_url(url) == url


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(5100, 6000, (5088, 5987)), (6000, 5100, (5987, 5088)), (1, 1_000_000, (1, 960_000))],
)
def test_fitting_preserves_aspect_ratio_with_patch_rounding(width: int, height: int, expected: tuple[int, int]) -> None:
    assert _fit_size(width, height) == expected


def test_unknown_provider_keeps_upstream_behavior() -> None:
    llm = LLM(model="custom-unknown-model", api_key="unused-test-key")
    assert not _uses_openai_image_limit(llm)
