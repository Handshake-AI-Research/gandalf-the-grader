"""Gandalf's file-editor adapter for provider-sized image observations."""

from collections.abc import Sequence

from litellm import get_llm_provider
from litellm.exceptions import BadRequestError
from openhands.sdk import LLM
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.llm import ImageContent
from openhands.sdk.tool import ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorAction, FileEditorObservation, FileEditorTool
from PIL import Image

from gandalf.image_inputs import prepare_openai_image_url

GANDALF_FILE_EDITOR_TOOL = "gandalf_file_editor"


class OpenAiImageFileEditorExecutor(ToolExecutor[FileEditorAction, FileEditorObservation]):
    def __init__(self, delegate: ToolExecutor[FileEditorAction, FileEditorObservation]) -> None:
        self.delegate = delegate

    def __call__(
        self, action: FileEditorAction, conversation: LocalConversation | None = None
    ) -> FileEditorObservation:
        observation = self.delegate(action, conversation)
        if action.command != "view" or observation.is_error:
            return observation
        try:
            content = [
                part.model_copy(update={"image_urls": [prepare_openai_image_url(url) for url in part.image_urls]})
                if isinstance(part, ImageContent)
                else part
                for part in observation.content
            ]
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            return FileEditorObservation.from_text(
                text=f"Cannot prepare image for the OpenAI judge: {error}",
                command=action.command,
                path=observation.path,
                prev_exist=observation.prev_exist,
                is_error=True,
            )
        return observation.model_copy(update={"content": content})

    def close(self) -> None:
        self.delegate.close()


def _uses_openai_image_limit(llm: LLM) -> bool:
    if llm.model.lower().startswith("openrouter/openai/"):
        return True
    try:
        _, provider, _, _ = get_llm_provider(model=llm.model, api_base=llm.base_url)
    except BadRequestError:
        return False
    return bool(provider == "openai")


def create_gandalf_file_editor(conv_state: ConversationState) -> Sequence[FileEditorTool]:
    tools = FileEditorTool.create(conv_state)
    if not _uses_openai_image_limit(conv_state.agent.llm):
        return tools
    return [tool.set_executor(OpenAiImageFileEditorExecutor(tool.as_executable().executor)) for tool in tools]


register_tool(GANDALF_FILE_EDITOR_TOOL, create_gandalf_file_editor)
