"""
prompt_builder.py
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

from app.schemas.responses import ExtractionResponse, TranslationResponse


_EXTRACTION_SYSTEM = "\n".join([
    "You are a professional NLP data parser.",
    "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
    "Do not generate any introduction or conclusion.",
])

_TRANSLATION_SYSTEM = "\n".join([
    "You are a professional Translator.",
    "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
    "Do not generate any introduction or conclusion.",
])


def _get_model_path() -> Path:
    """Project-root relative path to the model directory (portable)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "model"


class PromptBuilder:

    def __init__(self):
        model_path = _get_model_path().resolve()
        # local_files_only=True forces loading from disk; otherwise Windows paths can be mistaken for Hub repo_id
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )

        self._extraction_schema  = json.dumps(ExtractionResponse.model_json_schema(),  ensure_ascii=False)
        self._translation_schema = json.dumps(TranslationResponse.model_json_schema(), ensure_ascii=False)

    def _build_prompt(self, messages: list[dict]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_extraction_prompt(self, story: str) -> str:
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": "\n".join([
                "# Story:", story.strip(), "",
                "# Task:", "Extract the story details into a JSON.", "",
                "# Output Scheme:", self._extraction_schema, "",
                "# Output JSON:", "```json",
            ])},
        ]
        return self._build_prompt(messages)

    def build_translation_prompt(self, story: str, target_lang: str = "English") -> str:
        messages = [
            {"role": "system", "content": _TRANSLATION_SYSTEM},
            {"role": "user", "content": "\n".join([
                "# Story:", story.strip(), "",
                "# Task:", f"You have to translate the story content into {target_lang} associated with a title into a JSON.", "",
                "# Output Scheme:", self._translation_schema, "",
                "## target language:", target_lang, "",
                "# Output JSON:", "```json",
            ])},
        ]
        return self._build_prompt(messages)