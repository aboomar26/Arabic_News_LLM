# prompt_builder.py

from app.schemas.responses import ExtractionResponse, TranslationResponse
from app.config import settings


import json
from transformers import AutoTokenizer




# System prompts

_EXTRACTION_SYSTEM = "\n".join(["You are a professional NLP data parser.",
    "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
    "Do not generate any introduction or conclusion."]
)

_TRANSLATION_SYSTEM = "\n".join([
     "You are a professional Translator.",
    "Follow the provided `Task` by the user and the `Output Scheme` to generate the `Output JSON`.",
    "Do not generate any introduction or conclusion."
])

class PromptBuilder:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_ID)

        self._extraction_schema = json.dumps(ExtractionResponse.model_json_schema() , ensure_ascii=False)

        self._translation_schema = json.dumps(TranslationResponse.model_json_schema() , ensure_ascii=False)

    def _build_prompt(self , messages:list[dict]):

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt  = True
        )
    
    def build_extraction_prompt(self , story:str):

        messages = [

            {
                "role":"system",

                "content" : _EXTRACTION_SYSTEM
            },
            {
                "role":"user",

                "content" : "\n".join([
                    "## story:",story.strip(),"",

                    "# Task:",
                    "Extract the story details into a JSON.",
                    "",

                    "# Output Scheme:",
                    self._extraction_schema,
                    "",

                    "# Output JSON:",
                    "```json",
                ])
            }]
        return _build_prompt(messages)


    def build_translation_prompt(self , story:str, target_lang: str = settings.TARGET_LANG):

        messages = [

            {
                "role":"system",

                "content" : _TRANSLATION_SYSTEM
            },
            {
                "role":"user",

                "content" : "\n".join([
                    "## story:",story.strip(),"",

                    "# Task:",
                    "You have to translate the story content into {target_lang} associated with a title into a JSON.",
                    "",

                    "# Output Scheme:",
                    self._translation_schema,
                    "",

                    "## target language:",
                        target_lang,

                    "# Output JSON:",
                    "```json",
                ])
            }]
        
        return _build_prompt(messages)