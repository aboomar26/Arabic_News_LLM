import re
import json
import logging
from typing import Any, Optional

import json_repair

logger = logging.getLogger(__name__)

_CJK_PATTERN = re.compile(
    "["
    "\u4E00-\u9FFF"
    "\u3400-\u4DBF"
    "\uF900-\uFAFF"
    "]"
)

_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)


class PostProcessingError(Exception):
    pass


class PostProcessor:

    def strip_chinese(self, text: str) -> str:
        cleaned = _CJK_PATTERN.sub("", text)
        if cleaned != text:
            logger.warning("شال %d حرف صيني", len(text) - len(cleaned))
        return cleaned

    def strip_code_fences(self, text: str) -> str:
        return _CODE_FENCE_PATTERN.sub("", text).strip()

    def parse_json(self, text: str) -> Optional[Any]:
        try:
            return json_repair.loads(text)
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return None

    def process(self, raw_text: str) -> dict:
        text = self.strip_chinese(raw_text)
        text = self.strip_code_fences(text)
        result = self.parse_json(text)
        if result is None:
            raise PostProcessingError(f"مش قادر يعمل parse: {raw_text[:200]}")
        if not isinstance(result, dict):
            raise PostProcessingError(f"الـ output مش dict")
        return result