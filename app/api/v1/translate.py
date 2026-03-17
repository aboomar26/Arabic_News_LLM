# translate.py


import logging
from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import TranslationRequest
from app.schemas.responses import TranslationResponse
from app.services.vllm_client import VLLMClientError
from app.services.postprocessor import PostProcessingError


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="translate news summary",
    description="Takes the text of an Arabic article and returns the title and content translated.",
)
async def translate(body: TranslationRequest, request: Request):
    prompt_builder = request.app.state.prompt_builder
    vllm_client = request.app.state.vllm_client
    postprocessor = request.app.state.postprocessor

    prompt = prompt_builder.build_translation_prompt(body.story, body.target_lang)

    try:
        raw_text = await vllm_client.complete(prompt)
    except VLLMClientError as e:
        logger.error("vLLM error in translation: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    try:
        result = postprocessor.process(raw_text)
    except PostProcessingError as e:
        logger.error("postprocessor error in translation: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    return TranslationResponse(**result)
