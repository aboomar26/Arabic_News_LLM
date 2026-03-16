# extract.py


import logging
from fastapi import APIRouter,HTTPException ,Request

from app.schemas.requests import ExtractionRequest
from app.schemas.responses import ExtractionResponse
from app.services.vllm_client import VLLMClientError
from app.services.postprocessor import PostProcessingError


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="extract news sammary",
    description = "It takes the text of an Arabic article including the title, keywords, abstract, classification, and entities."
)

async def extract(body:ExtractionRequest , request: Request):

    prompt_builder = request.app.state.prompt_builder
    vllm_client    = request.app.state.vllm_client
    postprocessor  = request.app.state.postprocessor


    # prompt
    prompt = prompt_builder.build_extraction_prompt(body.story)

    #vllm
    try:
        raw_text = await vllm_client.complete(prompt)
    except VLLMClientError as e:
        logger.error("vLLM error in extraction: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    
    #parse
    try:
        result = postprocessor.process(raw_text)
    except PostProcessingError as e:
        logger.error("postprocessor error in extraction: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    #response
    return ExtractionResponse(**result)


