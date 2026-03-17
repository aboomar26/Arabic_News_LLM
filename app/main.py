"""
main.py - FastAPI application entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router
from app.services.vllm_client import VLLMClient
from app.services.prompt_builder import PromptBuilder
from app.services.postprocessor import PostProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    app.state.vllm_client    = VLLMClient()
    app.state.prompt_builder = PromptBuilder()
    app.state.postprocessor  = PostProcessor()

    healthy = await app.state.vllm_client.health()
    if healthy:
        logger.info("vLLM server connected")
    else:
        logger.warning("vLLM server not working; check scripts/start_vllm.bat (or start_vllm.sh on Linux)")

    yield

    logger.info("Shutting down...")
    await app.state.vllm_client.close()


app = FastAPI(
    title="Arabic News NLP API",
    description="API for extracting details and translating Arabic news articles using fine-tuned Qwen2.5",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["health"])
async def health():
    """Check if API and vLLM are running."""
    vllm_ok = await app.state.vllm_client.health()
    return {
        "api": "ok",
        "vllm": "ok" if vllm_ok else "down",
    }