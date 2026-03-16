# main.py



import logging
from contextlib import asynccontextmanager
 
from fastapi import FastAPI
 
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
   
    # Startup
    logger.info("Starting up...")
 
    app.state.vllm_client    = VLLMClient()
    app.state.prompt_builder = PromptBuilder()   
    app.state.postprocessor  = PostProcessor()
 
    # vLLM check
    healthy = await app.state.vllm_client.health()
    if healthy:
        logger.info("vLLM server ✓ متصل")
    else:
        logger.warning("vLLM server ✗ مش شغال — تأكد إنك شغّلت start_vllm.sh")
 
    yield 
 
    # Shutdown 
    logger.info("Shutting down...")
    await app.state.vllm_client.close()
 
 
app = FastAPI(
    title="Arabic News NLP API",
    description="API for extracting and translating details of Arabic articles using Qwen2.5 fine-tuned",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.include_router(router)
 
 
@app.get("/health", tags=["health"])
async def health():

    vllm_ok = await app.state.vllm_client.health()
    return {
        "api": "ok",
        "vllm": "ok" if vllm_ok else "down",
    }