# vllm_client.py
from app.config import settings


import httpx
import logging
from typing import Optional



logger = logging.getLogger(__name__)

class VLLMClientError(Exception):
    pass



class VLLMClient:

    def __init__(self):
        self._client = httpx.AsyncClient(timeout = settings.REQUEST_TIMEOUT)

    async def complete(self , prompt:str , max_tokens: Optional[int] = None , temperature: Optional[float] = None) :

        payload = {
            "model": settings.VLLM_MODEL_ID,
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens is not None else settings.MAX_TOKENS,
            "temperature": temperature if temperature is not None else settings.TEMPERATURE,
        }

        try:
            response = await self._client.post(
                 f"{settings.VLLM_BASE_URL}/v1/completions",
                    json = payload,
            )

            response.raise_for_status()

        except httpx.TimeoutException:
            raise VLLMClientError(
                 f"vLLM timeout {settings.REQUEST_TIMEOUT}"
            )
        
        except httpx.HTTPStatusError as e:
             raise VLLMClientError(
                 f"{e.response.status_code}: {e.response.text}"
                 f"error : {e}"
            )
        
        except httpx.RequestError as e:
             raise VLLMClientError(
                 f"vLLM cant reach {settings.VLLM_BASE_URL}"
                 f"error : {e}"
            )
        
        data = response.json()

        try :
            text = data["choices"][0]["text"]
        
        except (KeyError , IndexError) as e:
            raise VLLMClientError(f"error in response {data} \n{e}")
        
        logger.debug(
           "vLLM \"Done\" | tokens=%s | preview=%s",
            data.get("usage", {}).get("total_tokens", "?"),
            text[:80].replace("\n", " "),
        )


        return text
    
    async def health(self):
        try:
            resp = await self._client.get(
                f"{settings.VLLM_BASE_URL}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
        
    async def close(self):

        await self._client.aclose()



        



        
