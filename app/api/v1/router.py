# router.py


from fastapi import APIRouter
from app.api.v1 import extract, translate
 
router = APIRouter(prefix="/api/v1")
 
router.include_router(extract.router)
router.include_router(translate.router)