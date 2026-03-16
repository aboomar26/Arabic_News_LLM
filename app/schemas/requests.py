# requests.py

from pydantic import BaseModel , Field

class ExtractionRequest(BaseModel):

    # /api/v1/extract

    story :str = Field(..., min_length= 10 , 
                       description = "The text of the Arabic article from which he wants to extract the details" )
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال..."
            }
        }



class TranslationRequest(BaseModel):

    # /api/v1/extract

    story :str = Field(..., min_length= 10 , 
                       description = "The text of the Arabic article you want to translate" )
    target_lang :str = Field(..., min_length= 10 ,
                        default="English", 
                       description = "The language you want to translate into" )
    
    
    class Config:
        json_schema_extra = {
            "example": {
                "story": "ذكرت مجلة فوربس أن العائلة تلعب دورا محوريا في تشكيل علاقة الأفراد بالمال...",
                 "target_lang": "English"
            }
        }