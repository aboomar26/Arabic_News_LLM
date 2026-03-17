# responses.py


from pydantic import BaseModel,Field
from typing import List , Literal , Optional



# Extraction response


storyCategory = Literal["politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science",
    "not_specified"]


storyEntities = Literal[  "person-male", "person-female", "location", "organization", "event", "time",
    "quantity", "money", "product", "law", "disease", "artifact", "not_specified"]


class Entity(BaseModel):
    entity_value :str = Field( description='the actual name or value of the entity')

    entity_type :storyEntities = Field(description='the type of recognized entity')


class ExtractionResponse(BaseModel):
    story_titel: str = Field(min_length=10 , max_length=100 ,
                                        description='A fully informative and SEO optimized titel of the story.')

    story_keywords: List[str] = Field(
        min_length=2,
        description='Relevant Keywords associated with the story.')

    story_sammary: List[str] = Field(
        min_length=2, max_length=5,
        description='Summarized key points about the story (2-5 points)')

    story_category: storyCategory = Field(
        description='category of the news story.')

    story_entities: List[Entity] = Field(
        min_length=1, max_length=10,
        description='List of identified entities in the story.')


    class Config:
        json_schema_extra = {
            "example": {
                "story_titel": "دور العائلة في تشكيل الشخصية المالية للأفراد",
                "story_keywords": ["المال", "العائلة", "الشخصية المالية"],
                "story_sammary": [
                    "العائلة تؤثر على علاقة الأفراد بالمال",
                    "هناك ثلاثة أبعاد رئيسية: الاكتساب والاستخدام والإدارة"
                ],
                "story_category": "economy",
                "story_entities": [
                    {"entity_value": "فوربس", "entity_type": "organization"}
                ]
            }
        }


# translation response

class TranslationResponse(BaseModel):
    translated_titel: str = Field(
        min_length=10, max_length=300,
        description="Suggested translated title of the news story.")

    translated_content : str = Field( min_length = 10 ,
                                   description = "translated content of the news story.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "translated_titel": "The Role of Family in Shaping Financial Personality",
                "translated_content": "Forbes magazine mentioned that the family plays..."
            }
        }


# error response 


class ErrorResponse(BaseModel):

    error : str
    details : Optional[str] = None


    class Config:
        json_schema_extra = {
            "example" : {
                "error" :  "Processing failed",
                 "detail": "Could not parse model output as JSON"
            }
        }

  