from pydantic import BaseModel
from typing import List

from pydantic import BaseModel, Field

class Book(BaseModel):
    ISBN: str
    Book_Title: str = Field(alias="Book-Title")
    Book_Author: str = Field(alias="Book-Author")
    Year_Of_Publication: float = Field(alias="Year-Of-Publication")
    Publisher: str
    Description: str
    Categories: str
    Image: str
    rating: float

    class Config:
        populate_by_name = True

class SearchResponse(BaseModel):
    results: List[Book]
