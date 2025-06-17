from pydantic import BaseModel


class SchemaMultipleChoiceRag(BaseModel):
    correct_statement: int
    reasoning: str
