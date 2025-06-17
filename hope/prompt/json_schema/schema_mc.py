from pydantic import BaseModel
from typing import List


class SchemaMultipleChoice(BaseModel):
    derived_statement: str
    false_statements: List[str]

    def __str__(self):
        NLT = "\n\t"
        return f"Derived Statement: \n\t{self.derived_statement}\nFalse Statements: {NLT}{NLT.join(self.false_statements)}"
