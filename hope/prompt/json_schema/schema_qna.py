from pydantic import BaseModel
from typing import List


class QnA(BaseModel):
    question: str
    answer: str


class SchemaQnaGenerator(BaseModel):
    questions: List[QnA]


class SchemaQnaCritic(BaseModel):
    is_relevant: bool
    reason: str


if __name__ == "__main__":
    qna = SchemaQnaGenerator(
        questions=[
            QnA(question="What is the capital of France?", answer="Paris"),
            QnA(question="What is the capital of Germany?", answer="Berlin"),
            QnA(question="What is the capital of Italy?", answer="Rome"),
        ]
    )
    # print(qna.model_dump_json(indent=2))

    print(SchemaQnaCritic.model_json_schema())
