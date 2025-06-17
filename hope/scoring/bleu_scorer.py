from hope.scoring.base import BaseScorer
from nltk.translate.bleu_score import sentence_bleu
from langchain_core.documents import Document
from hope.common.dataclasses import ChunkData


class BLEUScorer(BaseScorer):
    def __init__(self):
        super().__init__(score_name="BLEU", llm=None, embedding_model=None)

    def evaluate(self, document: Document, chunks: list[ChunkData]):
        return sentence_bleu(
            [document.page_content], "\n".join([chunk.text for chunk in chunks])
        )


if __name__ == "__main__":
    scorer = BLEUScorer()
    print(scorer.evaluate(Document("This is a test"), [ChunkData("This is a test")]))
    print(
        scorer.evaluate(
            Document("This is a test"),
            [ChunkData("This is a test"), ChunkData("This is a test")],
        )
    )
    print(
        scorer.evaluate(
            Document("This is a test"),
            [
                ChunkData("This is a test"),
                ChunkData("This is a test"),
                ChunkData("This is a test"),
            ],
        )
    )
    print(
        scorer.evaluate(
            Document("This is a test"),
            [
                ChunkData("This is a test"),
                ChunkData("This is a test"),
                ChunkData("This is a test"),
                ChunkData("This is a test"),
            ],
        )
    )
