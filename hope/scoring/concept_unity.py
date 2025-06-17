from concurrent.futures import ThreadPoolExecutor

import numpy as np
from hope.common.dataclasses import ChunkData
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from hope.common import hope_config
from hope.common.dataclasses import Question
from hope.generators import QuestionsGenerator
from hope.scoring.base import BaseScorer
from hope.utils.cosine_sim import cosine_similarity_embeddings


class ConceptUnityScorer(BaseScorer):
    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
    ):
        super().__init__(
            score_name="conceptualism", llm=llm, embedding_model=embedding_model
        )
        self._question_generator = QuestionsGenerator(llm, llm)

    def evaluate(self, document: Document, chunks: list[ChunkData]) -> float:

        with ThreadPoolExecutor() as executor:
            scores = [x for x in executor.map(self._evaluate_chunk, chunks) if x]

        return float(np.mean(scores))

    @BaseScorer.post_process
    def _evaluate_chunk(self, chunk: ChunkData) -> float | None:
        questions = self._generate_questions_from_text(chunk.text)

        embeddings = self._embed_questions(questions)
        average_embedding: np.ndarray = np.mean(embeddings, axis=0)

        cosine_similarities = [
            cosine_similarity_embeddings(embedding, average_embedding)
            for embedding in embeddings
        ]

        score = np.mean(cosine_similarities)

        return score

    def _generate_questions_from_text(self, text: str) -> list[Question]:
        return self._question_generator.generate(
            text,
            num_questions_to_generate=hope_config.scoring.conceptualism.num_questions,
        )

    def _embed_questions(self, questions: list[Question]) -> np.ndarray:

        assert all(
            isinstance(q, Question) for q in questions
        ), "questions must be of type Question"
        assert len(questions) > 0, "There must be at least one question"
        assert all(q.question for q in questions), "Questions text must not be empty"

        embeddings = [
            self.embedding_model.embed_query(question.question)
            for question in questions
        ]

        if any(embedding is None for embedding in embeddings):
            raise ValueError("Some embeddings are None")

        return np.array(embeddings)


if __name__ == "__main__":

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    generator = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )
    embedding_model = AzureOpenAIEmbeddings(
        model=hope_config.openai_embedding.deployment,
        api_key=hope_config.openai_embedding.api_key,
        azure_endpoint=hope_config.openai_embedding.endpoint,
        api_version=hope_config.openai_embedding.api_version,
    )

    # documents = DirectoryLoader(r"/workspaces/phd-HOPE/dummy_files").lazy_load()

    documents = [
        Document(
            page_content="Point out that the sky is blue and the grass is green is a common saying and is commonly used in literature and poetry."
        ),
        Document(
            page_content=(
                "The sky is blue and the grass is green. "
                + "In ancient times, people believed the Earth was flat. "
                + "Quantum mechanics is a fundamental theory in physics. "
                + "The Great Wall of China is one of the Seven Wonders of the World. "
                + "Artificial intelligence is transforming industries. "
                + "Dinosaurs roamed the Earth millions of years ago. "
            )
        ),
    ]

    scorer = ConceptUnityScorer(llm=generator, embedding_model=embedding_model)

    for d in documents:
        result = scorer.evaluate(d.page_content, [ChunkData(text=d.page_content)])
        print(f"Conceptualism score: {result}, type: {type(result)}")
