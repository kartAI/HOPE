from concurrent.futures import ThreadPoolExecutor
from copy import copy
from itertools import count

import numpy as np
from hope.common.dataclasses import ChunkData
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from hope.common import hope_config
from hope.common.dataclasses import Question
from hope.generators import QuestionsGenerator
from hope.predictors import LlmAnswererRag
from hope.scoring.base import BaseScorer
from hope.utils.cosine_sim import cosine_similarity_text
from hope.utils.vectorstore import ChromaVectorStore


class SemanticIndependenceScorer(BaseScorer):
    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
    ):
        super().__init__(
            score_name="semantic_independence",
            llm=llm,
            embedding_model=embedding_model,
        )
        self._predictor = LlmAnswererRag(llm)
        self._question_generator = QuestionsGenerator(llm, llm)

        self._num_chunks_evaluated = count()

    class EvaluationCollection:
        def __init__(
            self,
            chunks: list[ChunkData],
            focus_chunk: ChunkData,
            embedding_model: Embeddings,
        ):
            self.vector_store = ChromaVectorStore(
                embedding_model=embedding_model,
                db_name=f"semantic_independence_{id(focus_chunk)}_vector_store",
            )
            local_chunks = chunks.copy()
            local_chunks.remove(focus_chunk)
            self.vector_store.add_chunks(local_chunks)

            self.focus_chunk = focus_chunk

    @property
    def num_chunks_evaluated(self):
        return max(0.0, next(copy(self._num_chunks_evaluated)))

    def _evaluate_question_collection(
        self, question: Question, collection: EvaluationCollection
    ) -> float:
        relevant_data = collection.vector_store.query(question.question, top_k=3)

        ans_ref = self._predictor.answer_question(
            question.question, reference=[collection.focus_chunk.text]
        )
        ans_rag = self._predictor.answer_question(
            question.question,
            reference=[d.page_content for d in relevant_data]
            + [collection.focus_chunk.text],
        )

        if ans_ref is None or ans_rag is None:
            return 0.0

        if ans_ref == "" or ans_rag == "":
            return 0.0

        cos_sim = cosine_similarity_text(ans_ref, ans_rag, self.embedding_model)

        return cos_sim

    @BaseScorer.post_process
    def _evaluate_collection(self, collection: EvaluationCollection) -> float:

        questions = self._question_generator.generate(
            collection.focus_chunk.text,
            num_questions_to_generate=hope_config.scoring.semantic_dependence.num_questions,
        )

        cosine_similarities = [
            self._evaluate_question_collection(question, collection)
            for question in questions
        ]

        score = np.mean(cosine_similarities)

        return score

    def _thread_eval(self, focus_chunk: ChunkData) -> float | None:
        eval_collection = self.EvaluationCollection(
            self._chunks, focus_chunk, self._embedding_model
        )
        score = self._evaluate_collection(eval_collection)
        next(self._num_chunks_evaluated)
        return score

    def evaluate(self, document: Document, chunks: list[ChunkData]) -> float:

        self._chunks = chunks.copy()

        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(self._thread_eval, chunks))

        self._chunks = None

        scores = [x for x in scores if x]  # Remove None values

        if len(scores) == 0:
            return 0.0

        return np.mean(scores)


if __name__ == "__main__":
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

    scorer = SemanticIndependenceScorer2(generator, embedding_model)

    # scorer._vector_store.add_texts(
    #     [
    #         "Marvin the dog is not good. But he is loyal. He likes to play fetch and go for walks.",
    #         "Cats are independent and curious. They enjoy climbing and exploring their surroundings.",
    #         "Birds can be very social and intelligent. They often enjoy interacting with humans and other birds.",
    #     ]
    # )

    # scorer._evaluate_chunk(
    #     ChunkData(
    #         "All dogs are good and loyal. They like to play fetch and go for walks."
    #     )
    # )

    score = scorer.evaluate(
        "",
        [
            ChunkData(x)
            for x in [
                "All dogs are good and loyal. They like to play fetch and go for walks.",
                "Marvin the dog is not good. But he is loyal. He likes to play fetch and go for walks.",
                "Cats are independent and curious. They enjoy climbing and exploring their surroundings.",
                "Birds can be very social and intelligent. They often enjoy interacting with humans and other birds.",
            ]
        ],
    )

    print("Semantic dependent chunk", score)

    score = scorer.evaluate(
        "",
        [
            ChunkData(x)
            for x in [
                # "All dogs are good and loyal. They like to play fetch and go for walks.",
                "Marvin the dog is not good. But he is loyal. He likes to play fetch and go for walks.",
                "Cats are independent and curious. They enjoy climbing and exploring their surroundings.",
                "Birds can be very social and intelligent. They often enjoy interacting with humans and other birds.",
            ]
        ],
    )

    print("No semantic dependent chunk", score)
