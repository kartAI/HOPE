from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path
from json import dumps
from typing import Any

import numpy as np
from hope.common.dataclasses import ChunkData
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from hope.common.exceptions import UnableToGenerateQuestionsException
from hope.utils.llm import EmbeddingWrapper, LlmWrapper


class BaseScorer(ABC):

    num_failed_evaluations_due_to_no_questions = count()

    def __init__(
        self,
        score_name: str,
        llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
    ):
        super().__init__()
        self._score_name = score_name
        self._llm: BaseChatModel | BaseLLM = llm
        self._embedding_model: Embeddings = embedding_model
        self._wrapped_embedding_model = EmbeddingWrapper(
            embedding_model, max_latency_seconds=0.1
        )
        self._wrapped_llm = LlmWrapper(llm, max_latency_seconds=0.1)

    @property
    def llm(self):
        return self._wrapped_llm

    @property
    def embedding_model(self):
        return self._wrapped_embedding_model

    @property
    def score_name(self):
        return self._score_name

    @abstractmethod
    def evaluate(self, document: Document, chunks: list[ChunkData]) -> float:
        pass

    @classmethod
    def post_process(cls, func) -> float | None:
        def wrapper(*args, **kwargs):
            try:
                score = func(*args, **kwargs)
                return np.clip(score, 0, 1)  # ** 2
            except UnableToGenerateQuestionsException:
                next(cls.num_failed_evaluations_due_to_no_questions)
                return None

        return wrapper

    def _save_sample(self, index: Any | None = None, **kwargs) -> Path:
        index = "" if index is None else f"_{index}"
        dst = Path(f"/workspaces/phd-HOPE/samples/{self.score_name}")
        dst.mkdir(parents=True, exist_ok=True)
        dst_file = dst / f"sample{index}.json"
        dst_file.write_text(dumps(kwargs, indent=4, ensure_ascii=False))
        return dst_file
