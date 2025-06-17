from abc import ABC
from itertools import count
from time import sleep, thread_time
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from openai import RateLimitError, ContentFilterFinishReasonError

from hope.common.exceptions import ExternalLlmProviderException
from collections import defaultdict


class LanguageModelWrapperBase(ABC):

    call_counters = defaultdict(count)
    call_timer = defaultdict(lambda: 0.0)
    max_sleep = defaultdict(lambda: 0.0)

    def __init__(self, max_retries: int = 10, max_latency_seconds: float = 1):
        self.MAX_RETRIES = max_retries
        self.MAX_LATENCY_SECONDS = max_latency_seconds
        self._logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

    @property
    def counter(self):
        return self.call_counters[self.__class__.__name__]

    def reset_timing(self):
        self.call_timer[self.__class__.__name__] = 0.0

    def add_timing(self, timing: float):
        self.call_timer[self.__class__.__name__] += timing

    @property
    def total_time(self):
        return self.call_timer[self.__class__.__name__]

    @staticmethod
    def _wrapper(func):
        LOG_INTERVAL = 1000

        def inner(self: LanguageModelWrapperBase, *args, **kwargs):
            for n in range(self.MAX_RETRIES):
                try:
                    num_calls = next(self.counter)
                    t0 = thread_time()
                    res = func(self, *args, **kwargs)
                    dt = thread_time() - t0
                    self.add_timing(dt)
                    if num_calls % LOG_INTERVAL == 0 and num_calls > 0:
                        self._logger.debug(
                            f"Class {self.__class__.__name__} have performed {num_calls} calls. Average time: {(self.total_time / LOG_INTERVAL):.4f} seconds."
                        )
                        self.reset_timing()
                    if dt > self.MAX_LATENCY_SECONDS:
                        self._logger.warning(
                            f"Class {self.__class__.__name__} took {dt} seconds to complete a call."
                        )
                    break

                except RateLimitError as e:
                    SLEEP_TIME = [1, 30, 60, 120, 300][n]
                    if SLEEP_TIME > self.max_sleep[self.__class__.__name__]:
                        self._logger.info(
                            f"Rate limit exceeded. New sleep time: {SLEEP_TIME} seconds."
                        )
                        self.max_sleep[self.__class__.__name__] = SLEEP_TIME
                    sleep(SLEEP_TIME)

                except ContentFilterFinishReasonError as e:
                    self._logger.warning(
                        f"Content filter : {e} \n\t\t {args} \n\t\t {kwargs}"
                    )
                    res = None
                    # break

                except Exception as e:
                    self._logger.warning(
                        f"An error occurred while invoking LLM: {e}. Retrying..."
                    )
                    res = None
                    # break

            else:
                raise ExternalLlmProviderException(
                    f"Failed to invoke LLM. Max number of retries reached ({self.MAX_RETRIES})."
                )

            return res

        return inner


class LlmWrapper(LanguageModelWrapperBase):

    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        max_retries: int = 10,
        max_latency_seconds: float = 1,
    ):
        super().__init__(max_retries, max_latency_seconds)
        self.llm = llm

    @LanguageModelWrapperBase._wrapper
    def invoke(self, *args, **kwargs) -> str:
        temp = kwargs.pop("temperature", 0.0)
        out = self.llm.invoke(*args, temperature=temp, **kwargs)
        return out if isinstance(out, str) else out.content

    __call__ = invoke


class EmbeddingWrapper(LanguageModelWrapperBase):

    def __init__(
        self,
        embedding_model: Embeddings,
        max_retries: int = 10,
        max_latency_seconds: float = 1,
    ):
        super().__init__(max_retries, max_latency_seconds)
        self.embedding_model = embedding_model

    @LanguageModelWrapperBase._wrapper
    def embed_query(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text)

    @LanguageModelWrapperBase._wrapper
    def embed_documents(self, text: list[str]) -> list[list[float]]:
        return self.embedding_model.embed_documents(text)
