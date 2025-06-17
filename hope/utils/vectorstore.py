from time import sleep
from openai import RateLimitError, BadRequestError
from hope.common.dataclasses import ChunkData
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


from hope.common.exceptions import ExternalLlmProviderException


def rate_limit_wrapper(func):
    def inner(*args, **kwargs):
        for SLEEP_TIME in [1, 30, 60, 120, 300, 600, 1200]:
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                sleep(SLEEP_TIME)

        else:
            raise ExternalLlmProviderException("Rate limit exceeded.")

    return inner


def bad_request_wrapper(func):
    def inner(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except BadRequestError:
            for arg in args[0]:
                try:
                    func(self, [arg], **kwargs)
                except BadRequestError:
                    pass
            return None

    return inner


class ChromaVectorStore:
    def __init__(
        self,
        embedding_model: Embeddings,
        db_name: str = Chroma._LANGCHAIN_DEFAULT_COLLECTION_NAME,
    ):
        self._embeddings = embedding_model
        self._vector_store = Chroma(
            collection_name=db_name, embedding_function=self._embeddings
        )

    def add_data(self, data: list[str | Document | ChunkData]):
        sample = data[0]

        if isinstance(sample, str):
            self.add_texts(data)
        elif isinstance(sample, Document):
            self.add_documents(data)
        elif isinstance(sample, ChunkData):
            self.add_chunks(data)
        else:
            raise ValueError(f"Unsupported data type: {type(sample)}")

    @rate_limit_wrapper
    @bad_request_wrapper
    def add_documents(self, documents: list[Document]):
        self._vector_store.add_documents(documents)

    @rate_limit_wrapper
    @bad_request_wrapper
    def add_texts(self, texts: list[str]):
        self._vector_store.add_texts(texts)

    @rate_limit_wrapper
    @bad_request_wrapper
    def add_chunks(self, chunks: list[ChunkData]):
        self._vector_store.add_texts([chunk.text for chunk in chunks])

    def clear(self):
        self._vector_store.reset_collection()

    @rate_limit_wrapper
    def query(
        self, query_text: str, top_k: int = 5, as_string: bool = False
    ) -> list[Document | str]:
        """Queries the vector store for similar documents.

        query_text (str): The text to query the vector store with.
        top_k (int, optional): The number of top similar documents to return. Defaults to 5.
        as_string (bool, optional): If True, returns the results as a list of strings. Defaults to False.

        list[Document | str]: A list of the top similar documents or their content as strings.
        """
        results = self._vector_store.similarity_search(query_text, k=top_k)
        if as_string:
            return [result.page_content for result in results]
        return results
